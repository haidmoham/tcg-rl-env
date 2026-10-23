"""Qualify copied teacher/student frames on recorded native ordered prompts."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    started = time.monotonic()
    bindings = json.loads(args.bindings.read_text())
    for path, expected in bindings["artifacts"].items():
        if sha(path) != expected:
            raise ValueError(f"bound artifact changed: {path}")
    artifacts_digest = hashlib.sha256(json.dumps(
        bindings["artifacts"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if artifacts_digest != bindings["artifact_manifest_sha256"]:
        raise ValueError("artifact map digest mismatch")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    os.environ["FASTENC_LIB"] = bindings["engine_library"]
    os.nice(max(0, 19 - os.nice(0)))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    from tcg.native_policy import (
        FrameCapture, SmallNativeConfig, build_student, configure_mew,
        load_student, load_teacher, model_identity, save_policy, score_policy, state_digest,
    )
    configure_mew(bindings["mew_root"])
    from fastenc.games import Games
    protocol = json.loads(args.protocol.read_text())
    teacher = load_teacher(bindings["teacher_checkpoint"], expected_sha256=bindings["teacher_sha256"])
    student = build_student(SmallNativeConfig(**protocol["student"]), seed=protocol["seeds"]["student_init"])
    teacher_before, student_before = state_digest(teacher), state_digest(student)
    checkpoint = args.output / "untrained-student.pt"
    checkpoint_sha = save_policy(student, checkpoint)
    reloaded = load_student(checkpoint, expected_sha256=checkpoint_sha)
    helper = ctypes.CDLL(bindings["state_probe_library"])
    helper.GateBegin.argtypes = [ctypes.c_void_p]
    helper.GateRestore.argtypes = []
    helper.GateLastRestore.restype = ctypes.c_char_p
    decks = json.loads(Path(bindings["deck_manifest"]).read_text())["decks"]
    cases = []
    for name, source in bindings["ordered_replays"].items():
        recorded = json.loads(Path(source).read_text())
        found = recorded["discovery"]
        deck = list(decks[1]["card_ids"])
        if name == "kofu":
            deck = [1200 if card == 1188 else card for card in deck]
        seed = found["seed"]
        with Games([[deck, deck]], seed=seed, seeds=[seed], threads=1) as game:
            game.set_external(0, 3)
            for index, action in enumerate(found["prefix"]):
                wire = game.observe(0)
                expected = found["frames"][index]
                if ({k: v for k, v in wire.items() if k != "search_begin_input"}
                        != {k: v for k, v in expected.items() if k != "search_begin_input"}
                        or game.snapshot(0) != found["snapshots"][index]):
                    raise ValueError("ordered discovery replay changed")
                game.act_external(0, action)
            # Consume the recorded actor observation before switching to native columns.
            wire = game.observe(0)
            if game.snapshot(0) != found["snapshots"][-1]:
                raise ValueError("ordered prompt snapshot changed")
            game.set_external(0, 0)
            capture = FrameCapture()
            for prefix_len in (0, 1):
                with game.encode() as encoded:
                    frame = capture.capture(encoded)
                record = frame.records[0]
                if record.prefix_len != prefix_len or not record.ordered:
                    raise ValueError("ordered native prefix not reached")
                snapshot, stats = game.snapshot(0), game.stats()
                raw_guard = prefix_len == 0
                if raw_guard and helper.GateBegin(game.handle) != 0:
                    raise ValueError("raw native state guard failed to start")
                teacher_scores = score_policy(teacher, frame)
                student_scores = score_policy(student, frame)
                reload_scores = score_policy(reloaded, frame)
                # Destroy a caller-owned reconstruction; immutable frame bytes must survive it.
                for value in frame._payload().values():
                    if torch.is_tensor(value):
                        value.zero_()
                repeated = score_policy(student, frame)
                if (not torch.equal(student_scores.legal_mask, teacher_scores.legal_mask)
                        or not torch.equal(student_scores.logits, reload_scores.logits)
                        or not torch.equal(student_scores.logits, repeated.logits)):
                    raise ValueError("teacher/student mask, checkpoint or frame ownership mismatch")
                if raw_guard and helper.GateRestore() != 0:
                    raise ValueError("tensor scoring changed full native State/Game/RNG/belief bytes")
                if game.snapshot(0) != snapshot or game.stats() != stats:
                    raise ValueError("tensor scoring consumed or changed the native prompt")
                cases.append({"probe": name, "prefix_len": prefix_len,
                              "frame_sha256": frame.digest, "eligible_columns": list(record.eligible_columns),
                              "released_batch_scoring": True, "teacher_student_legal_columns_equal": True,
                              "checkpoint_logits_exact": True, "frame_copy_isolated": True,
                              "snapshot_and_stats_unchanged": True,
                              "full_native_bytes_including_rng_unchanged": raw_guard,
                              "raw_guard_scope": "prefix zero only; all scoring paths receive tensor bytes and no game handle"})
                if prefix_len == 0:
                    game.step([found["indices"][0]])
    if state_digest(teacher) != teacher_before or state_digest(student) != student_before:
        raise ValueError("qualification changed model parameters")
    receipt = {"schema": "native-teacher-frame-qualification/1", "status": "pass",
               "protocol_sha256": sha(args.protocol), "bindings_sha256": sha(args.bindings),
               "artifact_manifest_sha256": artifacts_digest, "cases": cases,
               "teacher": model_identity(teacher), "student": model_identity(student),
               "teacher_unchanged": True, "student_unchanged": True, "optimizer_updates": 0,
               "checkpoint_sha256": checkpoint_sha, "wall_seconds": time.monotonic() - started,
               "scope": "Four recorded reachable order-sensitive prefix frames; no claim of full-format privacy, rule coverage or playing strength."}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "pass", "cases": len(cases), "wall_seconds": receipt["wall_seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
