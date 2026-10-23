"""Replay completed sustained calibration games to audit charged decisions."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def live_campaign_processes(campaign, transition=None):
    targets = [str(campaign).encode()]
    if transition is not None:
        targets.append(str(transition).encode())
    names = (b"run_native_campaign.py", b"resume_native_campaign_288.py", b"train_native_sustained.py",
             b"evaluate_native_checkpoint.py", b"evaluate_native_checkpoint_288.py")
    active = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
            state = (entry / "stat").read_text().rsplit(")", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            continue
        if state != "Z" and any(target in command for target in targets) and any(name in command for name in names):
            active.append(int(entry.name))
    return active


def terminal_receipt(campaign, transition):
    parent_sha = sha(campaign / "campaign-manifest.json")
    if transition is not None:
        takeover = read(transition / "takeover-manifest.json")
        if Path(takeover["campaign"]).resolve() != campaign or takeover["parent_manifest_sha256"] != parent_sha:
            raise ValueError("takeover manifest does not identify the original campaign")
        path = transition / "campaign-result.json"
        result = read(path)
        if (result.get("schema") != "native-sustained-campaign288-result/1"
                or Path(result["campaign"]).resolve() != campaign
                or result["original_campaign_manifest_sha256"] != parent_sha
                or result["training_pid"] != takeover["trainer_identity"]["pid"]
                or result["amendment_sha256"] != takeover["amendment_sha256"]):
            raise ValueError("takeover terminal receipt lineage differs")
        if not takeover["source_artifacts"] or any(sha(source) != expected for source, expected in takeover["source_artifacts"].items()):
            raise ValueError("takeover source identities differ")
    else:
        path = campaign / "campaign-result.json"
        result = read(path)
    if result["status"] not in {"campaign_wall_cap", "training_fault", "bounded_campaign_finished", "final_evaluation_incomplete"}:
        raise ValueError("campaign receipt does not describe a terminal outcome")
    return path


def calibration_assignment(protocol, domain, index):
    seed = protocol["seeds"][domain]
    seed_domain = "sustained-" + domain

    def derive(kind):
        encoded = json.dumps([seed, seed_domain, kind, index], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "little") or 1

    deck_seed = derive("decks")
    deck_rng = random.Random(deck_seed)
    return {"index": index, "learner_deck": deck_rng.randrange(6), "opponent_deck": deck_rng.randrange(6),
            "learner_seat": index % 2, "engine_seed": derive("engine"), "learner_seed": derive("learner"),
            "opponent_seed": derive("opponent"), "deck_assignment_seed": deck_seed,
            "deck_assignment": "independent_uniform_six_deck_draws"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transition", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.campaign, args.output = args.campaign.resolve(), args.output.resolve()
    args.transition = args.transition.resolve() if args.transition is not None else None
    if (os.name != "posix" or args.output.is_relative_to(ROOT) or args.output.is_relative_to(args.campaign)
            or args.transition is not None and args.output.is_relative_to(args.transition)):
        raise ValueError("use Linux and a new private audit directory outside the campaign and checkout")
    training = args.campaign / "training"
    manifest, bindings = read(training / "manifest.json"), read(args.bindings)
    if sha(args.bindings) != manifest["lineage"]["bindings_sha256"] or bindings["artifacts"] != manifest["source_artifact_manifest"]:
        raise ValueError("private source bindings differ from training")
    for path, expected in bindings["artifacts"].items():
        if sha(path) != expected:
            raise ValueError("frozen source or input changed")
    protocol = manifest["protocol"]
    if protocol["schema"] != "86-tcg-agent/native-sustained/1" or protocol["sequence"] != [1, 3, 4]:
        raise ValueError("replay requires the sustained three-stage campaign")
    campaign_manifest = read(args.campaign / "campaign-manifest.json")
    protocol_bytes = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if (hashlib.sha256(protocol_bytes).hexdigest() != manifest["lineage"]["protocol_sha256"]
            or campaign_manifest["protocol"] != protocol
            or campaign_manifest["protocol_sha256"] != manifest["lineage"]["protocol_file_sha256"]):
        raise ValueError("frozen campaign protocol identities differ")
    latest = read(training / "latest-checkpoint.json")
    sequence = int(latest["checkpoint_id"].removeprefix("checkpoint-"))
    checkpoints = [read(training / f"checkpoint-{index:06d}.json") for index in range(sequence + 1)]
    for index, row in enumerate(checkpoints):
        if (row["status"] != "completed" or row["checkpoint_id"] != f"checkpoint-{index:06d}"
                or row["lineage"]["protocol_sha256"] != manifest["lineage"]["protocol_sha256"]
                or row["lineage"]["source_artifact_manifest_sha256"] != bindings["artifact_manifest_sha256"]):
            raise ValueError("completed checkpoint identity differs from the frozen campaign")
    transfer = max((row for row in checkpoints if row["stage"] == 1), key=lambda row: row["stage_updates"])
    if (transfer["checkpoint_id"] != "checkpoint-000003" or transfer["stage_updates"] != protocol["stage1"]["max_updates"]
            or sha(transfer["weights_path"]) != transfer["weights_sha256"]):
        raise ValueError("frozen transfer checkpoint changed")
    calibration = [row for row in checkpoints if row["stage"] == 3]
    examples_path = training / "stage3-terminal-examples.pt"
    examples_sha = sha(examples_path)
    if not calibration or any(row["lineage"]["data_sha256"] != examples_sha for row in calibration):
        raise ValueError("calibration example bytes differ from completed checkpoint lineage")
    inputs = [training / "manifest.json", args.campaign / "campaign-manifest.json", args.bindings,
              Path(transfer["weights_path"]), examples_path, training / "stage1-result.json", training / "stage3-result.json"]
    inputs.extend(training / f"{row['checkpoint_id']}.json" for row in checkpoints)
    originals_by_domain = {}
    episode_pins = {}
    for domain, count_key in (("calibration", "training_games"), ("calibration_heldout", "heldout_games")):
        count = protocol["calibration"][count_key]
        if count != 16:
            raise ValueError("calibration replay requires the declared 16 games per domain")
        episode_path = training / f"stage3-{domain}-episodes.jsonl"
        episode_bytes = episode_path.read_bytes()
        episode_pins[str(episode_path.resolve())] = hashlib.sha256(episode_bytes).hexdigest()
        originals = [json.loads(line) for line in episode_bytes.splitlines()]
        if len(originals) != count:
            raise ValueError("calibration replay requires every original assignment")
        for index, original in enumerate(originals):
            expected = calibration_assignment(protocol, domain, index)
            if (any(original.get(key) != value for key, value in expected.items())
                    or original["policy_digest"] != transfer["model_identity"]["state_digest"]
                    or original["status"] != "terminal"):
                raise ValueError("original calibration receipt differs from independently derived schedule or transfer policy")
            inputs.append(training / f"stage3-{domain}-{index:06d}.rng.pt")
        inputs.append(episode_path)
        originals_by_domain[domain] = originals
    if not args.validate_only:
        active = live_campaign_processes(args.campaign, args.transition)
        if active:
            raise RuntimeError(f"campaign runtime processes are still live: {active}")
        inputs.append(terminal_receipt(args.campaign, args.transition))
        if args.transition is not None:
            inputs.append(args.transition / "takeover-manifest.json")
    pins = {str(path.resolve()): sha(path) for path in inputs}
    if pins[str(examples_path.resolve())] != examples_sha or any(pins[path] != expected for path, expected in episode_pins.items()):
        raise ValueError("calibration inputs changed while the audit plan was prepared")

    def pinned_bytes(path):
        path = Path(path).resolve()
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != pins[str(path)]:
            raise ValueError(f"pinned audit input changed: {path.name}")
        return data

    args.output.mkdir(parents=True, exist_ok=False)
    plan = {"schema": "86-tcg-agent/calibration-replay/1", "audit_source_sha256": sha(__file__),
            "training_manifest_sha256": sha(training / "manifest.json"), "bindings_sha256": sha(args.bindings),
            "checkpoint_sha256": transfer["weights_sha256"], "wall_seconds": 600,
            "learner_choice_cap": 10000, "native_game_cap": 32, "teacher_queries": 0,
            "optimizer_steps": 0, "runtime": "CPU1 nice19; separate post-campaign audit costs",
            "input_artifacts": pins, "transition": str(args.transition) if args.transition is not None else None,
            "purpose": "Reconstruct all charged calibration learner choices with deterministic native replay. Compare every retained frame, final action RNG, native counter, outcome and explicit microstep count."}
    write(args.output / "plan.json", plan)
    if args.validate_only:
        print(json.dumps({"status": "validated_without_models_or_games"}))
        return
    active = live_campaign_processes(args.campaign, args.transition)
    if active:
        raise RuntimeError(f"campaign runtime processes became live before imports: {active}")
    started = time.monotonic()
    def deadline_expired(signum, frame):
        raise TimeoutError("independent calibration replay wall allocation exhausted")
    signal.signal(signal.SIGALRM, deadline_expired)
    signal.setitimer(signal.ITIMER_REAL, 600)
    os.nice(max(0, 19 - os.nice(0)))
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    sys.path.insert(0, str(ROOT))
    from tcg.native_distillation import configure_runtime, wall_guard
    torch, native = configure_runtime(bindings)
    from tcg.native_actorcritic import score_actorcritic
    from fastenc.games import Games
    model = native.load_student(transfer["weights_path"], expected_sha256=transfer["weights_sha256"])
    model_before = native.state_digest(model)
    if model_before != transfer["model_identity"]["state_digest"]:
        raise ValueError("loaded transfer policy differs from every original calibration policy digest")
    examples = torch.load(io.BytesIO(pinned_bytes(examples_path)), map_location="cpu", weights_only=False)
    decks = [row["card_ids"] for row in read(bindings["deck_manifest"])["decks"]]
    charged, audited, results = 0, 0, []
    try:
        with wall_guard(max(0.001, 600 - (time.monotonic() - started))):
            for domain, data_key, limit_key in (("calibration", "fit", "retained_training_per_game"),
                                                 ("calibration_heldout", "heldout", "retained_heldout_per_game")):
                originals = originals_by_domain[domain]
                if len(originals) != 16:
                    raise ValueError("calibration replay requires all 16 natural games per domain")
                expected_frames = examples[data_key]
                frame_offset = 0
                for original in originals:
                    if audited >= 32 or original["status"] != "terminal":
                        raise ValueError("original calibration game is not a declared natural terminal")
                    pair, seat = [None, None], original["learner_seat"]
                    pair[seat], pair[1 - seat] = decks[original["learner_deck"]], decks[original["opponent_deck"]]
                    learner_rng = torch.Generator(device="cpu").manual_seed(original["learner_seed"])
                    opponent_rng = torch.Generator(device="cpu").manual_seed(original["opponent_seed"])
                    local_charged, retained, explicit, winner = 0, [], 0, None
                    with wall_guard(min(120, max(0.001, 600 - (time.monotonic() - started)))), Games(
                            [pair], seed=original["engine_seed"], seeds=[original["engine_seed"]], threads=1) as games:
                        capture = native.FrameCapture()
                        for _ in range(protocol["limits"]["max_game_microsteps"]):
                            with games.encode() as encoded:
                                wire = encoded.records[0]
                                if int(wire["terminal"]) == 1:
                                    winner = int(wire["winner"])
                                    break
                                frame = capture.capture(encoded)
                            record = frame.records[0]
                            if record.select_player == seat:
                                charged, local_charged = charged + 1, local_charged + 1
                                if charged > 10000:
                                    raise RuntimeError("independent replay choice allocation exhausted")
                                logits, _, _ = score_actorcritic(torch, native, model, frame)
                                action = int(torch.multinomial(logits[0].softmax(-1), 1, generator=learner_rng))
                                if len(retained) < protocol["calibration"][limit_key]:
                                    retained.append(frame.digest)
                            else:
                                action = record.eligible_columns[int(torch.randint(len(record.eligible_columns), (1,), generator=opponent_rng))]
                            games.step([action])
                            explicit += 1
                        stats = games.stats()
                    expected_rng = torch.load(io.BytesIO(pinned_bytes(training / f"stage3-{domain}-{original['index']:06d}.rng.pt")),
                                              map_location="cpu", weights_only=False)
                    targets = expected_frames[frame_offset:frame_offset + len(retained)]
                    frame_offset += len(retained)
                    outcome = 0.0 if winner == 2 else 1.0 if winner == seat else -1.0
                    if (winner != original["winner"] or explicit != original["explicit_microsteps"]
                            or stats != original["native_stats"] or len(retained) != original["retained_learner_steps"]
                            or retained != [frame.digest for frame, target in targets]
                            or any(target != outcome for frame, target in targets)
                            or not torch.equal(expected_rng["learner"], learner_rng.get_state())
                            or not torch.equal(expected_rng["opponent"], opponent_rng.get_state())):
                        raise ValueError("independent native calibration replay differs from original evidence")
                    audited += 1
                    results.append({"domain": domain, "index": original["index"], "charged_learner_choices": local_charged,
                                    "explicit_both_seat_microsteps": explicit, "retained_frames_matched": len(retained),
                                    "outcome": outcome, "native_counters_and_final_action_rng_exact": True})
                if frame_offset != len(expected_frames):
                    raise ValueError("stored calibration examples were not fully matched")
            expected_charged = (read(training / "stage3-result.json")["resources"]["cumulative_training_learner_steps"]
                                - read(training / "stage1-result.json")["resources"]["cumulative_training_learner_steps"])
            if charged != expected_charged or native.state_digest(model) != model_before:
                raise ValueError("reconstructed stage-3 charge or frozen model differs")
            for path, expected in bindings["artifacts"].items():
                if sha(path) != expected:
                    raise ValueError("frozen source changed during replay")
            for path, expected in pins.items():
                if sha(path) != expected:
                    raise ValueError("pinned audit input changed during replay")
            write(args.output / "result.json", {"status": "passed", "games": audited, "charged_learner_choices": charged,
                "original_stage3_charge": expected_charged, "games_replayed": results,
                "wall_seconds": time.monotonic() - started, "teacher_queries": 0, "optimizer_steps": 0,
                "scope": "Deterministic replay through an independently counted loop and original qualified engine/scorer. Retained observable frames and final action RNG match. This is not a new rules oracle or native-privacy proof."})
    except Exception as error:
        write(args.output / "failure.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
              "games_replayed": audited, "charged_learner_choices": charged, "wall_seconds": time.monotonic() - started})
        raise
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({"status": "passed", "games_replayed": audited, "charged_learner_choices": charged}))


if __name__ == "__main__":
    main()
