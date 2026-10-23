"""Measure one real retained minibatch after two independent checkpoint restores."""

import argparse
from dataclasses import asdict
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_rollout(path, identity, weights, version):
    from profile_cabt_loss_inputs import decode, keys

    checksum = hashlib.sha256()
    rows, footer = [], None
    with gzip.open(path, "rb") as stream:
        line = stream.readline()
        header = decode(line)
        keys(header, ("schema", "identity", "assignments", "counts", "censored_episode_ids"))
        if header["schema"] != "86-tcg-agent/private-loss-inputs/1":
            raise ValueError("unsupported retained rollout schema")
        expected = {"learner_weights_sha256": weights, "learner_policy_version": version,
                    "engine_sha256": identity.engine_sha256, "projection_sha256": identity.projection_sha256,
                    "protocol_sha256": identity.protocol_sha256}
        if header["identity"] != expected:
            raise ValueError("retained rollout does not belong to this trained boundary")
        checksum.update(line.replace(b"\r\n", b"\n"))
        assignments = {}
        for assignment in header["assignments"]:
            episode = assignment["episode_id"]
            if episode in assignments or assignment["identity"] != expected:
                raise ValueError("mixed or duplicate retained assignments")
            assignments[episode] = assignment
        sequences = {}
        for line in stream:
            value = decode(line)
            if footer is not None:
                raise ValueError("data after retained rollout footer")
            if type(value) is dict and value.get("footer") is True:
                footer = keys(value, ("footer", "rows", "sha256"))
                continue
            keys(value, ("step", "return", "advantage", "normalized_advantage"))
            provenance = value["step"]["provenance"]
            episode = provenance["episode_id"]
            assignment = assignments[episode]
            expected_provenance = {"episode_id": episode, "learner_seat": assignment["learner_seat"],
                                   "deck_hash": assignment["learner_deck_sha256"], "policy_version": version,
                                   "opponent_hash": assignment["opponent"]["weights_sha256"]}
            if (provenance != expected_provenance or episode in header["censored_episode_ids"]
                    or value["step"]["sequence"] != sequences.get(episode, 0)):
                raise ValueError("retained learner provenance or sequence mismatch")
            sequences[episode] = sequences.get(episode, 0) + 1
            checksum.update(line.replace(b"\r\n", b"\n"))
            rows.append(value)
            if len(rows) > 100000:
                raise ValueError("retained rollout exceeds experiment bound")
    if (footer is None or type(footer["rows"]) is not int or footer["rows"] != len(rows)
            or len(rows) < 256 or footer["sha256"] != checksum.hexdigest()):
        raise ValueError("retained rollout checksum or row count mismatch")
    return rows, footer


def execute(args):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(ROOT))
    import torch
    from profile_cabt_loss_inputs import decode, typed_targets
    from tcg.policy import PolicyConfig, Vocabulary
    from tcg.population import Snapshot
    from tcg.ppo_update import _minibatch_step, _validate_optimizer
    from tcg.training_checkpoint import TrainingIdentity, load_training_checkpoint
    from train_cabt_baseline import training_draw

    started = time.monotonic()
    deadline = started + 110
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.5)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.cuda.reset_peak_memory_stats()

    checkpoint = Path(args.checkpoint).resolve(strict=True)
    launch_path = Path(args.launch).resolve(strict=True)
    rollout = Path(args.loss_inputs).resolve(strict=True)
    launch = decode(launch_path.read_bytes())
    expected_sources = launch["source_sha256"]
    required_sources = {str(path.relative_to(ROOT)).replace("\\", "/") for path in (ROOT / "tcg").glob("*.py")}
    required_sources.add("scripts/train_cabt_baseline.py")
    if set(expected_sources) != required_sources:
        raise ValueError("launch source manifest is incomplete")
    source_paths = {name: ROOT / name for name in required_sources}
    source_paths["scripts/profile_cabt_loss_inputs.py"] = ROOT / "scripts/profile_cabt_loss_inputs.py"
    source_paths["scripts/qualify_cabt_training_resume.py"] = Path(__file__).resolve()
    source_before = {name: digest(path) for name, path in source_paths.items()}
    if any(source_before[name] != value for name, value in expected_sources.items()):
        raise ValueError("current training source differs from the baseline launch")
    inputs = {"checkpoint/" + path.name: path for path in checkpoint.iterdir()}
    inputs.update(launch=launch_path, loss_inputs=rollout)
    hashes_before = {name: digest(path) for name, path in inputs.items()}
    payload = decode((checkpoint / "manifest.json").read_bytes())["payload"]
    identity = TrainingIdentity(**payload["identity"])
    if launch["protocol_sha256"] != identity.protocol_sha256 or launch["device"] != args.device:
        raise ValueError("launch protocol or device differs from resumed experiment")
    vocabulary = Vocabulary(**{name: tuple(value) for name, value in payload["vocabulary"].items()})
    config = PolicyConfig(**payload["model_config"])
    rows, footer = read_rollout(rollout, identity, payload["weights_sha256"], payload["counters"]["policy_version"])
    progress_bytes = Path(args.progress).read_bytes()
    next_episode = f"train-{payload['counters']['games_started']:08d}"
    progress_lines = progress_bytes.splitlines(keepends=True)
    if progress_lines and not progress_lines[-1].endswith(b"\n"):
        progress_lines.pop()
    progress = [decode(line) for line in progress_lines if line.strip()]
    actual = [row for row in progress if row.get("event") == "game" and row.get("episode_id") == next_episode
              and row.get("phase") == "training"]
    expected_collection = {"learner_weights_sha256": payload["weights_sha256"],
                           "learner_policy_version": payload["counters"]["policy_version"],
                           "engine_sha256": identity.engine_sha256, "projection_sha256": identity.projection_sha256,
                           "protocol_sha256": identity.protocol_sha256}
    if len(actual) != 1 or actual[0]["identity"] != expected_collection:
        raise ValueError("actual next-game continuation evidence missing or ambiguous")
    actual_minibatch = None
    fitting_boundary = None
    for event in progress:
        if event.get("event") == "fit_started":
            fitting_boundary = event["policy_version"]
        elif (event.get("event") == "fit_minibatch"
              and fitting_boundary == payload["counters"]["policy_version"]):
            actual_minibatch = event
            break
    if actual_minibatch is None or actual_minibatch["epoch"] != 1 or actual_minibatch["minibatch"] != 1:
        raise ValueError("actual first-minibatch continuation evidence missing")
    raw_optimizer = torch.load(BytesIO((checkpoint / "optimizer.pt").read_bytes()), map_location="cpu", weights_only=True)

    def verify_saved_optimizer(optimizer):
        state = optimizer.state_dict()
        if state["param_groups"] != raw_optimizer["param_groups"] or set(state["state"]) != set(raw_optimizer["state"]):
            raise ValueError("restored optimizer groups or parameter order differ from raw saved state")
        for index, entries in raw_optimizer["state"].items():
            if set(state["state"][index]) != set(entries) or any(
                not torch.equal(value, state["state"][index][name].detach().cpu()) for name, value in entries.items()
            ):
                raise ValueError("restored optimizer moments differ from raw saved state")

    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded resume measurement deadline")

    def trial():
        check_deadline()
        restored = load_training_checkpoint(checkpoint, expected_identity=identity, vocabulary=vocabulary, config=config)
        if set(restored.policy_rngs) != {"learner", "opponent", "shuffle"}:
            raise ValueError("expected the three baseline named sampling streams")
        verify_saved_optimizer(restored.optimizer)
        current = Snapshot(restored.counters.policy_version, restored.weights_sha256)
        next_draw = asdict(training_draw(restored.deck_rng))
        next_opponent = asdict(restored.population.sample_opponent(current=current))
        continuation_matches = next_draw == actual[0]["draw"] and next_opponent == actual[0]["opponent"]
        if not continuation_matches:
            raise ValueError("restored deck/population RNG differs from the actual uninterrupted next game")
        order = torch.randperm(len(rows), generator=restored.policy_rngs["shuffle"]).tolist()
        indices = tuple(order[:256])
        targets = typed_targets(tuple(rows[index] for index in indices))
        policy, optimizer = restored.policy, restored.optimizer
        before = {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}
        policy.to(device=args.device, dtype=torch.float32).requires_grad_(True).train()
        for state in optimizer.state.values():
            for name in ("exp_avg", "exp_avg_sq"):
                state[name] = state[name].to(args.device)
            state["step"] = state["step"].cpu()
        _validate_optimizer(policy, optimizer)
        verify_saved_optimizer(optimizer)
        metrics = _minibatch_step(policy, optimizer, targets, tuple(range(256)), 8, check_deadline)
        if args.device == "cuda":
            torch.cuda.synchronize()
        parameters = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
        optimizer_state = optimizer.state_dict()
        moments = {str(index) + "/" + name: value.detach().cpu().clone()
                   for index, state in optimizer_state["state"].items() for name, value in state.items()}
        draws = {"deck_uint64": [restored.deck_rng.getrandbits(64) for _ in range(32)],
                 "population": [asdict(restored.population.sample_opponent(current=current)) for _ in range(32)],
                 "policy": {name: torch.rand(32, generator=generator).tolist()
                            for name, generator in restored.policy_rngs.items()}}
        steps = [float(value) for name, value in moments.items() if name.endswith("/step")]
        if max(steps) != restored.counters.optimizer_steps + 1:
            raise ValueError("resumed optimizer did not advance the boundary by one minibatch")
        if not any(not torch.equal(before[name], value) for name, value in parameters.items()):
            raise ValueError("diagnostic optimizer step did not move any parameter")
        del restored, policy, optimizer, targets, optimizer_state
        return parameters, moments, metrics, indices, draws

    first = trial()
    second = trial()
    comparisons = {}
    for name, left, right in (("parameters", first[0], second[0]), ("optimizer_state", first[1], second[1])):
        if set(left) != set(right):
            raise ValueError("restored tensor state keys differ")
        comparisons[name] = {"tensors": len(left), "exact": all(torch.equal(left[key], right[key]) for key in left),
                             "allclose_rtol_1e-6_atol_1e-7": all(torch.allclose(left[key], right[key], rtol=1e-6, atol=1e-7) for key in left),
                             "max_absolute_difference": max(float((left[key] - right[key]).abs().max()) for key in left)}
    comparisons.update(metrics_exact=first[2] == second[2], permutation_exact=first[3] == second[3],
                       next_named_stream_draws_exact=first[4] == second[4])
    actual_metrics = {name: actual_minibatch[name] for name in first[2]}
    comparisons["actual_uninterrupted_first_minibatch_metrics_exact"] = first[2] == actual_metrics
    unchanged = (hashes_before == {name: digest(path) for name, path in inputs.items()}
                 and source_before == {name: digest(path) for name, path in source_paths.items()})
    exact = unchanged and all(comparisons[name]["exact"] for name in ("parameters", "optimizer_state")) and all(
        comparisons[name] for name in ("metrics_exact", "permutation_exact", "next_named_stream_draws_exact",
                                      "actual_uninterrupted_first_minibatch_metrics_exact"))
    return {"schema": "86-tcg-agent/trained-boundary-resume-measurement/1", "status": "exact" if exact else "mismatch",
            "identity": asdict(identity), "boundary_counters": payload["counters"], "input_sha256": hashes_before,
            "source_sha256": source_before, "inputs_and_sources_unchanged": unchanged,
            "retained_rollout_rows": len(rows), "retained_content_sha256": footer["sha256"],
            "continuation_progress_snapshot_sha256": hashlib.sha256(progress_bytes).hexdigest(),
            "actual_next_game": next_episode, "deck_and_population_continuation_match": True,
            "raw_optimizer_moments_groups_and_transport_exact": True,
            "selected_indices": first[3], "minibatch_rows": 256, "microbatch_rows": 8,
            "normalized_advantages": "stored full-rollout values, not renormalized",
            "optimizer_steps_executed": 2, "admitted_baseline_updates": 0,
            "metrics_first_restore": first[2], "metrics_second_restore": second[2], "comparisons": comparisons,
            "actual_uninterrupted_first_minibatch_metrics": actual_metrics,
            "device": args.device, "torch_version": str(torch.__version__), "python_version": sys.version,
            "float32": True, "amp": False, "deterministic_algorithms": True, "tf32": False,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
            "wall_seconds": time.monotonic() - started,
            "scope": "two-restoration one-minibatch repeatability, raw optimizer transport equivalence, actual first-minibatch observable metrics and next-game deck/population RNG continuation; no uninterrupted parameter-state, active-game native or full campaign resume"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--loss-inputs", required=True)
    parser.add_argument("--launch", required=True)
    parser.add_argument("--progress", required=True, help="Actual baseline progress JSONL; read once as a pinned snapshot")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", required=True, help="Exclusive fresh directory for immutable receipt and log")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required for the two diagnostic optimizer steps")
    directory = Path(args.output).resolve()
    if args.child:
        try:
            result = execute(args)
        except Exception as error:
            result = {"schema": "86-tcg-agent/trained-boundary-resume-measurement/1", "status": "failed",
                      "error_type": type(error).__name__, "error": str(error), "admitted_baseline_updates": 0}
        with (directory / "receipt.json").open("xb") as stream:
            stream.write(encode(result) + b"\n")
        return 0 if result["status"] == "exact" else 1
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", str(Path(args.checkpoint).resolve()),
               "--loss-inputs", str(Path(args.loss_inputs).resolve()), "--launch", str(Path(args.launch).resolve()),
               "--progress", str(Path(args.progress).resolve()), "--device", args.device,
               "--output", str(directory), "--execute", "--child"]
    with (directory / "child.log").open("xb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            code = process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
            receipt = directory / "receipt.json"
            if not receipt.exists():
                with receipt.open("xb") as stream:
                    stream.write(encode({"schema": "86-tcg-agent/trained-boundary-resume-measurement/1",
                                         "status": "watchdog_timeout", "watchdog_seconds": 120,
                                         "admitted_baseline_updates": 0}) + b"\n")
            return 1
    print(str(directory / "receipt.json"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
