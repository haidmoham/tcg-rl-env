"""Replay retained baseline loss rows for forward/backward diagnostic profiling."""

import argparse
import cProfile
from dataclasses import asdict, fields
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
INITIAL_SHA256 = "4b3f7240b29153c3f0670c56487f51b9445deb05f512892add981bafd9c8a7aa"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def decode(raw):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant {value}")
    return json.loads(raw, object_pairs_hook=json_object, parse_constant=reject)


def keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"invalid keys, expected {sorted(expected)}")
    return dict(value)


def retained_rows(path):
    digest = hashlib.sha256()
    rows = []
    footer = None
    with gzip.open(path, "rb") as stream:
        first = stream.readline()
        header = decode(first)
        keys(header, ("schema", "identity", "assignments", "counts", "censored_episode_ids"))
        if header["schema"] != "86-tcg-agent/private-loss-inputs/1":
            raise ValueError("unsupported loss input schema")
        digest.update(first.replace(b"\r\n", b"\n"))
        for line in stream:
            value = decode(line)
            if footer is not None:
                raise ValueError("data after footer")
            if type(value) is dict and value.get("footer") is True:
                footer = keys(value, ("footer", "rows", "sha256"))
            else:
                keys(value, ("step", "return", "advantage", "normalized_advantage"))
                digest.update(line.replace(b"\r\n", b"\n"))
                rows.append(value)
    if (footer is None or type(footer["rows"]) is not int or footer["rows"] != len(rows)
            or footer["sha256"] != digest.hexdigest() or len(rows) < 8):
        raise ValueError("loss input footer, row count, or digest mismatch")
    indices = tuple(index * (len(rows) - 1) // 7 for index in range(8))
    return header, tuple(rows[index] for index in indices), indices, footer


def typed_targets(rows):
    from tcg.decision import ActorDecision, LegalPrompt, Option, Token
    from tcg.rollout import EpisodeProvenance, LearnerStep, RolloutTargets

    def record(cls, value):
        return keys(value, (field.name for field in fields(cls)))

    def token(value):
        value = record(Token, value)
        if type(value["features"]) is not list:
            raise ValueError("token features must be an array")
        value["features"] = tuple(value["features"])
        return Token(**value)

    steps, returns, advantages, normalized = [], [], [], []
    for row in rows:
        value = record(LearnerStep, row["step"])
        value["provenance"] = EpisodeProvenance(**record(EpisodeProvenance, value["provenance"]))
        decision = record(ActorDecision, value["decision"])
        prompt = record(LegalPrompt, decision["prompt"])
        if any(type(array) is not list for array in (decision["tokens"], decision["selected"], prompt["options"])):
            raise ValueError("decision collections must be arrays")
        options = []
        for option in prompt["options"]:
            option = record(Option, option)
            option["token"] = token(option["token"])
            if type(option["state_indices"]) is not list:
                raise ValueError("state indices must be an array")
            option["state_indices"] = tuple(option["state_indices"])
            options.append(Option(**option))
        prompt["options"] = tuple(options)
        value["decision"] = ActorDecision(tuple(map(token, decision["tokens"])),
                                          LegalPrompt(**prompt), tuple(decision["selected"]))
        steps.append(LearnerStep(**value))
        for name, output in (("return", returns), ("advantage", advantages),
                             ("normalized_advantage", normalized)):
            target = row[name]
            if type(target) not in (int, float) or not math.isfinite(target):
                raise ValueError("nonfinite or invalid retained target")
            output.append(target)
    return RolloutTargets(tuple(steps), tuple(advantages), tuple(returns), tuple(normalized), ())


def execute(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(ROOT))
    import torch
    from tcg.checkpoint import load_checkpoint
    from tcg.collection import CollectionIdentity
    from tcg.objective import ppo_objective
    from tcg.policy import PolicyConfig, Vocabulary
    from tcg.ppo_update import action_statistics

    input_hash = sha256(args.rollout)
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in
                     (Path(__file__), *sorted((ROOT / "tcg").glob("*.py")))}
    header, rows, indices, footer = retained_rows(args.rollout)
    identity = CollectionIdentity(**header["identity"])
    if identity.learner_policy_version != 0 or identity.learner_weights_sha256 != INITIAL_SHA256:
        raise ValueError("profiler requires the pinned random initial learner")
    if sha256(args.checkpoint / "weights.pt") != INITIAL_SHA256:
        raise ValueError("initial checkpoint file SHA-256 mismatch")
    manifest = decode((args.checkpoint / "manifest.json").read_bytes())
    vocabulary = Vocabulary(**{key: tuple(value) for key, value in
                              keys(manifest["vocabulary"], ("card_ids", "zones", "roles")).items()})
    config = PolicyConfig(**manifest["config"])
    targets = typed_targets(rows)
    if any(step.provenance.policy_version != 0 for step in targets.steps):
        raise ValueError("retained steps mix policy versions")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    policy = load_checkpoint(args.checkpoint, vocabulary, config)
    initial = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable")
        torch.cuda.set_per_process_memory_fraction(0.5)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    args.output.mkdir(parents=True, exist_ok=False)
    policy.to(args.device).eval()
    policy.zero_grad(set_to_none=True)
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    python_profile = cProfile.Profile()
    started = time.perf_counter()
    with torch.profiler.profile(activities=activities, record_shapes=True, with_stack=True) as profile:
        python_profile.enable()
        with torch.profiler.record_function("retained_eight_rows_forward_backward"):
            logp, values, entropy = action_statistics(policy, targets, tuple(range(8)))
            objective = ppo_objective(logp, values, entropy,
                                     values.new_tensor([step.old_log_probability for step in targets.steps]),
                                     values.new_tensor(targets.returns),
                                     values.new_tensor(targets.normalized_advantages))
            objective.loss.backward()
        if args.device == "cuda":
            torch.cuda.synchronize()
        python_profile.disable()
    elapsed = time.perf_counter() - started
    finite_gradients = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                           for parameter in policy.parameters())
    gradient_parameters = sum(parameter.grad is not None for parameter in policy.parameters())
    unchanged = all(torch.equal(initial[name], value.detach().cpu()) for name, value in policy.state_dict().items())
    if not finite_gradients or not gradient_parameters or not unchanged:
        raise ValueError("gradient or unchanged-parameter diagnostic failed")
    if sha256(args.rollout) != input_hash or sha256(args.checkpoint / "weights.pt") != INITIAL_SHA256:
        raise ValueError("replay input changed during profiling")
    if any(sha256(ROOT / path) != digest for path, digest in source_hashes.items()):
        raise ValueError("profiled source changed during execution")
    profile.export_chrome_trace(str(args.output / "torch-trace.json"))
    python_profile.dump_stats(str(args.output / "python-profile.pstats"))
    summary = io.StringIO()
    pstats.Stats(python_profile, stream=summary).sort_stats("cumulative").print_stats(60)
    (args.output / "python-profile.txt").write_text(summary.getvalue(), encoding="utf-8")
    operators = [{"name": event.key, "count": event.count,
                  "self_cpu_time_us": event.self_cpu_time_total,
                  "cpu_time_us": event.cpu_time_total,
                  "device_time_us": getattr(event, "device_time_total", 0),
                  "stack": event.stack} for event in profile.key_averages(group_by_stack_n=8)]
    (args.output / "operators.json").write_text(json.dumps(operators, indent=2), encoding="utf-8")
    receipt = {
        "schema": "86-tcg-agent/loss-input-profile/1", "device": args.device,
        "measurement_scope": "one instrumented eight-row forward/backward diagnostic; no throughput comparison",
        "concurrent_work": "unknown; no claim of isolated hardware", "optimizer_updates": 0,
        "parameters_unchanged": unchanged, "finite_gradients": finite_gradients,
        "parameters_with_gradients": gradient_parameters, "wall_seconds_with_profiler": elapsed,
        "input_sha256": input_hash, "footer": footer,
        "footer_digest_encoding": "UTF-8 JSONL with LF; writer text mode emits CRLF on Windows",
        "initial_weights_sha256": INITIAL_SHA256, "checkpoint_manifest_sha256": sha256(args.checkpoint / "manifest.json"),
        "identity": asdict(identity), "config": asdict(config), "row_indices": indices,
        "rows": [{"episode_id": step.provenance.episode_id, "sequence": step.sequence,
                  "action_id": step.action_id, "visible_tokens": len(step.decision.tokens),
                  "options": len(step.decision.prompt.options),
                  "transformer_tokens": 2 + len(step.decision.tokens) + len(step.decision.prompt.options)
                  + (len(step.decision.selected) if step.decision.prompt.ordered else 0)} for step in targets.steps],
        "objective": {name: float(value.detach().cpu()) for name, value in vars(objective).items()},
        "replayed_log_probabilities": logp.detach().cpu().tolist(),
        "recorded_log_probabilities": [step.old_log_probability for step in targets.steps],
        "replayed_values": values.detach().cpu().tolist(),
        "recorded_values": [step.old_value for step in targets.steps],
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "source_sha256": source_hashes,
        "artifacts": {name: sha256(args.output / name) for name in
                      ("torch-trace.json", "operators.json", "python-profile.pstats", "python-profile.txt")},
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"receipt": str(args.output / "receipt.json"), "optimizer_updates": 0,
                      "parameters_unchanged": unchanged}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--execute", action="store_true", help="authorize one forward/backward diagnostic without optimizer")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required; this profiler performs backward but never creates an optimizer")
    execute(args)


if __name__ == "__main__":
    main()
