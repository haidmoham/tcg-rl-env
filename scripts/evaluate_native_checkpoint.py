"""Evaluate one frozen native student on the declared 576-game historical pool.

Reference checkpoints are identified without loading or reevaluating them.
Completed evaluations may supply paired reference outcomes. Evidence is private.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from tcg.native_distillation import Budget, canonical, configure_runtime, file_hash, run_episode, seed32

SCHEMA = "86-tcg-agent/native-checkpoint-evaluation/1"
ARMS = ("initial", "pre_rl")
STATUSES = {"terminal", "fault", "censored", "not_started"}


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    destination = Path(path)
    with destination.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def atomic_game(path, value):
    destination = Path(path)
    temporary = destination.with_suffix(".pending")
    write(temporary, value)
    if destination.exists():
        raise FileExistsError(destination)
    temporary.replace(destination)


def evaluation_plan(evaluation):
    if (evaluation["games_per_checkpoint"] != 576 or evaluation["pair_repetitions"] != 8
            or evaluation["seat_swap"] is not True):
        raise ValueError("evaluation requires 36 ordered pairs, eight seeds and both seats")
    seed, domain = evaluation["seed"], evaluation["seed_domain"]
    if type(seed) is not int or not 0 < seed < 2**32 or not isinstance(domain, str) or not domain:
        raise ValueError("evaluation requires an explicit nonzero seed and domain")
    assignments, all_seeds = [], set()
    for left in range(6):
        for right in range(6):
            for repetition in range(8):
                engine_seed = seed32(seed, domain, "engine", left, right, repetition)
                if engine_seed in all_seeds:
                    raise ValueError("evaluation seed collision")
                all_seeds.add(engine_seed)
                cluster_index = len(assignments) // 2
                for seat in (0, 1):
                    learner_seed = seed32(seed, domain, "learner", left, right, repetition, seat)
                    opponent_seed = seed32(seed, domain, "opponent", left, right, repetition, seat)
                    if learner_seed == opponent_seed or {learner_seed, opponent_seed} & all_seeds:
                        raise ValueError("evaluation action seed collision")
                    all_seeds.update((learner_seed, opponent_seed))
                    assignments.append({"index": len(assignments), "cluster_index": cluster_index,
                                        "repetition": repetition, "learner_deck": left,
                                        "opponent_deck": right, "learner_seat": seat,
                                        "engine_seed": engine_seed, "learner_seed": learner_seed,
                                        "opponent_seed": opponent_seed})
    return assignments


def checkpoint_descriptor(path):
    supplied = read(path)
    immutable_path = Path(supplied["receipt_path"]).resolve(strict=True)
    immutable = read(immutable_path)
    if Path(path).resolve() != immutable_path:
        expected = supplied.get("receipt_sha256")
        if expected != file_hash(immutable_path) or {key: value for key, value in supplied.items()
                                                    if key != "receipt_sha256"} != immutable:
            raise ValueError("latest descriptor differs from its immutable completed receipt")
    if immutable.get("schema") != "86-tcg-agent/native-sustained-checkpoint/1" or immutable.get("status") != "completed":
        raise ValueError("only completed sustained checkpoint receipts are usable")
    for key in ("weights", "state"):
        if not Path(immutable[key + "_path"]).is_absolute():
            raise ValueError("completed checkpoint artifact paths must be absolute")
        actual = Path(immutable[key + "_path"]).resolve(strict=True)
        if file_hash(actual) != immutable[key + "_sha256"]:
            raise ValueError(f"completed checkpoint {key} identity differs")
    return {"receipt_path": str(immutable_path), "receipt_sha256": file_hash(immutable_path),
            "descriptor": immutable}


def validate_sources(bindings):
    artifacts = bindings["artifacts"]
    if not artifacts or hashlib.sha256(canonical(artifacts)).hexdigest() != bindings["artifact_manifest_sha256"]:
        raise ValueError("source artifact manifest digest differs")
    resolved = set()
    for source, expected in artifacts.items():
        path = Path(source).resolve(strict=True)
        if file_hash(path) != expected:
            raise ValueError(f"bound source changed: {path.name}")
        resolved.add(path)
    required = [Path(__file__), REPOSITORY / "tcg/native_policy.py", REPOSITORY / "tcg/native_distillation.py",
                Path(bindings["engine_library"]), Path(bindings["deck_manifest"])]
    if any(path.resolve(strict=True) not in resolved for path in required):
        raise ValueError("evaluator, policy, collector, engine and decks must be bound")
    for key in ("engine", "deck_manifest"):
        path = bindings["engine_library" if key == "engine" else key]
        if file_hash(path) != bindings[key + "_sha256"]:
            raise ValueError(f"{key} differs from its explicit identity")


def validate_lineage(checkpoint, bindings, protocol_sha, protocol_file_sha):
    lineage = checkpoint["descriptor"]["lineage"]
    for key, expected in (("protocol_sha256", protocol_sha), ("engine_sha256", bindings["engine_sha256"]),
                          ("deck_manifest_sha256", bindings["deck_manifest_sha256"]),
                          ("protocol_file_sha256", protocol_file_sha),
                          ("source_artifact_manifest_sha256", bindings["artifact_manifest_sha256"])):
        if lineage[key] != expected:
            raise ValueError(f"checkpoint lineage mismatch: {key}")
    expected_schema = checkpoint["descriptor"]["model_identity"]["schema_digest"]
    if not isinstance(expected_schema, str) or len(expected_schema) != 64:
        raise ValueError("checkpoint actor schema identity is missing")


def normalized_limits(protocol):
    evaluation = protocol["evaluation"]
    wall = protocol["limits"]["evaluation_wall_seconds_per_checkpoint"]
    choices = protocol["limits"]["evaluation_learner_microsteps_per_checkpoint"]
    per_game = evaluation.get("per_game_wall_seconds", 120)
    microsteps = evaluation.get("max_game_microsteps", 4096)
    for value, maximum in ((wall, 1800), (choices, 100000), (per_game, 120), (microsteps, 4096)):
        if type(value) not in (int, float) or not 0 < value <= maximum:
            raise ValueError("evaluation exceeds a declared CPU pilot bound")
    if type(choices) is not int or type(microsteps) is not int:
        raise ValueError("evaluation choice and microstep ceilings must be integers")
    return {"wall_seconds": wall, "admitted_learner_microsteps": choices,
            "per_game_wall_seconds": per_game, "max_game_microsteps": microsteps,
            "teacher_queries": 1, "teacher_query_seconds": 1}


def unresolved(assignment, status, reason):
    return dict(assignment, status=status, reason=reason, winner=None, score=None)


def actor_digest(torch, model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("value_head."):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(canonical([name, str(value.dtype), list(value.shape)]))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def worker(output, deadline):
    started = time.monotonic()
    manifest = read(output / "manifest.json")
    bindings = read(manifest["bindings_path"])
    validate_sources(bindings)
    descriptor = manifest["checkpoint"]["descriptor"]
    if file_hash(descriptor["weights_path"]) != descriptor["weights_sha256"]:
        raise ValueError("checkpoint changed before worker load")
    torch, native = configure_runtime(bindings)
    model = native.load_student(descriptor["weights_path"], expected_sha256=descriptor["weights_sha256"])
    identity = native.model_identity(model)
    expected = descriptor["model_identity"]
    if identity["state_digest"] != expected["state_digest"] or identity["config"] != expected["config"]:
        raise ValueError("actual loaded model differs from completed checkpoint")
    if identity["schema_digest"] != expected["schema_digest"]:
        raise ValueError("loaded model actor schema differs")
    if actor_digest(torch, model) != descriptor["actor_sha256"]:
        raise ValueError("actual loaded actor differs from the checkpoint actor identity")
    write(output / "loaded-player.json", {"identity": identity, "teacher_model_loads": 0,
                                          "model_load_and_startup_seconds": time.monotonic() - started})
    plan = read(output / "evaluation-plan.json")["assignments"]
    decks = tuple(tuple(row["card_ids"]) for row in read(bindings["deck_manifest"])["decks"])
    limits = manifest["limits"]
    budget = Budget(limits)
    budget.started = manifest["invocation_started_monotonic"]
    budget.work_deadline = deadline
    episode_protocol = {"limits": limits}
    for assignment in plan:
        if budget.stop_reason():
            row = unresolved(assignment, "not_started", budget.stop_reason())
        else:
            write(output / f"{assignment['index']:04d}.started.json", assignment)
            row, _, _ = run_episode(torch, native, model, decks, assignment, episode_protocol, budget,
                                     external_wall_remaining=deadline - time.monotonic())
        atomic_game(output / f"{assignment['index']:04d}.json", row)
    unchanged = native.state_digest(model) == identity["state_digest"]
    write(output / "worker-result.json", {"frozen_player_unchanged": unchanged, "budget": budget.receipt(),
                                          "teacher_model_loads": 0, "new_teacher_queries": 0})
    if not unchanged:
        raise ValueError("evaluation mutated the frozen player")


def summarize(rows):
    complete = [row for row in rows if row["status"] == "terminal"]
    score = sum(row["score"] for row in complete)
    stats = Counter()
    genuine_known = []
    choices_known = []
    wall_known = []
    unknown = Counter()
    for row in rows:
        not_started = row["status"] == "not_started"
        native_stats = row.get("native_stats")
        if native_stats is not None:
            stats.update(native_stats)
        elif not not_started:
            unknown["native_counters"] += 1
        genuine = row.get("learner_genuine_choices")
        if genuine is not None:
            genuine_known.append(genuine)
        elif not not_started:
            unknown["genuine_choices"] += 1
        choices = row.get("choices")
        if choices is not None:
            choices_known.append(sum(choice["learner"] for choice in choices))
        elif not not_started:
            unknown["learner_choices"] += 1
        wall = row.get("wall_seconds")
        if wall is not None:
            wall_known.append(wall)
        elif not not_started:
            unknown["game_wall_seconds"] += 1
    return {"planned_games": 576, "statuses": dict(Counter(row["status"] for row in rows)),
            "wins": sum(row["score"] == 1 for row in complete),
            "draws": sum(row["score"] == 0.5 for row in complete),
            "losses": sum(row["score"] == 0 for row in complete),
            "planned_denominator_score_bounds": [score / 576, (score + 576 - len(complete)) / 576],
            "recorded_learner_choices": sum(choices_known),
            "genuine_learner_choices": sum(genuine_known) if not unknown["genuine_choices"] else None,
            "recorded_genuine_learner_choices": sum(genuine_known),
            "games_with_unknown_accounting": dict(unknown),
            "native_counters_complete": not bool(unknown["native_counters"]),
            "native_counters": dict(stats), "games_with_recorded_wall_seconds": len(wall_known),
            "games_with_native_counters": sum(bool(row.get("native_stats")) for row in rows),
            "summed_game_wall_seconds": sum(wall_known)}


def reference_evaluation(path, checkpoint, manifest, plan):
    result = read(path)
    reference_manifest_path = Path(path).parent / "manifest.json"
    reference_manifest = read(reference_manifest_path)
    original = reference_manifest["checkpoint"]["descriptor"]
    requested = checkpoint["descriptor"]
    identical_actor = (original["actor_sha256"] == requested["actor_sha256"]
                       and original["model_identity"]["config"] == requested["model_identity"]["config"]
                       and original["model_identity"]["schema_digest"] == requested["model_identity"]["schema_digest"])
    if (result["schema"] != SCHEMA or reference_manifest["protocol_sha256"] != manifest["protocol_sha256"]
            or file_hash(reference_manifest_path) != result["manifest_sha256"]
            or reference_manifest["artifact_manifest_sha256"] != manifest["artifact_manifest_sha256"]
            or reference_manifest["assignment_plan_sha256"] != manifest["assignment_plan_sha256"]
            or reference_manifest["engine_sha256"] != manifest["engine_sha256"]
            or reference_manifest["deck_manifest_sha256"] != manifest["deck_manifest_sha256"]
            or not identical_actor or not result["checkpoint_files_unchanged"] or result["frozen_player_unchanged"] is not True):
        raise ValueError("cached reference does not match this checkpoint, plan and environment")
    games = Path(path).parent / "evaluation-games.jsonl"
    if file_hash(games) != result["evaluation_games_sha256"]:
        raise ValueError("cached reference game receipts changed")
    rows = [json.loads(line) for line in games.read_bytes().splitlines()]
    validate_rows(rows, plan)
    return rows, {"receipt_path": str(Path(path).resolve()), "receipt_sha256": file_hash(path),
                  "source_checkpoint_weights_sha256": result["checkpoint_weights_sha256"],
                  "requested_checkpoint_weights_sha256": requested["weights_sha256"],
                  "actor_sha256": requested["actor_sha256"], "identical_actor_reuse": True,
                  "new_reference_games_executed": 0}


def validate_rows(rows, plan):
    if len(rows) != len(plan) or len({row["index"] for row in rows}) != len(plan):
        raise ValueError("all 576 unique assignment receipts are required")
    for row, assignment in zip(rows, plan):
        if any(row[key] != value for key, value in assignment.items()) or row["status"] not in STATUSES:
            raise ValueError("actual game receipt differs from frozen assignment")
        if row["status"] == "terminal":
            if row["winner"] not in (0, 1, 2):
                raise ValueError("unknown terminal winner")
            expected = 0.5 if row["winner"] == 2 else float(row["winner"] == row["learner_seat"])
            if row["score"] != expected:
                raise ValueError("terminal outcome perspective differs")
        elif row["score"] is not None:
            raise ValueError("unresolved game cannot receive a score")


def paired_contrast(candidate, reference, alpha):
    lows, highs = [], []
    for start in range(0, 576, 2):
        low = high = 0.0
        for left, right in zip(candidate[start:start + 2], reference[start:start + 2]):
            a = [left["score"]] * 2 if left["status"] == "terminal" else [0, 1]
            b = [right["score"]] * 2 if right["status"] == "terminal" else [0, 1]
            low += (a[0] - b[1]) / 2
            high += (a[1] - b[0]) / 2
        lows.append(low)
        highs.append(high)
    difference = [sum(lows) / 288, sum(highs) / 288]
    radius = math.sqrt(2 * math.log(2 / alpha) / 288)
    interval = [max(-1, difference[0] - radius), min(1, difference[1] + radius)]
    return {"clusters": 288, "difference_bounds": difference, "interval": interval, "radius": radius,
            "per_comparison_alpha": alpha, "verdict": "improved_against_uniform_anchor" if interval[0] > 0 else
            ("worse_against_uniform_anchor" if interval[1] < 0 else "inconclusive")}


def evaluate(args):
    started = time.monotonic()
    campaign_deadline = args.campaign_evaluation_deadline_monotonic
    if campaign_deadline is not None and (not math.isfinite(campaign_deadline) or campaign_deadline <= started):
        raise ValueError("campaign evaluation deadline must be finite and in the future")
    protocol_bytes = Path(args.protocol).read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol.get("schema") not in ("86-tcg-agent/native-sustained/1", "native-sustained/1"):
        raise ValueError("unsupported sustained protocol")
    if protocol.get("opponent") != "fixed_uniform_native_legal":
        raise ValueError("the evaluation opponent must remain uniform over native legal choices")
    plan = evaluation_plan(protocol["evaluation"])
    if args.plan_only:
        print(json.dumps({"games": len(plan), "clusters": len(plan) // 2,
                          "plan_content_sha256": hashlib.sha256(canonical({"assignments": plan})).hexdigest()}))
        return
    bindings = read(args.bindings)
    if protocol.get("execution_enabled") is not True:
        raise ValueError("the frozen sustained protocol must enable execution")
    validate_sources(bindings)
    protocol_file_sha = hashlib.sha256(protocol_bytes).hexdigest()
    protocol_sha = hashlib.sha256(canonical(protocol)).hexdigest()
    checkpoint = checkpoint_descriptor(args.checkpoint_receipt)
    validate_lineage(checkpoint, bindings, protocol_sha, protocol_file_sha)
    references = {}
    for arm in ARMS:
        receipt = getattr(args, arm + "_checkpoint_receipt")
        evaluation = getattr(args, arm + "_evaluation_receipt")
        if receipt is not None:
            references[arm] = checkpoint_descriptor(receipt)
            validate_lineage(references[arm], bindings, protocol_sha, protocol_file_sha)
        elif evaluation is not None:
            raise ValueError("cached reference evaluation requires its checkpoint descriptor")
    limits = normalized_limits(protocol)
    parent_deadline = min(started + limits["wall_seconds"], campaign_deadline) if campaign_deadline is not None else started + limits["wall_seconds"]
    deadline = parent_deadline - 5
    max_checkpoints = protocol["evaluation"].get("max_evaluation_checkpoints", 7)
    if max_checkpoints != 7:
        raise ValueError("the declared family contains at most seven checkpoints and two comparisons each")
    alpha = 0.05 / (max_checkpoints * 2)
    if (protocol["evaluation"]["family_alpha"] != 0.05
            or protocol["evaluation"]["comparisons_per_checkpoint"] != 2
            or not math.isclose(protocol["evaluation"]["per_comparison_alpha"], alpha,
                                rel_tol=1e-12, abs_tol=1e-15)):
        raise ValueError("evaluation family or multiplicity allocation differs")
    decks = read(bindings["deck_manifest"])["decks"]
    if len(decks) != 6 or any(len(row["card_ids"]) != 60 or any(type(card) is not int or card <= 0
                                                             for card in row["card_ids"]) for row in decks):
        raise ValueError("historical evaluation requires six exact 60-card lists")
    if args.output.resolve().is_relative_to(REPOSITORY):
        raise ValueError("raw evaluation evidence must remain outside the public checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "protocol.json").open("xb") as stream:
        stream.write(protocol_bytes)
    write(args.output / "evaluation-plan.json", {"assignments": plan})
    prior_costs = []
    for path in args.prior_evaluation_receipt:
        previous = read(path)
        prior_costs.append({"receipt_sha256": file_hash(path), "evaluation_wall_seconds": previous["evaluation_wall_seconds"],
                            "recorded_learner_choices": previous["summary"]["recorded_learner_choices"]})
    manifest = {"schema": SCHEMA, "protocol": protocol, "protocol_sha256": protocol_sha,
                "protocol_file_sha256": protocol_file_sha,
                "bindings_path": str(args.bindings.resolve()), "bindings_sha256": file_hash(args.bindings),
                "source_artifacts": bindings["artifacts"], "artifact_manifest_sha256": bindings["artifact_manifest_sha256"],
                "engine_sha256": bindings["engine_sha256"], "deck_manifest_sha256": bindings["deck_manifest_sha256"],
                "checkpoint": checkpoint, "references": references, "limits": limits,
                "assignment_plan_sha256": file_hash(args.output / "evaluation-plan.json"),
                "invocation_started_monotonic": started, "teacher_access": "forbidden",
                "evaluation_deadline_monotonic": parent_deadline, "worker_deadline_monotonic": deadline,
                "campaign_evaluation_deadline_monotonic": campaign_deadline,
                "opponent": "fixed_uniform_native_legal", "inference": "seeded_policy_sampling_temperature1",
                "statistics": {"clusters": 288, "family_alpha": 0.05, "max_checkpoints": 7,
                               "comparisons_per_checkpoint": 2, "per_comparison_alpha": alpha},
                "checkpoint_selection": "declared_time_boundary_only_no_outcome_selection",
                "prior_evaluation_costs": prior_costs, "prior_costs_reduce_current_cap": False}
    cached_references = {}
    reference_provenance = {}
    for arm, reference in references.items():
        cached = getattr(args, arm + "_evaluation_receipt")
        if cached is not None:
            cached_references[arm], reference_provenance[arm] = reference_evaluation(cached, reference, manifest, plan)
    manifest["reference_evaluations"] = reference_provenance
    write(args.output / "manifest.json", manifest)
    failure = None
    command = [sys.executable, str(Path(__file__).resolve()), "--_worker", str(args.output.resolve()),
               "--_deadline", str(deadline)]
    with (args.output / "worker.log").open("xb") as log:
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = "evaluation deadline exhausted during validation"
            else:
                with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as child:
                    try:
                        code = child.wait(timeout=remaining)
                        if code:
                            failure = f"worker exited with status {code}"
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                        failure = "hard evaluation watchdog expired"
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
    rows = []
    for assignment in plan:
        index = assignment["index"]
        path = args.output / f"{index:04d}.json"
        if path.is_file():
            row = read(path)
        else:
            begun = (args.output / f"{index:04d}.started.json").exists()
            row = unresolved(assignment, "censored" if begun else "not_started", failure or "worker did not produce a game receipt")
        rows.append(row)
    validate_rows(rows, plan)
    with (args.output / "evaluation-games.jsonl").open("xb") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")
    descriptor = checkpoint["descriptor"]
    files_unchanged = (file_hash(descriptor["weights_path"]) == descriptor["weights_sha256"]
                       and file_hash(descriptor["state_path"]) == descriptor["state_sha256"])
    validate_sources(bindings)
    if file_hash(args.protocol) != protocol_file_sha:
        raise ValueError("frozen protocol changed during evaluation")
    worker_path = args.output / "worker-result.json"
    worker_result = read(worker_path) if worker_path.exists() else None
    contrasts = {}
    for arm, reference in references.items():
        contrasts[arm] = {"status": "reference_not_evaluated"} if arm not in cached_references else paired_contrast(
            rows, cached_references[arm], alpha)
    summary = summarize(rows)
    result = {"schema": SCHEMA, "status": "complete" if summary["statuses"] == {"terminal": 576} else "completed_with_unresolved",
              "checkpoint_weights_sha256": descriptor["weights_sha256"], "checkpoint_receipt_sha256": checkpoint["receipt_sha256"],
              "manifest_sha256": file_hash(args.output / "manifest.json"),
              "evaluation_games_sha256": file_hash(args.output / "evaluation-games.jsonl"), "summary": summary,
              "contrasts": contrasts, "evaluation_wall_seconds": time.monotonic() - started,
              "teacher_model_loads": 0, "new_teacher_queries": 0, "checkpoint_files_unchanged": files_unchanged,
              "frozen_player_unchanged": worker_result["frozen_player_unchanged"] if worker_result else None,
              "worker_failure": failure, "worker_budget": worker_result["budget"] if worker_result else None,
              "charged_evaluation_learner_microsteps": worker_result["budget"]["admitted_learner_microsteps"] if worker_result else None,
              "evaluation_learner_microstep_bounds": ([worker_result["budget"]["admitted_learner_microsteps"]] * 2
                                                       if worker_result else [summary["recorded_learner_choices"], limits["admitted_learner_microsteps"]]),
              "cost_scope": "One checkpoint CPU1 nice19, shared host. Prior evaluations are reported separately. Training campaign wall belongs to the report manager.",
              "limitations": ["Killed native calls may omit choice counters and native statistics for the censored game; these are unknown rather than zero.",
                              "Model immutability is checked in the worker. It is unproven if the worker terminates before its final digest.",
                              "Historical six-deck uniform-opponent results do not establish competitive or current Standard strength."]}
    write(args.output / "evaluation-result.json", result)
    if not files_unchanged or (worker_result is not None and not worker_result["frozen_player_unchanged"]):
        raise ValueError("frozen checkpoint changed during evaluation")
    print(json.dumps({"status": result["status"], "summary": summary,
                      "result_sha256": file_hash(args.output / "evaluation-result.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--bindings", type=Path)
    parser.add_argument("--checkpoint-receipt", type=Path)
    parser.add_argument("--initial-checkpoint-receipt", type=Path)
    parser.add_argument("--pre-rl-checkpoint-receipt", type=Path)
    parser.add_argument("--initial-evaluation-receipt", type=Path)
    parser.add_argument("--pre-rl-evaluation-receipt", type=Path)
    parser.add_argument("--prior-evaluation-receipt", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--campaign-evaluation-deadline-monotonic", type=float)
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker is not None:
        worker(args._worker, args._deadline)
        return
    if args.protocol is None or (not args.plan_only and any(value is None for value in (args.bindings, args.checkpoint_receipt, args.output))):
        parser.error("protocol is required; execution also requires bindings, checkpoint receipt and output")
    try:
        evaluate(args)
    except Exception as error:
        if args.output is not None and args.output.is_dir() and not (args.output / "failure.json").exists():
            write(args.output / "failure.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
