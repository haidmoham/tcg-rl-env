"""Teacher-free, frozen-checkpoint evaluation of the staged native pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tcg.native_distillation import (
    Budget, append_json, canonical, configure_runtime, evaluation_plan,
    file_hash, run_episode, validate_inputs, wall_guard,
)


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def evaluate(args):
    started = time.monotonic()
    protocol, bindings, decks = validate_inputs(args.protocol, args.bindings, "evaluate")
    if str(Path(__file__).resolve()) not in {str(Path(path).resolve()) for path in bindings["artifacts"]}:
        raise ValueError("executing evaluation source is not bound")
    prior = read(args.stage1_run / "training-result.json")
    stage1_manifest = read(args.stage1_run / "manifest.json")
    current_hash = hashlib.sha256(canonical(protocol)).hexdigest()
    if stage1_manifest["protocol_sha256"] != current_hash:
        raise ValueError("evaluation protocol differs from stage-1 frozen protocol")
    final = read(args.actorcritic_run / "actorcritic-result.json")
    if final["stage1_weights_sha256"] != prior["student_weights_sha256"]:
        raise ValueError("actor/critic lineage does not start at stage-1 endpoint")
    if final["stage1_protocol_sha256"] != current_hash or final["teacher_queries_added"] != 0:
        raise ValueError("actor/critic protocol lineage or teacher-free contract differs")
    if not final["stage3_actor_freeze_verified"]:
        raise ValueError("pre-RL actor freeze has not been established")
    endpoints = {arm: read(args.actorcritic_run / f"{name}.json")
                 for arm, name in (("pre_rl", "stage3"), ("post_rl", "stage4"))}
    torch, native = configure_runtime(bindings)
    initial_manifest = read(args.stage1_run / "initial.json")
    models = {"initial": native.load_student(args.stage1_run / "initial.pt",
                                              expected_sha256=initial_manifest["weights_sha256"])}
    for arm, name in (("pre_rl", "stage3"), ("post_rl", "stage4")):
        models[arm] = native.load_student(args.actorcritic_run / f"{name}.pt",
                                          expected_sha256=endpoints[arm]["weights_sha256"])
    identities = {arm: native.model_identity(model) for arm, model in models.items()}
    for arm in ("pre_rl", "post_rl"):
        if identities[arm]["state_digest"] != endpoints[arm]["identity"]["state_digest"]:
            raise ValueError("actual loaded checkpoint differs from its endpoint identity")
    if len({json.dumps(identity["config"], sort_keys=True) for identity in identities.values()}) != 1:
        raise ValueError("frozen players do not share the same student architecture")
    plan = evaluation_plan(protocol["seeds"]["evaluation"])
    if list(plan) != read(args.stage1_run / "evaluation-plan.json")["assignments"]:
        raise ValueError("evaluation assignment plan changed after training")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "manifest.json", {
        "schema": "native-pilot-evaluation/1", "protocol": protocol, "protocol_sha256": current_hash,
        "artifact_manifest_sha256": bindings["artifact_manifest_sha256"],
        "stage1_result_sha256": file_hash(args.stage1_run / "training-result.json"),
        "actorcritic_result_sha256": file_hash(args.actorcritic_run / "actorcritic-result.json"),
        "players": identities, "teacher_access": "forbidden",
        "contrasts": ["pre_rl_vs_initial", "post_rl_vs_pre_rl"],
        "pre_rl_actor_equals_distilled_actor": True,
    })
    write(args.output / "evaluation-plan.json", {"assignments": plan})
    budgets = {arm: Budget(protocol["limits"]) for arm in models}
    shared_startup = time.monotonic() - started
    elapsed = {"initial": shared_startup}
    for arm in ("pre_rl", "post_rl"):
        carried = endpoints[arm]["budget"]
        elapsed[arm] = carried["wall_seconds"] + shared_startup
        budgets[arm].admitted = carried["admitted_learner_microsteps"]
        budgets[arm].teacher_queries = carried.get("teacher_queries", 0)
        budgets[arm].teacher_seconds = carried.get("teacher_seconds", 0)
    eval_elapsed = dict.fromkeys(models, 0.0)
    eval_limit = protocol["pilot_sequence"]["stage_wall_allocations"]["evaluation"]
    phase_started = time.monotonic()
    for assignment in plan:
        for arm, model in models.items():
            budget = budgets[arm]
            budget.started = time.monotonic() - elapsed[arm]
            tick = time.monotonic()
            if budget.stop_reason() or eval_elapsed[arm] >= eval_limit:
                row = dict(assignment, status="not_started", reason=budget.stop_reason() or "evaluation_stage_cap",
                           score=None, winner=None)
            else:
                try:
                    with wall_guard(min(eval_limit - eval_elapsed[arm],
                                        budget.limits["wall_seconds"] - elapsed[arm])):
                        row, _, _ = run_episode(torch, native, model, decks, assignment, protocol, budget)
                except Exception as error:
                    row = dict(assignment, status="censored" if isinstance(error, TimeoutError) else "fault",
                               reason=f"{type(error).__name__}: {error}", score=None, winner=None)
            append_json(args.output / "evaluation-games.jsonl", dict(row, arm=arm))
            cost = time.monotonic() - tick
            elapsed[arm] += cost
            eval_elapsed[arm] += cost
            print(json.dumps({"index": assignment["index"], "arm": arm,
                              "status": row["status"], "score": row["score"]}), flush=True)
    if any(native.state_digest(model) != identities[arm]["state_digest"] for arm, model in models.items()):
        raise ValueError("evaluation changed a frozen player")
    final_budgets = {arm: dict(budget.receipt(), wall_seconds=elapsed[arm],
                                evaluation_wall_seconds=eval_elapsed[arm]) for arm, budget in budgets.items()}
    write(args.output / "evaluation-result.json", {
        "teacher_off": True, "teacher_model_loads": 0, "new_teacher_queries": 0,
        "frozen_players_unchanged": True, "arm_budgets": final_budgets,
        "evaluation_invocation_wall_seconds": time.monotonic() - started,
        "evaluation_game_phase_wall_seconds": time.monotonic() - phase_started,
        "cost_scope": "Shared host, CPU1 nice19. Arm budgets charge active evaluation time and shared startup; total invocation wall is also retained.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--stage1-run", type=Path, required=True)
    parser.add_argument("--actorcritic-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())
