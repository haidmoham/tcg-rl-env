"""Run the frozen historical stage-3 and stage-4 CPU pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tcg import native_actorcritic as learning
from tcg.native_distillation import append_json, canonical, configure_runtime, file_hash, wall_guard


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def checkpoint(torch, native, model, optimizer, output, name, budget, lineage):
    weights = output / f"{name}.pt"
    sha = native.save_policy(model, weights)
    state = output / f"{name}.state.pt"
    torch.save({"optimizer": optimizer.state_dict() if optimizer is not None else None,
                "torch_rng": torch.get_rng_state(), "lineage": lineage, "budget": budget.receipt()}, state)
    receipt = {"weights_sha256": sha, "state_sha256": file_hash(state),
               "actor_sha256": learning.actor_digest(model), "identity": native.model_identity(model),
               "lineage": lineage, "budget": budget.receipt(), "resources": budget.receipt()}
    write_json(output / f"{name}.json", receipt)
    return receipt


def collect(torch, native, model, decks, protocol, budget, output, domain, count, *, retain_limit=None,
            per_game_limit=None):
    episodes, rows = [], []
    for index in range(count):
        budget.check()
        item = learning.assignment(protocol["seeds"][domain], domain, index)
        capacity = retain_limit if retain_limit is None else max(0, retain_limit - sum(len(ep.steps) for ep in episodes))
        if per_game_limit is not None:
            capacity = per_game_limit if capacity is None else min(capacity, per_game_limit)
        row, episode, rng = learning.collect_episode(torch, native, model, decks, item, protocol, budget,
                                                     retain_limit=capacity)
        append_json(output / f"{domain}-episodes.jsonl", row)
        torch.save(rng, output / f"{domain}-{index:04d}.rng.pt")
        rows.append(row)
        if row["status"] == "fault":
            raise RuntimeError(f"{domain} native game fault; inspect episode receipt")
        if episode is not None:
            episodes.append(episode)
    return episodes, rows


def calibrate(torch, native, model, decks, protocol, budget, output, lineage):
    actor_before = learning.actor_digest(model)
    parameters = learning.set_trainable(model, critic_only=True)
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable_names or any(not name.startswith("value_head.") for name in trainable_names):
        raise ValueError("calibration parameter allowlist includes an actor parameter")
    optimizer = torch.optim.AdamW(parameters, lr=protocol["calibration"]["lr"], weight_decay=0.0)
    calibration, train_rows = collect(torch, native, model, decks, protocol, budget, output,
                                       "calibration", protocol["calibration"]["training_games"],
                                       retain_limit=protocol["calibration"]["max_training_examples"])
    heldout, heldout_rows = collect(torch, native, model, decks, protocol, budget, output,
                                    "calibration_heldout", protocol["calibration"]["heldout_games"],
                                    per_game_limit=max(1, protocol["calibration"]["max_heldout_examples"]
                                                       // protocol["calibration"]["heldout_games"]))
    examples = [(step.frame, episode.outcome) for episode in calibration for step in episode.steps]
    probes = [(step.frame, episode.outcome) for episode in heldout for step in episode.steps]
    natural_heldout = sum(row["status"] == "terminal" for row in heldout_rows)
    if (not examples or natural_heldout < protocol["calibration"]["heldout_games"]
            or len(heldout) < natural_heldout or not probes):
        raise RuntimeError("insufficient natural-terminal calibration or independent held-out games")
    constant = sum(target for _, target in examples) / len(examples)
    fixed = tuple(frame for frame, _ in (examples + probes)[:8])
    before_logits = tuple(native.score_policy(model, frame).logits.clone() for frame in fixed)
    before = learning.calibration_metrics(torch, native, model, probes, constant)
    report = learning.fit_critic(torch, native, model, optimizer, examples, budget)
    after = learning.calibration_metrics(torch, native, model, probes, constant)
    if learning.actor_digest(model) != actor_before:
        raise ValueError("calibration changed actor or shared encoder parameters")
    if any(not torch.equal(old, native.score_policy(model, frame).logits) for old, frame in zip(before_logits, fixed)):
        raise ValueError("calibration changed exact actor logits on frozen frames")
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("nonfinite calibrated critic")
    torch.save({"training_examples": examples, "heldout_examples": probes}, output / "stage3-data.pt")
    data_sha = file_hash(output / "stage3-data.pt")
    current_lineage = dict(lineage, stage=3, data_sha256=data_sha, teacher_queries=0,
                           objective="terminal_MC_value_MSE_head_only")
    receipt = checkpoint(torch, native, model, optimizer, output, "stage3", budget, current_lineage)
    result = {"status": "completed_bounded_stage", "actor_unchanged": True,
              "exact_fixed_frame_logits_unchanged": True, "fixed_frame_count": len(fixed),
              "trainable_names": trainable_names, "critic_initialization": "stage1_original_random_head_stage2_skipped",
              "training_games": len(train_rows), "natural_training_games": sum(r["status"] == "terminal" for r in train_rows),
              "heldout_games": len(heldout_rows), "natural_heldout_games": natural_heldout,
              "fit": report, "heldout_before": before, "heldout_after": after,
              "data_sha256": data_sha, "checkpoint": receipt, "budget": budget.receipt(),
              "interpretation": "Outcome calibration diagnostics only. Neither lower MSE nor completed fitting establishes playing strength."}
    write_json(output / "stage3-result.json", result)
    return result


def improve(torch, native, model, decks, protocol, budget, output, lineage):
    optimizer = torch.optim.AdamW(learning.set_trainable(model, critic_only=False),
                                  lr=protocol["ppo"]["lr"], weight_decay=0.0)
    completed, reports = 0, []
    for update in range(protocol["ppo"]["max_updates"]):
        budget.check()
        episodes = []
        policy_digest = native.state_digest(model)
        for game in range(protocol["ppo"]["episodes_per_update"]):
            budget.check()
            index = update * protocol["ppo"]["episodes_per_update"] + game
            item = learning.assignment(protocol["seeds"]["ppo"], "ppo", index)
            row, episode, rng = learning.collect_episode(torch, native, model, decks, item, protocol, budget)
            row.update(update=update + 1)
            append_json(output / "stage4-episodes.jsonl", row)
            torch.save(rng, output / f"stage4-{index:04d}.rng.pt")
            if row["status"] == "fault":
                raise RuntimeError("stage 4 native game fault; frozen batch invalidated")
            if episode is None:
                write_json(output / f"stage4-update-{update + 1:04d}-censored.json",
                           {"status": "no_update", "reason": "complete two-episode batch unavailable",
                            "discarded_terminal_episodes": len(episodes), "budget": budget.receipt()})
                break
            episodes.append(episode)
        if len(episodes) != protocol["ppo"]["episodes_per_update"]:
            break
        if native.state_digest(model) != policy_digest:
            raise ValueError("collection mutated frozen learner weights")
        data_path = output / f"stage4-update-{update + 1:04d}-episodes.pt"
        torch.save(episodes, data_path)
        update_report = learning.ppo_update(torch, native, model, optimizer, episodes, protocol["ppo"], budget)
        completed += 1
        current = dict(lineage, stage=4, input_policy_digest=policy_digest, update=completed,
                       data_sha256=file_hash(data_path), teacher_queries=0,
                       objective="PPO_clipped_surrogate_plus_MC_value_MSE", opponent="fixed_uniform_native_legal")
        receipt = checkpoint(torch, native, model, optimizer, output, f"stage4-update-{completed:04d}", budget, current)
        report = {"update": completed, "epochs": update_report, "checkpoint": receipt, "budget": budget.receipt()}
        reports.append(report)
        append_json(output / "stage4-updates.jsonl", report)
    endpoint = checkpoint(torch, native, model, optimizer, output, "stage4", budget,
                          dict(lineage, stage=4, completed_updates=completed, teacher_queries=0))
    result = {"status": "completed_bounded_stage", "updates": completed, "checkpoint": endpoint,
              "budget": budget.receipt(), "evaluation_required": True, "teacher_queries": 0,
              "league": "not implemented", "playing_strength": "not established by this bounded training receipt"}
    write_json(output / "stage4-result.json", result)
    return result


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--stage1-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("validate", "train"), default="validate")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    budget = None
    stage = "validation"
    try:
        protocol, bindings, prior, decks = learning.validate_inputs(args.protocol, args.bindings, args.stage1_run)
        if args.mode == "train" and protocol.get("execution_enabled") is not True:
            raise ValueError("frozen actor/critic protocol must enable execution")
        protocol_sha = file_hash(args.protocol)
        bindings_sha = file_hash(args.bindings)
        lineage = {"protocol_sha256": protocol_sha, "bindings_sha256": bindings_sha,
                   "source_artifact_manifest_sha256": bindings["artifact_manifest_sha256"],
                   "stage1_protocol_sha256": json.loads((Path(args.stage1_run) / "student.json").read_bytes())["protocol_sha256"],
                   "stage1_result_sha256": file_hash(Path(args.stage1_run) / "training-result.json"),
                   "stage1_weights_sha256": prior["student_weights_sha256"], "stage2": "explicitly_skipped"}
        write_json(output / "inputs.json", {"protocol": protocol, "lineage": lineage,
                                           "prior_budget": prior["budget"]})
        if args.mode == "validate":
            write_json(output / "validation.json", {"passed": True, "training_started": False})
            return
        budget = learning.PilotBudget(protocol, prior["budget"], started=started)
        budget.stage(protocol["limits"]["stage3_wall_seconds"])
        budget.stage_started = started
        torch, native = configure_runtime(bindings)
        torch.manual_seed(protocol["seeds"]["optimizer"])
        model = native.load_student(Path(args.stage1_run) / "student.pt",
                                    expected_sha256=prior["student_weights_sha256"])
        original_identity = json.loads((Path(args.stage1_run) / "student.json").read_bytes())["model_identity"]
        current_identity = native.model_identity(model)
        if (current_identity["schema_digest"] != original_identity["schema_digest"]
                or current_identity["state_digest"] != original_identity["state_digest"]):
            raise ValueError("continuation model schema or state differs from the stage 1 checkpoint identity")
        checkpoint(torch, native, model, None, output, "stage1-actor", budget, lineage)
        stage = "stage3"
        with wall_guard(budget.remaining(reserve=False)):
            calibrated = calibrate(torch, native, model, decks, protocol, budget, output, lineage)
        stage = "stage4"
        budget.stage(protocol["limits"]["stage4_wall_seconds"])
        with wall_guard(budget.remaining(reserve=False)):
            final = improve(torch, native, model, decks, protocol, budget, output,
                            dict(lineage, stage3_weights_sha256=calibrated["checkpoint"]["weights_sha256"]))
        budget.check(reserve=False)
        write_json(output / "actorcritic-result.json", {"status": "completed_bounded_pipeline", "sequence": [1, 3, 4],
                   "skipped_stage": 2, "stage3": calibrated["checkpoint"]["weights_sha256"],
                   "stage4": final["checkpoint"]["weights_sha256"], "budget": budget.receipt(),
                   "stage1_protocol_sha256": lineage["stage1_protocol_sha256"],
                   "stage1_weights_sha256": lineage["stage1_weights_sha256"],
                   "stage3_sha256": calibrated["checkpoint"]["weights_sha256"],
                   "stage4_sha256": final["checkpoint"]["weights_sha256"],
                   "source_artifact_manifest": bindings["artifacts"],
                   "source_artifact_manifest_sha256": bindings["artifact_manifest_sha256"],
                   "resources": budget.receipt(), "stage3_actor_freeze_verified": calibrated["actor_unchanged"],
                   "budgets": {"stage1": prior["budget"], "stage3": calibrated["budget"], "stage4": final["budget"]},
                   "evaluation_required": True, "teacher_queries_added": 0})
    except Exception as error:
        write_json(output / "failure.json", {"status": "censored" if isinstance(error, TimeoutError) else "fault",
                   "stage": stage, "error": f"{type(error).__name__}: {error}",
                   "budget": budget.receipt() if budget is not None else None,
                   "checkpoint_policy": "Only prior checkpoints with completed JSON receipts are usable. Never resume a partially applied optimizer update."})
        raise


if __name__ == "__main__":
    main()
