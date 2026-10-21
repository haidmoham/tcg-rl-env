"""Run the separately gated historical Mew policy-distillation pilot on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tcg.native_distillation import (Budget, append_json, canonical, configure_runtime, evaluation_plan,
                                     file_hash, fit_labels, label_metrics, run_episode, seed32, validate_inputs)
from tcg.native_distillation import wall_guard


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def checkpoint(torch, native, student, optimizer, output, name, query_rng, episode_rng, progress, protocol_hash):
    weights = output / f"{name}.pt"
    digest = native.save_policy(student, weights)
    torch.save({"optimizer": optimizer.state_dict() if optimizer is not None else None,
                "torch_rng": torch.get_rng_state(), "query_rng": query_rng.getstate() if query_rng else None,
                "episode_rng": episode_rng.getstate() if episode_rng else None,
                "progress": progress, "protocol_sha256": protocol_hash}, output / f"{name}.state.pt")
    write_json(output / f"{name}.json", {"weights_sha256": digest, "model_identity": native.model_identity(student),
                                       "progress": progress, "protocol_sha256": protocol_hash})
    return weights, digest


def qualification(torch, native, student, teacher, decks, protocol, budget, output):
    from fastenc.games import Games
    teacher_before = native.state_digest(teacher)
    critic_before = {name: value.detach().clone() for name, value in student.named_parameters() if "value_head" in name}
    seed = seed32(protocol["seeds"]["probe"], "stage1-qualify-engine-v1")
    with Games([[decks[0], decks[1]]], seed=seed, seeds=[seed], threads=1) as games:
        with games.encode() as encoded:
            frame = native.FrameCapture().capture(encoded)
        before_snapshot, before_stats = games.snapshot(0), games.stats()
        started = time.monotonic()
        teacher_scores = native.score_policy(teacher, frame)
        budget.teacher_queries += 1
        budget.teacher_seconds += time.monotonic() - started
        student_scores = native.score_policy(student, frame)
        if not torch.equal(teacher_scores.legal_mask, student_scores.legal_mask):
            raise ValueError("same-frame teacher and student legal masks differ")
        for scores in (teacher_scores, student_scores):
            if not torch.isfinite(scores.logits[scores.legal_mask]).all() or not torch.isneginf(scores.logits[~scores.legal_mask]).all():
                raise ValueError("same-frame policy logits violate finite/legal contract")
        if games.snapshot(0) != before_snapshot or games.stats() != before_stats:
            raise ValueError("tensor-only scoring changed native game state or statistics")
        if native.state_digest(teacher) != teacher_before:
            raise ValueError("teacher parameters changed")
        path = output / "qualification-student.pt"
        sha = native.save_policy(student, path)
        restored = native.load_student(path, expected_sha256=sha)
        if not torch.equal(native.score_policy(restored, frame).logits, student_scores.logits):
            raise ValueError("save/reload changed policy logits")
        if any(parameter.requires_grad for name, parameter in student.named_parameters() if "value_head" in name):
            raise ValueError("student critic parameters are trainable")
        if any(not torch.equal(value, dict(student.named_parameters())[name]) for name, value in critic_before.items()):
            raise ValueError("student critic parameters changed")
        torch.save({"frame": frame, "teacher_probabilities": torch.softmax(teacher_scores.logits[0], -1)}, output / "qualification-frame.pt")
    receipt = {"passed": True, "frame_digest": frame.digest, "teacher_digest": teacher_before,
               "finite_masks": True, "snapshot_and_stats_unchanged": True, "reload_logits_exact": True,
               "frame_survived_native_release": True, "critic_parameters_frozen": True,
               "ordered_prefix_coverage": "not established by single-frame qualification", "budget": budget.receipt()}
    write_json(output / "qualification.json", receipt)
    return receipt


def collect_probes(torch, native, student, teacher, decks, protocol, budget, output):
    labels = []
    seed = protocol["seeds"]["probe"]
    max_frames = protocol["probe"]["max_frames"]
    index = 0
    while len(labels) < max_frames and not budget.stop_reason() and budget.teacher_available():
        assignment = {"index": index, "learner_deck": index % 6, "opponent_deck": (index + 1) % 6,
                      "learner_seat": index % 2, "engine_seed": seed32(seed, "stage1-probe-engine-v1", index),
                      "learner_seed": seed32(seed, "stage1-probe-learner-v1", index),
                      "opponent_seed": seed32(seed, "stage1-probe-opponent-v1", index)}
        probe_protocol = dict(protocol, limits=dict(protocol["limits"], max_game_microsteps=min(
            protocol["probe"]["max_microsteps"], protocol["limits"]["max_game_microsteps"])))
        row, batch, _ = run_episode(torch, native, student, decks, assignment, probe_protocol, budget,
                                    teacher=teacher, label_capacity=max_frames - len(labels), probe=True)
        append_json(output / "probe-episodes.jsonl", row)
        labels.extend(batch)
        if row["status"] == "fault":
            raise RuntimeError("probe native game fault; inspect private episode receipt")
        index += 1
        if index >= max_frames * 4:
            break
    if not labels:
        raise RuntimeError("no genuine held-out probe choices were captured")
    torch.save(labels, output / "heldout-probe-frames.pt")
    write_json(output / "probe-before.json", label_metrics(torch, native, student, labels))
    return labels


def train(torch, native, student, teacher, decks, protocol, budget, output, protocol_hash):
    parameters = native.freeze_policy_only(student)
    optimizer = torch.optim.AdamW([parameter for _, parameter in parameters], lr=protocol["learning"]["lr"], weight_decay=0.0)
    query_rng = random.Random(seed32(protocol["seeds"]["training"], "stage1-training-queries-v1"))
    episode_rng = random.Random(seed32(protocol["seeds"]["training"], "stage1-training-decks-v1"))
    initial, initial_sha = checkpoint(torch, native, student, optimizer, output, "initial", query_rng, episode_rng, {}, protocol_hash)
    teacher_digest = native.state_digest(teacher)
    critic_digest = {name: value.detach().clone() for name, value in student.named_parameters() if "value_head" in name}
    probes = collect_probes(torch, native, student, teacher, decks, protocol, budget, output)
    updates, episode_index, pending, discarded_labels = 0, 0, [], 0
    stop_reason = None
    try:
        while updates < protocol["learning"]["max_updates"] and not budget.stop_reason():
            if not budget.teacher_available():
                stop_reason = "teacher_query_budget"
                break
            seed = protocol["seeds"]["training"]
            assignment = {"index": episode_index, "learner_deck": episode_rng.randrange(6),
                          "opponent_deck": episode_rng.randrange(6), "learner_seat": episode_index % 2,
                          "engine_seed": seed32(seed, "stage1-training-engine-v1", episode_index),
                          "learner_seed": seed32(seed, "stage1-training-learner-v1", episode_index),
                          "opponent_seed": seed32(seed, "stage1-training-opponent-v1", episode_index)}
            capacity = protocol["learning"]["update_every_queried_choices"] - len(pending)
            row, labels, action_rng = run_episode(torch, native, student, decks, assignment, protocol, budget,
                                                   teacher=teacher, query_rng=query_rng, label_capacity=capacity)
            row.update(student_digest=native.state_digest(student), update=updates)
            append_json(output / "training-episodes.jsonl", row)
            torch.save(action_rng, output / f"episode-{episode_index:06d}-rng.pt")
            pending.extend(labels)
            episode_index += 1
            if row["status"] == "fault":
                raise RuntimeError("training native game fault; inspect private episode receipt")
            if len(pending) >= protocol["learning"]["update_every_queried_choices"]:
                label_path = output / f"labels-{updates + 1:04d}.pt"
                torch.save(pending, label_path)
                report = fit_labels(torch, native, student, optimizer, pending, budget)
                if report is None:
                    stop_reason = budget.stop_reason()
                    break
                updates += 1
                append_json(output / "updates.jsonl", dict(report, update=updates, budget=budget.receipt(),
                                                           labels_sha256=file_hash(label_path)))
                pending.clear()
                checkpoint(torch, native, student, optimizer, output, f"update-{updates:04d}", query_rng,
                           episode_rng, {"updates": updates, "episodes": episode_index}, protocol_hash)
        if pending:
            discarded_labels = len(pending)
            torch.save(pending, output / "unoptimized-labels.pt")
            pending.clear()
        if native.state_digest(teacher) != teacher_digest:
            raise ValueError("frozen teacher changed during distillation")
        if any(not torch.equal(value, dict(student.named_parameters())[name]) for name, value in critic_digest.items()):
            raise ValueError("critic parameter changed during policy-only training")
        final, final_sha = checkpoint(torch, native, student, optimizer, output, "student", query_rng,
                                      episode_rng, {"updates": updates, "episodes": episode_index}, protocol_hash)
        write_json(output / "probe-after.json", label_metrics(torch, native, student, probes))
        if budget.stop_reason() == "arm_wall_cap":
            raise TimeoutError("arm wall cap reached during finalization; retain only prior verified checkpoints")
        receipt = {"status": "completed_bounded_stage", "updates": updates, "episodes": episode_index,
                   "stop_reason": stop_reason or budget.stop_reason() or "update_cap", "initial_weights_sha256": initial_sha,
                   "student_weights_sha256": final_sha, "unoptimized_labels": discarded_labels,
                   "teacher_unchanged": True, "critic_parameters_unchanged": True,
                   "critic_outputs_may_change_with_shared_encoder": True, "objective": "policy_cross_entropy_only",
                   "opponent": "immutable_uniform_over_native_legal_choices", "budget": budget.receipt(),
                   "evaluation_required": True}
        write_json(output / "training-result.json", receipt)
        return receipt
    except Exception:
        if pending:
            torch.save(pending, output / "failure-unoptimized-labels.pt")
        raise


def evaluate(torch, native, initial, trained, decks, protocol, budget, output, training_receipt):
    plan = evaluation_plan(protocol["seeds"]["evaluation"])
    write_json(output / "evaluation-plan.json", {"assignments": plan})
    summaries, results = {}, {"initial": [], "student": []}
    budgets = {arm: Budget(protocol["limits"]) for arm in results}
    elapsed = {"initial": 0.0, "student": training_receipt["budget"]["wall_seconds"] + budget.receipt()["wall_seconds"]}
    budgets["student"].admitted = training_receipt["budget"]["admitted_learner_microsteps"]
    budgets["student"].teacher_queries = training_receipt["budget"]["teacher_queries"]
    budgets["student"].teacher_seconds = training_receipt["budget"]["teacher_seconds"]
    invocation_deadline = budget.started + protocol["limits"]["wall_seconds"]
    for assignment in plan:
        for arm, model in (("initial", initial), ("student", trained)):
            arm_budget = budgets[arm]
            arm_budget.started = time.monotonic() - elapsed[arm]
            invocation_remaining = invocation_deadline - time.monotonic()
            if arm_budget.stop_reason() or invocation_remaining <= 0:
                row = dict(assignment, status="not_started", reason=arm_budget.stop_reason() or "evaluation_invocation_wall_cap", score=None, winner=None)
            else:
                row, _, _ = run_episode(torch, native, model, decks, assignment, protocol, arm_budget,
                                         external_wall_remaining=invocation_remaining)
            append_json(output / "evaluation-games.jsonl", dict(row, arm=arm))
            elapsed[arm] = arm_budget.receipt()["wall_seconds"]
            results[arm].append(row)
    for arm, rows in results.items():
        complete = [row for row in rows if row["status"] == "terminal"]
        summaries[arm] = {"scheduled": len(rows), "terminal": len(complete),
                          "faults": sum(row["status"] == "fault" for row in rows),
                          "censored_or_not_started": sum(row["status"] in ("censored", "not_started") for row in rows),
                          "terminal_only_mean_score": sum(row["score"] for row in complete) / len(complete) if complete else None}
    arm_receipts = {arm: dict(value.receipt(), wall_seconds=elapsed[arm]) for arm, value in budgets.items()}
    receipt = {"status": "completed_scheduled_receipts" if all(row["status"] == "terminal" for rows in results.values() for row in rows) else "incomplete_games_retained",
               "teacher_off": True, "summaries": summaries, "arm_budgets": arm_receipts,
               "evaluation_invocation_wall_seconds": budget.receipt()["wall_seconds"],
               "limitation": "uniform legal opponent is a weak fixed anchor; censored games are not draws"}
    write_json(output / "evaluation-result.json", receipt)
    return receipt


def main(argv=None):
    invocation_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("qualify", "train", "evaluate"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    protocol, bindings, decks = validate_inputs(args.protocol, args.bindings, args.mode)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol_hash = hashlib.sha256(canonical(protocol)).hexdigest()
    write_json(args.output / "manifest.json", {"protocol": protocol, "protocol_sha256": protocol_hash,
                                              "artifact_manifest_sha256": bindings["artifact_manifest_sha256"],
                                              "mode": args.mode, "runtime": "cpu_fp32_eager_threads1_nice19",
                                              "evidence_scope": "historical_policy_transfer_pilot"})
    budget = Budget(protocol["limits"])
    budget.started = invocation_started
    try:
        with wall_guard(protocol["limits"]["wall_seconds"] - budget.receipt()["wall_seconds"]):
            torch, native = configure_runtime(bindings)
            student = native.build_student(native.SmallNativeConfig(**protocol["student"]), seed=protocol["seeds"]["student_init"])
            native.freeze_policy_only(student)
        if args.mode == "evaluate":
            if args.checkpoint is None:
                raise ValueError("evaluation requires --checkpoint")
            training_receipt = json.loads((args.checkpoint.parent / "training-result.json").read_bytes())
            with wall_guard(protocol["limits"]["wall_seconds"] - budget.receipt()["wall_seconds"]):
                trained = native.load_student(args.checkpoint, expected_sha256=training_receipt["student_weights_sha256"])
                initial_manifest = json.loads((args.checkpoint.parent / "initial.json").read_bytes())
                if native.state_digest(student) != initial_manifest["model_identity"]["state_digest"]:
                    raise ValueError("evaluation initial student differs from the frozen training control")
                if native.model_identity(trained)["config"] != native.model_identity(student)["config"]:
                    raise ValueError("evaluation student architecture differs from the initial control")
                if initial_manifest["protocol_sha256"] != protocol_hash:
                    raise ValueError("evaluation protocol differs from training")
            receipt = evaluate(torch, native, student, trained, decks, protocol, budget, args.output, training_receipt)
        else:
            mode_wall = protocol["limits"]["stage_wall_seconds"] if args.mode == "train" else protocol["limits"]["wall_seconds"]
            if args.mode == "train":
                budget.work_deadline = budget.started + mode_wall - protocol["limits"]["finalization_reserve_seconds"]
            with wall_guard(mode_wall - budget.receipt()["wall_seconds"]):
                teacher = native.load_teacher(bindings["teacher_checkpoint"], expected_sha256=bindings["teacher_sha256"])
                if args.mode == "qualify":
                    receipt = qualification(torch, native, student, teacher, decks, protocol, budget, args.output)
                else:
                    write_json(args.output / "evaluation-plan.json", {"assignments": evaluation_plan(protocol["seeds"]["evaluation"])})
                    receipt = train(torch, native, student, teacher, decks, protocol, budget, args.output, protocol_hash)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    except Exception as error:
        write_json(args.output / "failure.json", {"type": type(error).__name__, "error": str(error), "budget": budget.receipt()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
