"""Audit paired outcome receipts separately from pilot training diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from summarize_mew_student import read, sha


def outcome_summary(rows, plan):
    indexed = {(row["arm"], row["index"]): row for row in rows}
    if len(rows) != 216 or len(indexed) != 216:
        raise ValueError("all 216 unique evaluation receipts are required")
    if Counter((row["learner_deck"], row["opponent_deck"], row["learner_seat"]) for row in plan) != Counter(
            (left, right, seat) for left in range(6) for right in range(6) for seat in (0, 1)):
        raise ValueError("plan must cover 36 deck pairs with both seats")
    summaries, bounds = {}, {}
    for arm in ("initial", "pre_rl", "post_rl"):
        selected = [indexed[arm, item["index"]] for item in plan]
        for row, item in zip(selected, plan):
            if any(row[key] != value for key, value in item.items()):
                raise ValueError("actual evaluation assignment differs from frozen plan")
            if row["status"] == "terminal":
                if row["winner"] not in (0, 1, 2):
                    raise ValueError("unknown native terminal winner")
                expected = 0.5 if row["winner"] == 2 else float(row["winner"] == row["learner_seat"])
                if row["score"] != expected:
                    raise ValueError("terminal outcome/learner perspective mismatch")
            elif row["score"] is not None:
                raise ValueError("unresolved game was assigned a score")
        terminal = [row for row in selected if row["status"] == "terminal"]
        total = sum(row["score"] for row in terminal)
        bounds[arm] = [total / 72, (total + 72 - len(terminal)) / 72]
        costs = sorted(row["wall_seconds"] for row in selected if "wall_seconds" in row)
        def quantile(p):
            if not costs:
                return None
            index = (len(costs) - 1) * p
            lo = int(index)
            return costs[lo] + (costs[min(lo + 1, len(costs) - 1)] - costs[lo]) * (index - lo)
        summaries[arm] = {"planned": 72, "statuses": dict(Counter(row["status"] for row in selected)),
                          "wins": sum(row["score"] == 1 for row in terminal),
                          "draws": sum(row["score"] == 0.5 for row in terminal),
                          "losses": sum(row["score"] == 0 for row in terminal),
                          "planned_denominator_score_bounds": bounds[arm],
                          "game_wall_seconds": {"recorded": len(costs), "sum": sum(costs),
                                                "p50": quantile(0.5), "p95": quantile(0.95)},
                          "learner_genuine_choices": sum(row.get("learner_genuine_choices", 0) for row in selected),
                          "native_stats": dict(sum((Counter(row.get("native_stats", {})) for row in selected), Counter()))}
    return summaries, bounds


def summarize(args):
    stage1 = read(args.stage1 / "training-result.json")
    stage3 = read(args.actorcritic / "stage3-result.json")
    stage4 = read(args.actorcritic / "stage4-result.json")
    pipeline = read(args.actorcritic / "actorcritic-result.json")
    manifest = read(args.evaluation / "manifest.json")
    evaluation = read(args.evaluation / "evaluation-result.json")
    novelty = read(args.probe_audit)
    audits = {"stage1": read(args.stage1_audit), "actorcritic": read(args.actorcritic_audit),
              "evaluation": read(args.evaluation_audit)}
    if any(audit.get("status") not in ("pass", "passed") for audit in audits.values()):
        raise ValueError("independent actual-checkpoint audits must pass")
    if novelty["status"] != "passed":
        raise ValueError("independent probe novelty audit must pass")
    if (stage1["updates"] < 1 or stage4["updates"] < 1 or not stage1["teacher_unchanged"]
            or not stage1["critic_parameters_unchanged"] or not stage3["actor_unchanged"]
            or not stage3["exact_fixed_frame_logits_unchanged"] or pipeline["skipped_stage"] != 2
            or pipeline["teacher_queries_added"] != 0 or not evaluation["teacher_off"]
            or evaluation["new_teacher_queries"] != 0 or not evaluation["frozen_players_unchanged"]):
        raise ValueError("stage separation, real updates or frozen-player evidence failed")
    if (sha(args.stage1 / "training-result.json") != manifest["stage1_result_sha256"]
            or sha(args.actorcritic / "actorcritic-result.json") != manifest["actorcritic_result_sha256"]):
        raise ValueError("evaluation did not bind these training endpoints")
    for arm, budget in evaluation["arm_budgets"].items():
        if (budget["wall_seconds"] > 7200 or budget["admitted_learner_microsteps"] > 100000
                or budget["evaluation_wall_seconds"] > 1800):
            raise ValueError(f"{arm} exceeded its declared pilot budget")
    plan = read(args.evaluation / "evaluation-plan.json")["assignments"]
    rows = [json.loads(line) for line in (args.evaluation / "evaluation-games.jsonl").read_text().splitlines()]
    summaries, bounds = outcome_summary(rows, plan)
    alpha = manifest["protocol"]["evaluation"]["per_contrast_alpha"]
    radius = math.sqrt(2 * math.log(2 / alpha) / 36)
    contrasts = {}
    for name, candidate, control in (("policy_transfer", "pre_rl", "initial"), ("ppo_improvement", "post_rl", "pre_rl")):
        delta = [bounds[candidate][0] - bounds[control][1], bounds[candidate][1] - bounds[control][0]]
        interval = [max(-1, delta[0] - radius), min(1, delta[1] + radius)]
        verdict = "improved_against_uniform_anchor" if interval[0] > 0 else (
            "worse_against_uniform_anchor" if interval[1] < 0 else "inconclusive")
        contrasts[name] = {"candidate": candidate, "control": control,
                           "planned_denominator_difference_bounds": delta,
                           "family_adjusted_interval": interval, "verdict": verdict}
    before, after = [read(args.stage1 / f"probe-{phase}.json") for phase in ("before", "after")]
    updates = [json.loads(line) for line in (args.actorcritic / "stage4-updates.jsonl").read_text().splitlines()]
    return {"schema": "native-staged-pilot-readout/1", "status": "audited_receipt_summary",
            "roadmap": "https://github.com/haidmoham/86-tcg-agent/issues/3", "sequence": [1, 3, 4], "skipped_stage": 2,
            "identities": {"stage1_checkpoint_sha256": sha(args.stage1 / "student.pt"),
                           "pre_rl_checkpoint_sha256": sha(args.actorcritic / "stage3.pt"),
                           "post_rl_checkpoint_sha256": sha(args.actorcritic / "stage4.pt"),
                           "stage1_result_sha256": sha(args.stage1 / "training-result.json"),
                           "actorcritic_result_sha256": sha(args.actorcritic / "actorcritic-result.json"),
                           "evaluation_manifest_sha256": sha(args.evaluation / "manifest.json"),
                           "evaluation_games_sha256": sha(args.evaluation / "evaluation-games.jsonl"),
                           "evaluation_result_sha256": sha(args.evaluation / "evaluation-result.json"),
                           "stage1_source_artifact_manifest_sha256": read(args.stage1 / "manifest.json")["artifact_manifest_sha256"],
                           "continuation_source_artifact_manifest_sha256": pipeline["source_artifact_manifest_sha256"]},
            "independent_audits": {"stage1_sha256": sha(args.stage1_audit),
                                   "actorcritic_sha256": sha(args.actorcritic_audit),
                                   "evaluation_sha256": sha(args.evaluation_audit),
                                   "probe_novelty_sha256": sha(args.probe_audit)},
            "pipeline": {"policy_updates": stage1["updates"], "policy_training_games": stage1["episodes"],
                         "stage1_budget": stage1["budget"], "critic_fit": stage3["fit"],
                         "critic_heldout_games": stage3["natural_heldout_games"],
                         "critic_heldout_mse_before": stage3["heldout_before"]["mse"],
                         "critic_heldout_mse_after": stage3["heldout_after"]["mse"],
                         "critic_constant_baseline_mse": stage3["heldout_after"]["constant_mse"],
                         "critic_stage_actor_exactly_frozen": True,
                         "ppo_updates": stage4["updates"],
                         "ppo_epochs": [row["epochs"] for row in updates],
                         "all_evaluation_checkpoints_unchanged": True, "teacher_off_after_stage1": True},
            "fixed_independent_seed_policy_probe": {"frames": before["count"],
                         "cross_entropy_before": before["mean_cross_entropy"], "cross_entropy_after": after["mean_cross_entropy"],
                         "agreement_before": before["top1_agreement"], "agreement_after": after["top1_agreement"],
                         "training_overlap_frames": novelty["overlap_frames"],
                         "exact_digest_disjoint_frames": novelty["exact_digest_disjoint_frames"],
                         "novel_frames_metrics": {arm: {key: value for key, value in metrics.items() if key != "rows"}
                                                  for arm, metrics in novelty["metrics"].items()},
                         "scope": novelty["split"],
                         "limitation": "Independent seeds do not guarantee unseen observable states. Whole-probe metrics include exact optimizer overlap; identity-disjoint metrics are reported separately."},
            "evaluation": summaries, "contrasts": contrasts, "arm_budgets": evaluation["arm_budgets"],
            "evaluation_invocation_wall_seconds": evaluation["evaluation_invocation_wall_seconds"],
            "uncertainty": {"method": "Hoeffding over 36 independent deck-pair clusters, two seats per cluster; paired differences bounded in [-1,1]; unresolved outcomes take worst-case scores.",
                            "per_contrast_alpha": alpha, "family_alpha": 0.05,
                            "scope": "Two predeclared primary contrasts. The 72-game floor cannot reliably resolve small gains."},
            "decision": {"pipeline": "completed_bounded_historical_pilot", "playing_strength": contrasts["ppo_improvement"]["verdict"],
                         "league": "not implemented or authorized by this readout", "ranked": "full current-Standard gate remains held"},
            "limitations": ["One student seed and a weak uniform legal opponent. No general competitive strength claim.",
                            "Historical six-deck engine; no current-Standard legality, dense-vocabulary, reader or effect-coverage claim.",
                            "Calibration and policy probes contain correlated frames; value metrics do not establish game skill.",
                            "Costs are descriptive shared-host CPU1 nice19 observations, not isolated throughput or speedups.",
                            "Native forced shortcuts are separately reported; explicit learner-choice budgets differ from historical baseline admitted-step units."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--actorcritic", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--actorcritic-audit", type=Path, required=True)
    parser.add_argument("--probe-audit", type=Path, required=True)
    parser.add_argument("--evaluation-audit", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args)
    with args.receipt.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"pipeline": result["decision"]["pipeline"], "ppo": result["decision"]["playing_strength"]}))
