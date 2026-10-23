"""Summarize a frozen policy-transfer run and all paired evaluation assignments."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def summarize(training, evaluation):
    train = read(training / "training-result.json")
    before, after = [read(training / f"probe-{phase}.json") for phase in ("before", "after")]
    manifest = read(training / "manifest.json")
    evaluation_manifest = read(evaluation / "manifest.json")
    if manifest["protocol_sha256"] != evaluation_manifest["protocol_sha256"]:
        raise ValueError("training/evaluation protocol mismatch")
    protocol = manifest["protocol"]
    if (train["objective"] != "policy_cross_entropy_only" or not train["teacher_unchanged"]
            or not train["critic_parameters_unchanged"] or train["updates"] < 1):
        raise ValueError("stage-1 objective, frozen weights or real update evidence failed")
    if (before["count"] == 0 or before["count"] != after["count"]
            or [row["frame_digest"] for row in before["rows"]]
            != [row["frame_digest"] for row in after["rows"]]):
        raise ValueError("before/after probes are not the same fixed frames")
    plan = read(evaluation / "evaluation-plan.json")["assignments"]
    if len(plan) != 72 or len({row["index"] for row in plan}) != 72:
        raise ValueError("evaluation must schedule exactly 72 distinct assignments per arm")
    if Counter((row["learner_deck"], row["opponent_deck"], row["learner_seat"]) for row in plan) != Counter(
            (left, right, seat) for left in range(6) for right in range(6) for seat in (0, 1)):
        raise ValueError("evaluation does not cover all six-deck pairs and both seats")
    rows = [json.loads(line) for line in (evaluation / "evaluation-games.jsonl").read_text().splitlines()]
    by_key = {(row["arm"], row["index"]): row for row in rows}
    if len(rows) != 144 or len(by_key) != 144:
        raise ValueError("evaluation receipts are missing or duplicated")
    summaries, bounds = {}, {}
    for arm in ("initial", "student"):
        selected = [by_key[arm, assignment["index"]] for assignment in plan]
        for assignment, row in zip(plan, selected):
            if any(row[key] != value for key, value in assignment.items()):
                raise ValueError("evaluation assignment differs from frozen plan")
            if row["status"] == "terminal":
                expected = 0.5 if row["winner"] == 2 else float(row["winner"] == row["learner_seat"])
                if row["winner"] not in (0, 1, 2) or row["score"] != expected:
                    raise ValueError("evaluation score differs from native terminal result")
            elif row["score"] is not None:
                raise ValueError("unresolved evaluation game has a fabricated score")
        terminal = [row for row in selected if row["status"] == "terminal"]
        total_score = sum(row["score"] for row in terminal)
        unresolved = 72 - len(terminal)
        bounds[arm] = [total_score / 72, (total_score + unresolved) / 72]
        summaries[arm] = {"scheduled": 72, "statuses": dict(Counter(row["status"] for row in selected)),
                          "wins": sum(row["score"] == 1 for row in terminal),
                          "draws": sum(row["score"] == 0.5 for row in terminal),
                          "losses": sum(row["score"] == 0 for row in terminal),
                          "planned_denominator_score_bounds": bounds[arm]}
    delta_bounds = [bounds["student"][0] - bounds["initial"][1],
                    bounds["student"][1] - bounds["initial"][0]]
    alpha = protocol["evaluation"]["per_contrast_alpha"]
    radius = math.sqrt(2 * math.log(2 / alpha) / 36)
    interval = [max(-1, delta_bounds[0] - radius), min(1, delta_bounds[1] + radius)]
    verdict = "improved_vs_fixed_uniform_anchor" if interval[0] > 0 else (
        "worse_vs_fixed_uniform_anchor" if interval[1] < 0 else "inconclusive_playing_strength")
    return {"schema": "mew-student-stage1-readout/1", "status": "audited_receipt_summary",
            "scope": "Historical six-deck policy transfer, frozen uniform legal opponent. No current-Standard, ranked, PPO or league claim.",
            "identities": {"protocol_sha256": manifest["protocol_sha256"],
                           "artifact_manifest_sha256": manifest["artifact_manifest_sha256"],
                           "training_result_sha256": sha(training / "training-result.json"),
                           "evaluation_games_sha256": sha(evaluation / "evaluation-games.jsonl"),
                           "initial_checkpoint_sha256": sha(training / "initial.pt"),
                           "student_checkpoint_sha256": sha(training / "student.pt")},
            "updates": train["updates"], "training_budget": train["budget"],
            "probe": {"frames": before["count"],
                      "initial_cross_entropy": before["mean_cross_entropy"],
                      "student_cross_entropy": after["mean_cross_entropy"],
                      "initial_top1_agreement": before["top1_agreement"],
                      "student_top1_agreement": after["top1_agreement"],
                      "limitation": "Fixed initial-student held-out states; correlated frames and no claim of distribution-wide generalization."},
            "evaluation": summaries, "student_minus_initial_score_bounds": delta_bounds,
            "paired_difference_family_adjusted_interval": interval,
            "uncertainty": {"method": "Hoeffding over 36 independent deck-pair clusters, two seats per cluster, bounded paired differences in [-1,1]; unresolved games take worst-case outcomes.",
                            "per_contrast_alpha": alpha, "family_alpha": protocol["evaluation"]["family_alpha"]},
            "verdict": verdict,
            "limitations": ["Frozen-parameter and zero teacher use assertions depend on runner receipts and source inspection; this summarizer does not instrument the process.",
                            "One fresh student seed and weak uniform opponent cannot establish general competitive strength.",
                            "Auto-forced native transitions are separately counted environment work, not labeled learner examples.",
                            "Policy-only training can change critic outputs through the shared encoder although value-head parameters remain frozen."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.training, args.evaluation)
    with args.receipt.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "verdict": result["verdict"]}))
