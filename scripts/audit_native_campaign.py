"""Audit immutable sustained-campaign receipts without loading models or games."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


@dataclass(frozen=True)
class Finding:
    requirement: str
    status: str
    evidence: object


class Audit:
    def __init__(self):
        self.findings = []

    def check(self, requirement, passed, evidence):
        self.findings.append(Finding(requirement, "pass" if passed else "fail", evidence))

    def missing(self, requirement, evidence):
        self.findings.append(Finding(requirement, "not_yet_evidenced", evidence))


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def read_lines(path):
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    complete = [json.loads(line) for line in lines if line.endswith(b"\n")]
    return complete, bool(lines and not lines[-1].endswith(b"\n"))


def seed32(seed, domain, *identity):
    return int.from_bytes(hashlib.sha256(canonical([seed, domain, *identity])).digest()[:4], "little") or 1


def plan(protocol):
    evaluation = protocol["evaluation"]
    assignments = []
    for left in range(6):
        for right in range(6):
            for repetition in range(8):
                for seat in (0, 1):
                    fields = (left, right, repetition)
                    assignments.append({"index": len(assignments), "cluster_index": len(assignments) // 2,
                        "repetition": repetition, "learner_deck": left, "opponent_deck": right,
                        "learner_seat": seat,
                        "engine_seed": seed32(evaluation["seed"], evaluation["seed_domain"], "engine", *fields),
                        "learner_seed": seed32(evaluation["seed"], evaluation["seed_domain"], "learner", *fields, seat),
                        "opponent_seed": seed32(evaluation["seed"], evaluation["seed_domain"], "opponent", *fields, seat)})
    return assignments


def contrast(candidate, reference, alpha):
    bounds = [0.0, 0.0]
    for index in range(0, 576, 2):
        cluster = [0.0, 0.0]
        for a, b in zip(candidate[index:index + 2], reference[index:index + 2]):
            a_bounds = (a["score"], a["score"]) if a["status"] == "terminal" else (0, 1)
            b_bounds = (b["score"], b["score"]) if b["status"] == "terminal" else (0, 1)
            cluster[0] += (a_bounds[0] - b_bounds[1]) / 2
            cluster[1] += (a_bounds[1] - b_bounds[0]) / 2
        for side in range(2):
            bounds[side] += cluster[side] / 288
    radius = math.sqrt(2 * math.log(2 / alpha) / 288)
    interval = [max(-1.0, bounds[0] - radius), min(1.0, bounds[1] + radius)]
    return bounds, radius, interval


def audit_rows(audit, directory, protocol):
    rows, partial = read_lines(directory / "evaluation-games.jsonl")
    assignments = plan(protocol)
    required = set(assignments[0]) | {"status", "winner", "score"}
    malformed = [index for index, row in enumerate(rows)
                 if not isinstance(row, dict) or not required <= row.keys()
                 or type(row["index"]) is not int
                 or (row.get("choices") is not None and (not isinstance(row["choices"], list)
                     or any(not isinstance(choice, dict) or type(choice.get("learner")) is not bool for choice in row["choices"])))]
    if malformed or partial or len(rows) != 576:
        audit.check(f"{directory.name}: complete evaluation row structure", False,
                    {"rows": len(rows), "malformed_row_indices": malformed, "trailing_partial_line": partial})
        return None
    valid = len({row["index"] for row in rows}) == 576
    valid = valid and all(all(row.get(key) == value for key, value in expected.items())
                          for row, expected in zip(rows, assignments))
    statuses = Counter(row["status"] for row in rows)
    valid = valid and set(statuses) <= {"terminal", "fault", "censored", "not_started"}
    for row in rows:
        if row["status"] == "terminal":
            winner = row.get("winner")
            score = 0.5 if winner == 2 else float(winner == row["learner_seat"])
            valid = valid and winner in (0, 1, 2) and row.get("score") == score
        else:
            valid = valid and row.get("score") is None
    audit.check(f"{directory.name}: all frozen assignments and outcome perspectives", valid,
                {"rows": len(rows), "statuses": dict(statuses), "trailing_partial_line": partial})
    return rows


def audit_evaluations(audit, campaign, protocol, closure_sha):
    assignments = plan(protocol)
    expected_plan_sha = hashlib.sha256(canonical({"assignments": assignments}) + b"\n").hexdigest()
    alpha = protocol["evaluation"]["family_alpha"] / (protocol["evaluation"]["max_evaluation_checkpoints"] * 2)
    evaluation = protocol["evaluation"]
    audit.check("declared evaluation multiplicity", close(alpha, evaluation["per_comparison_alpha"])
                and evaluation["max_evaluation_checkpoints"] == 7 and evaluation["family_alpha"] == 0.05
                and evaluation["games_per_checkpoint"] == 576 and evaluation["pair_repetitions"] == 8
                and evaluation["seat_swap"] is True,
                {"checkpoints": 7, "primary_contrasts_per_checkpoint": 2, "alpha": alpha})
    engine_seeds = [row["engine_seed"] for row in assignments[::2]]
    action_seeds = [row[key] for row in assignments for key in ("learner_seed", "opponent_seed")]
    audit.check("evaluation independent pair/action seed domains and swapped seats",
                len(set(engine_seeds + action_seeds)) == len(engine_seeds + action_seeds)
                and all(a["engine_seed"] == b["engine_seed"] and (a["learner_seat"], b["learner_seat"]) == (0, 1)
                        for a, b in zip(assignments[::2], assignments[1::2])),
                {"independent_pair_seeds": len(engine_seeds), "action_seeds": len(action_seeds)})
    completed = []
    for directory in sorted(campaign.glob("evaluation-[0-9][0-9]")):
        path = directory / "evaluation-result.json"
        if not path.exists():
            audit.missing(f"{directory.name}: completed evaluation", "No completed result receipt in this snapshot")
            continue
        result, manifest = read(path), read(directory / "manifest.json")
        audit.check(f"{directory.name}: receipt and plan hashes",
            sha(directory / "manifest.json") == result["manifest_sha256"]
            and sha(directory / "evaluation-plan.json") == manifest["assignment_plan_sha256"] == expected_plan_sha
            and sha(directory / "evaluation-games.jsonl") == result["evaluation_games_sha256"]
            and manifest["artifact_manifest_sha256"] == closure_sha
            and manifest["protocol"] == protocol,
            {"result_sha256": sha(path), "plan_sha256": manifest["assignment_plan_sha256"]})
        rows = audit_rows(audit, directory, protocol)
        if rows is None:
            completed.append({"directory": directory.name, "result_sha256": sha(path), "status": "malformed_game_receipts"})
            continue
        summary = result["summary"]
        terminal = [row for row in rows if row["status"] == "terminal"]
        score = sum(row["score"] for row in terminal)
        score_bounds = [score / 576, (score + 576 - len(terminal)) / 576]
        expected = {"wins": sum(row["score"] == 1 for row in terminal),
                    "draws": sum(row["score"] == 0.5 for row in terminal),
                    "losses": sum(row["score"] == 0 for row in terminal)}
        audit.check(f"{directory.name}: score and unresolved denominator",
            all(summary[key] == value for key, value in expected.items())
            and all(close(a, b) for a, b in zip(summary["planned_denominator_score_bounds"], score_bounds))
            and summary["statuses"] == dict(Counter(row["status"] for row in rows)), expected)
        known_choices = sum(sum(bool(choice["learner"]) for choice in (row.get("choices") or [])) for row in rows)
        unknown_choices = sum(row.get("choices") is None for row in rows)
        charged = result["charged_evaluation_learner_microsteps"]
        audit.check(f"{directory.name}: recorded choice accounting", known_choices == summary["recorded_learner_choices"]
                    and (charged is None or known_choices <= charged <= protocol["limits"]["evaluation_learner_microsteps_per_checkpoint"]),
                    {"recorded_choices_lower_bound": known_choices, "games_with_unknown_choice_records": unknown_choices,
                     "recorded_choice_accounting_complete": not bool(unknown_choices), "charged": charged})
        for arm, stored in result["contrasts"].items():
            if stored.get("status") == "reference_not_evaluated":
                audit.missing(f"{directory.name}: {arm} paired contrast", stored["status"])
                continue
            provenance = manifest["reference_evaluations"][arm]
            reference_path = Path(provenance["receipt_path"])
            reference_result = read(reference_path)
            reference_rows = audit_rows(audit, reference_path.parent, protocol)
            if reference_rows is None:
                continue
            audit.check(f"{directory.name}: {arm} reference hashes",
                sha(reference_path) == provenance["receipt_sha256"]
                and sha(reference_path.parent / "evaluation-games.jsonl") == reference_result["evaluation_games_sha256"],
                {"reference_receipt_sha256": provenance["receipt_sha256"]})
            bounds, radius, interval = contrast(rows, reference_rows, alpha)
            audit.check(f"{directory.name}: {arm} paired multiplicity bounds",
                stored["clusters"] == 288 and close(stored["per_comparison_alpha"], alpha)
                and close(stored["radius"], radius)
                and all(close(a, b) for a, b in zip(stored["difference_bounds"], bounds))
                and all(close(a, b) for a, b in zip(stored["interval"], interval)),
                {"difference_bounds": bounds, "radius": radius, "interval": interval})
        completed.append({"directory": directory.name, "result_sha256": sha(path), "statuses": summary["statuses"]})
    if not completed:
        audit.missing("completed 576-game evaluation", "No finished evaluation receipt yet")
    return completed


def audit_calibration(audit, training):
    path = training / "stage3-result.json"
    if not path.exists():
        audit.missing("calibration receipt numeric crosscheck", "Stage 3 result is not complete")
        return
    result = read(path)
    metrics = {}
    for name in ("heldout_before", "heldout_after"):
        stored = result[name]
        rows = stored["rows"]
        count = len(rows)
        mse = sum((row["prediction"] - row["target"]) ** 2 for row in rows) / count
        bias = sum(row["prediction"] - row["target"] for row in rows) / count
        constant = sum((stored["constant_training_mean"] - row["target"]) ** 2 for row in rows) / count
        audit.check(f"calibration {name}: stored prediction arithmetic",
                    count == stored["count"] and close(mse, stored["mse"])
                    and close(bias, stored["mean_bias"]) and close(constant, stored["constant_mse"]),
                    {"count": count, "mse": mse, "bias": bias, "constant_mse": constant,
                     "scope": "Recomputed receipt arithmetic. Predictions were not independently reforwarded."})
        metrics[name] = (mse, bias, constant)
    audit.check("calibration fixed heldout identities and targets",
        [(row["frame_digest"], row["target"]) for row in result["heldout_before"]["rows"]]
        == [(row["frame_digest"], row["target"]) for row in result["heldout_after"]["rows"]],
        {"before_mse": metrics["heldout_before"][0], "after_mse": metrics["heldout_after"][0]})


def run(args):
    audit = Audit()
    campaign, training = args.campaign, args.campaign / "training"
    campaign_manifest, training_manifest = read(campaign / "campaign-manifest.json"), read(training / "manifest.json")
    protocol, lineage = training_manifest["protocol"], training_manifest["lineage"]
    artifacts = training_manifest["source_artifact_manifest"]
    closure_sha = hashlib.sha256(canonical(artifacts)).hexdigest()
    audit.check("source artifact manifest digest", closure_sha == lineage["source_artifact_manifest_sha256"]
                == campaign_manifest["source_artifact_manifest_sha256"], {"artifacts": len(artifacts), "sha256": closure_sha})
    changed = [Path(path).name for path, expected in artifacts.items() if not Path(path).is_file() or sha(path) != expected]
    audit.check("all pinned source artifacts unchanged", not changed, {"changed_artifact_names": changed})
    audit.check("campaign and trainer protocol agree", campaign_manifest["protocol"] == protocol
                and hashlib.sha256(canonical(protocol)).hexdigest() == lineage["protocol_sha256"]
                and campaign_manifest["protocol_sha256"] == lineage["protocol_file_sha256"],
                {"canonical_protocol_sha256": lineage["protocol_sha256"]})
    evaluation_manifests = sorted(campaign.glob("evaluation-[0-9][0-9]/manifest.json"))
    if evaluation_manifests:
        evaluation_manifest = read(evaluation_manifests[0])
        bindings_path = Path(evaluation_manifest["bindings_path"])
        bindings = read(bindings_path)
        audit.check("actual private bindings bytes and artifact map",
                    sha(bindings_path) == lineage["bindings_sha256"] == campaign_manifest["bindings_sha256"]
                    and bindings["artifacts"] == artifacts,
                    {"bindings_sha256": sha(bindings_path)})
        for key, identity in (("teacher_checkpoint", "teacher"), ("engine_library", "engine"), ("deck_manifest", "deck_manifest")):
            audit.check(f"actual {identity} file and declared identity",
                        sha(bindings[key]) == lineage[identity + "_sha256"],
                        {"sha256": lineage[identity + "_sha256"]})
        audit.check("verbatim evaluation protocol copy",
                    sha(evaluation_manifests[0].parent / "protocol.json") == lineage["protocol_file_sha256"],
                    {"protocol_file_sha256": lineage["protocol_file_sha256"]})
    else:
        audit.missing("private bindings bytes and verbatim protocol", "No evaluation manifest with bindings location exists yet")
    for kind in ("teacher", "engine", "deck_manifest"):
        audit.check(f"{kind} identity present in verified closure", lineage[kind + "_sha256"] in artifacts.values(),
                    {"sha256": lineage[kind + "_sha256"]})
    latest = read(training / "latest-checkpoint.json")
    match = re.fullmatch(r"checkpoint-(\d{6})", latest["checkpoint_id"])
    if not match:
        raise ValueError("invalid completed checkpoint sequence")
    checkpoints = []
    fixed_lineage = ("protocol_sha256", "protocol_file_sha256", "bindings_sha256", "teacher_sha256", "engine_sha256",
                     "deck_manifest_sha256", "source_artifact_manifest_sha256", "campaign_start_monotonic")
    for sequence in range(int(match[1]) + 1):
        path = training / f"checkpoint-{sequence:06d}.json"
        row = read(path)
        valid = row["schema"] == "86-tcg-agent/native-sustained-checkpoint/1" and row["status"] == "completed"
        valid = valid and all(row["lineage"][key] == lineage[key] for key in fixed_lineage)
        valid = valid and all(sha(row[kind + "_path"]) == row[kind + "_sha256"] for kind in ("weights", "state"))
        valid = valid and row["training_updates_total"] == sequence and row["learner_steps_total"] <= protocol["limits"]["training_learner_microsteps"]
        valid = valid and row["training_wall_seconds"] <= protocol["limits"]["training_stop_campaign_seconds"]
        if checkpoints:
            previous = checkpoints[-1]
            same_stage = row["stage"] == previous["stage"]
            valid = valid and row["stage"] in (1, 3, 4) and row["stage"] >= previous["stage"]
            valid = valid and row["lineage"]["parent_checkpoint_sha256"] == previous["weights_sha256"]
            valid = valid and row["stage_updates"] == (previous["stage_updates"] + 1 if same_stage else 1)
            valid = valid and row["optimizer_steps_total"] > previous["optimizer_steps_total"]
            valid = valid and row["ppo_updates"] == previous["ppo_updates"] + int(row["stage"] == 4)
            valid = valid and row["learner_steps_total"] >= previous["learner_steps_total"]
            valid = valid and row["model_identity"]["config"] == previous["model_identity"]["config"]
            valid = valid and row["model_identity"]["schema_digest"] == previous["model_identity"]["schema_digest"]
            data_file = (training / f"stage1-update-{row['stage_updates']:04d}-labels.pt" if row["stage"] == 1 else
                         training / "stage3-terminal-examples.pt" if row["stage"] == 3 else
                         training / f"stage4-update-{row['stage_updates']:04d}-episodes.pt")
            valid = valid and sha(data_file) == row["lineage"]["data_sha256"]
            if row["stage"] == 1:
                valid = valid and sha(training / "stage1-heldout-probes.pt") == row["lineage"]["probe_data_sha256"]
        else:
            valid = valid and row["stage"] == 0 and row["optimizer_steps_total"] == row["ppo_updates"] == 0
        audit.check(f"checkpoint {sequence}: artifact hashes, lineage and committed counters", valid,
                    {"receipt_sha256": sha(path), "stage": row["stage"], "training_updates_total": row["training_updates_total"],
                     "optimizer_steps_total": row["optimizer_steps_total"], "learner_steps_total": row["learner_steps_total"]})
        checkpoints.append(row)
    audit.check("atomic latest pointer matches its completed receipt",
                {key: value for key, value in latest.items() if key != "receipt_sha256"} == checkpoints[-1]
                and sha(latest["receipt_path"]) == latest["receipt_sha256"],
                {"snapshot_checkpoint": latest["checkpoint_id"]})
    episode_snapshots = {}
    for path in sorted(training.glob("*-episodes.jsonl")):
        rows, partial = read_lines(path)
        statuses = Counter(row["status"] for row in rows)
        has_choices = path.name.startswith("stage1-")
        recorded = [sum(bool(choice["learner"]) for choice in row["choices"])
                    if has_choices and row.get("choices") is not None else
                    row.get("retained_learner_steps") if not has_choices else None for row in rows]
        lower = sum(value for value in recorded if value is not None)
        unknown_count = sum(value is None for value in recorded)
        episode_snapshots[path.name] = {"rows": len(rows), "statuses": dict(statuses), "trailing_partial_line": partial,
            "recorded_learner_choice_lower_bound": lower,
            "episodes_with_unknown_recorded_learner_count": unknown_count,
            "recorded_count_complete": not bool(unknown_count),
            "exact_charged_count_reconstructible": False,
            "scope": "Collector can charge before recording a failed inference. Calibration additionally caps retained frames. The currently running game may have no episode receipt yet."}
        audit.check(f"{path.name}: episode status accounting", set(statuses) <= {"terminal", "fault", "censored"},
                    episode_snapshots[path.name])
    audit_calibration(audit, training)
    evaluations = audit_evaluations(audit, campaign, protocol, closure_sha)
    audit.missing("exact stage-3 charged learner-choice reconstruction",
        "After campaign completion, declare a separate bounded, teacher-free replay of all 32 stage-3 games from frozen stage-1 checkpoint-000003 and original seeds/opponent. Compare every retained frame/action/old-value, natural outcome, explicit microsteps and native statistics with original receipts while independently counting all learner choices. The original retained-frame cap and both-seat microstep count cannot establish this total. No replay was executed by this audit.")
    for requirement in ("independent policy/critic frozen-parameter tensor comparisons", "checkpoint reload actor-logit equivalence",
                        "independent stage-1 label/probe separation and teacher-target recomputation",
                        "independent PPO old-logp, old-value, GAE and Monte Carlo target reconstruction"):
        audit.missing(requirement, "Requires the separate tensor/model audit. Receipt hashes and self-reported booleans do not prove this invariant.")
    counts = dict(Counter(finding.status for finding in audit.findings))
    result = {"schema": "86-tcg-agent/native-campaign-receipt-audit/1", "audit_source_sha256": sha(__file__),
              "created_unix_seconds": time.time(), "status": "issues" if counts.get("fail") else "receipt_checks_pass_with_open_gates",
              "snapshot_latest_checkpoint": latest["checkpoint_id"], "findings": [asdict(row) for row in audit.findings],
              "counts": counts, "episode_snapshots": episode_snapshots, "completed_evaluations": evaluations,
              "models_loaded": 0, "native_imports": 0, "games_started": 0,
              "scope": "Read-only receipt and numeric arithmetic audit of a potentially live campaign. This is not a full independent tensor audit, liveness proof or playing-strength claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(json.dumps({"status": result["status"], "counts": counts, "snapshot_checkpoint": latest["checkpoint_id"],
                      "audit_sha256": sha(args.output)}))
    return int(bool(counts.get("fail")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.campaign, args.output = args.campaign.resolve(), args.output.resolve()
    if args.output.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("raw campaign audit output must remain outside the public checkout")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
