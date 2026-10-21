"""Independently audit amended evaluation receipts using only the standard library."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "86-tcg-agent/native-checkpoint-evaluation-288/1"
PARENT_SCHEMA = "86-tcg-agent/native-checkpoint-evaluation/1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def same(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    if type(left) in (int, float) and type(right) in (int, float):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return type(left) is type(right) and left == right


def assignments(protocol, repetitions):
    evaluation = protocol["evaluation"]
    if evaluation["seed"] != 862001 or evaluation["seed_domain"] != "native-sustained-eval576-v1":
        raise ValueError("unexpected parent seed domain")
    def seed(kind, left, right, repetition, seat=None):
        identity = [evaluation["seed"], evaluation["seed_domain"], kind, left, right, repetition]
        if seat is not None:
            identity.append(seat)
        return int.from_bytes(hashlib.sha256(canonical(identity)).digest()[:4], "little") or 1
    return [{"index": (left * 6 + right) * 16 + repetition * 2 + seat,
             "cluster_index": (left * 6 + right) * 8 + repetition,
             "repetition": repetition, "learner_deck": left, "opponent_deck": right,
             "learner_seat": seat, "engine_seed": seed("engine", left, right, repetition),
             "learner_seed": seed("learner", left, right, repetition, seat),
             "opponent_seed": seed("opponent", left, right, repetition, seat)}
            for left in range(6) for right in range(6) for repetition in range(repetitions) for seat in (0, 1)]


def validate_rows(rows, plan):
    if len(rows) != len(plan) or len({row["index"] for row in rows}) != len(plan):
        raise ValueError("game receipt coverage differs")
    for row, expected in zip(rows, plan):
        if any(type(row.get(key)) is not type(value) or row[key] != value for key, value in expected.items()):
            raise ValueError("game assignment differs from independent schedule")
        if row["status"] not in {"terminal", "fault", "censored", "not_started"}:
            raise ValueError("unknown game status")
        if row["status"] == "terminal":
            winner = row["winner"]
            if type(winner) is not int or winner not in (0, 1, 2):
                raise ValueError("terminal winner is invalid")
            score = 0.5 if winner == 2 else int(winner == row["learner_seat"])
            if type(row["score"]) not in (int, float) or row["score"] != score:
                raise ValueError("terminal learner perspective differs")
        elif row["score"] is not None:
            raise ValueError("unresolved outcome was scored")


def summary(rows):
    terminal = [row for row in rows if row["status"] == "terminal"]
    scores = [row["score"] for row in terminal]
    unknown, counters = Counter(), Counter()
    choices, genuine, walls = [], [], []
    for row in rows:
        for key, missing, destination in (("learner_genuine_choices", "genuine_choices", genuine),
                                           ("wall_seconds", "game_wall_seconds", walls)):
            if row.get(key) is not None:
                value = row[key]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("negative or nonfinite game accounting")
                destination.append(value)
            elif row["status"] != "not_started":
                unknown[missing] += 1
        if row.get("choices") is None:
            if row["status"] != "not_started":
                unknown["learner_choices"] += 1
        else:
            values = [choice["learner"] for choice in row["choices"]]
            if any(type(choice["learner"]) is not bool or type(choice["actor_seat"]) is not int
                   or choice["actor_seat"] not in (0, 1)
                   or choice["learner"] != (choice["actor_seat"] == row["learner_seat"])
                   for choice in row["choices"]):
                raise ValueError("learner choice perspective is invalid")
            choices.append(sum(values))
        if row.get("native_stats") is None:
            if row["status"] != "not_started":
                unknown["native_counters"] += 1
        else:
            if any(type(value) is not int or value < 0 for value in row["native_stats"].values()):
                raise ValueError("native counters are invalid")
            counters.update(row["native_stats"])
    return {"planned_games": len(rows), "statuses": dict(Counter(row["status"] for row in rows)),
            "wins": scores.count(1), "draws": scores.count(0.5), "losses": scores.count(0),
            "planned_denominator_score_bounds": [sum(scores) / len(rows), (sum(scores) + len(rows) - len(terminal)) / len(rows)],
            "recorded_learner_choices": sum(choices), "genuine_learner_choices": None if unknown["genuine_choices"] else sum(genuine),
            "recorded_genuine_learner_choices": sum(genuine), "games_with_unknown_accounting": dict(unknown),
            "native_counters_complete": not bool(unknown["native_counters"]), "native_counters": dict(counters),
            "games_with_recorded_wall_seconds": len(walls), "games_with_native_counters": sum(bool(row.get("native_stats")) for row in rows),
            "summed_game_wall_seconds": sum(walls)}


def contrast(candidate, reference):
    grouped = {}
    for left, right in zip(candidate, reference):
        cluster = left["cluster_index"]
        grouped.setdefault(cluster, []).append((left, right))
    if len(grouped) != 144 or any([left["learner_seat"] for left, _ in group] != [0, 1] for group in grouped.values()):
        raise ValueError("paired cluster coverage differs")
    lower, upper = [], []
    for group in grouped.values():
        low = high = 0
        for left, right in group:
            a = (left["score"], left["score"]) if left["status"] == "terminal" else (0, 1)
            b = (right["score"], right["score"]) if right["status"] == "terminal" else (0, 1)
            low += (a[0] - b[1]) / 2
            high += (a[1] - b[0]) / 2
        lower.append(low)
        upper.append(high)
    bounds = [sum(lower) / len(grouped), sum(upper) / len(grouped)]
    alpha = 0.05 / 14
    radius = math.sqrt(2 * math.log(2 / alpha) / len(grouped))
    interval = [max(-1, bounds[0] - radius), min(1, bounds[1] + radius)]
    return {"clusters": 144, "difference_bounds": bounds, "interval": interval, "radius": radius,
            "per_comparison_alpha": alpha, "verdict": "improved_against_uniform_anchor" if interval[0] > 0
            else "worse_against_uniform_anchor" if interval[1] < 0 else "inconclusive"}


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, condition):
        self.checks.append({"check": name, "passed": bool(condition)})
        if not condition:
            raise ValueError(name)

    def checkpoint(self, checkpoint):
        row = checkpoint["descriptor"]
        self.check("immutable checkpoint receipt", sha(checkpoint["receipt_path"]) == checkpoint["receipt_sha256"]
                   and read(checkpoint["receipt_path"]) == row and row["status"] == "completed")
        self.check("checkpoint weight and optimizer-state identities", all(
            Path(row[key + "_path"]).is_absolute() and sha(row[key + "_path"]) == row[key + "_sha256"]
            for key in ("weights", "state")))

    def receipt(self, directory, repetitions):
        manifest, result = read(directory / "manifest.json"), read(directory / "evaluation-result.json")
        self.check("manifest and raw-game receipt identities", sha(directory / "manifest.json") == result["manifest_sha256"]
                   and sha(directory / "evaluation-games.jsonl") == result["evaluation_games_sha256"])
        plan = assignments(manifest["protocol"], repetitions)
        self.check("independent complete assignment schedule", read(directory / "evaluation-plan.json") == {"assignments": plan}
                   and sha(directory / "evaluation-plan.json") == manifest["assignment_plan_sha256"])
        rows = [json.loads(line) for line in (directory / "evaluation-games.jsonl").read_bytes().splitlines()]
        validate_rows(rows, plan)
        self.check("all assignment outcomes and accounting", same(summary(rows), result["summary"]))
        self.checkpoint(manifest["checkpoint"])
        self.check("result checkpoint identities", result["checkpoint_weights_sha256"] == manifest["checkpoint"]["descriptor"]["weights_sha256"]
                   and result["checkpoint_receipt_sha256"] == manifest["checkpoint"]["receipt_sha256"])
        return manifest, result, rows


def run(args, audit):
    manifest, result, rows = audit.receipt(args.evaluation, 4)
    audit.check("amended result schema and completion classification", manifest["schema"] == result["schema"] == SCHEMA
                and result["status"] == ("complete" if all(row["status"] == "terminal" for row in rows) else "completed_with_unresolved"))
    audit.check("checkpoint file immutability receipt agrees with current identities", result["checkpoint_files_unchanged"] is True)
    protocol = manifest["protocol"]
    amendment = read(args.evaluation / "amendment.json")
    audit.check("parent protocol and amendment snapshots", read(args.evaluation / "protocol.json") == protocol
                and sha(args.evaluation / "protocol.json") == manifest["protocol_file_sha256"]
                and digest(protocol) == manifest["protocol_sha256"]
                and sha(args.evaluation / "amendment.json") == manifest["amendment_sha256"]
                and amendment == manifest["amendment"]
                and amendment["parent_protocol_file_sha256"] == manifest["protocol_file_sha256"]
                and amendment["parent_protocol_sha256"] == manifest["protocol_sha256"]
                and amendment["games_per_checkpoint"] == 288 and amendment["clusters"] == 144
                and amendment["pair_repetitions"] == [0, 1, 2, 3])
    bindings = read(manifest["bindings_path"])
    audit.check("original bound source closure and current hashes", sha(manifest["bindings_path"]) == manifest["bindings_sha256"]
                and bindings["artifacts"] == manifest["source_artifacts"]
                and digest(bindings["artifacts"]) == bindings["artifact_manifest_sha256"] == manifest["artifact_manifest_sha256"]
                and all(sha(path) == expected for path, expected in bindings["artifacts"].items()))
    audit.check("amended evaluator source closure and current hashes", bool(manifest["amendment_sources"])
                and all(sha(path) == expected for path, expected in manifest["amendment_sources"].items())
                and {Path(path).name for path in manifest["amendment_sources"]} == {
                    "evaluate_native_checkpoint_288.py", "mew-evaluation-288-v1.json", "evaluate_native_checkpoint.py", "native_distillation.py"})
    audit.check("engine and deck identities", all(sha(bindings[path]) == bindings[key] == manifest[key] for path, key in (
                ("engine_library", "engine_sha256"), ("deck_manifest", "deck_manifest_sha256"))))
    stats = {"clusters": 144, "family_alpha": 0.05, "max_checkpoints": 7,
             "comparisons_per_checkpoint": 2, "per_comparison_alpha": 0.05 / 14}
    audit.check("paired multiplicity allocation", same(stats, manifest["statistics"])
                and all(amendment[key] == stats[key] for key in ("family_alpha", "comparisons_per_checkpoint", "per_comparison_alpha"))
                and amendment["max_evaluation_checkpoints"] == 7)
    for key, expected in (("protocol_sha256", manifest["protocol_sha256"]),
                          ("protocol_file_sha256", manifest["protocol_file_sha256"]),
                          ("source_artifact_manifest_sha256", manifest["artifact_manifest_sha256"]),
                          ("engine_sha256", manifest["engine_sha256"]), ("deck_manifest_sha256", manifest["deck_manifest_sha256"])):
        audit.check("candidate checkpoint lineage " + key, manifest["checkpoint"]["descriptor"]["lineage"][key] == expected)
    for arm, provenance in manifest["reference_evaluations"].items():
        reference_path = Path(provenance["receipt_path"])
        reference_result = read(reference_path)
        audit.check(arm + " supported reference receipt schema", reference_result["schema"] in (PARENT_SCHEMA, SCHEMA))
        source_repetitions = 8 if reference_result["schema"] == PARENT_SCHEMA else 4
        original, original_result, source_rows = audit.receipt(reference_path.parent, source_repetitions)
        requested = manifest["references"][arm]
        audit.checkpoint(requested)
        before, wanted = original["checkpoint"]["descriptor"], requested["descriptor"]
        audit.check(arm + " immutable complete reference and identical actor", sha(reference_path) == provenance["receipt_sha256"]
                    and original_result["status"] == "complete" and all(row["status"] == "terminal" for row in source_rows)
                    and original_result["checkpoint_files_unchanged"] is True and original_result["frozen_player_unchanged"] is True
                    and before["actor_sha256"] == wanted["actor_sha256"]
                    and before["model_identity"]["config"] == wanted["model_identity"]["config"]
                    and before["model_identity"]["schema_digest"] == wanted["model_identity"]["schema_digest"])
        audit.check(arm + " reference environment and parent protocol", all(original[key] == manifest[key] for key in (
                    "protocol_sha256", "protocol_file_sha256", "artifact_manifest_sha256", "engine_sha256", "deck_manifest_sha256")))
        audit.check(arm + " reference checkpoint lineage", all(wanted["lineage"][key] == before["lineage"][key]
                    == manifest["checkpoint"]["descriptor"]["lineage"][key] for key in (
                    "protocol_sha256", "protocol_file_sha256", "source_artifact_manifest_sha256", "engine_sha256", "deck_manifest_sha256")))
        if source_repetitions == 4:
            audit.check(arm + " amended reference identity", original["amendment_sha256"] == manifest["amendment_sha256"])
        selected = [row for row in source_rows if row["repetition"] < 4]
        validate_rows(selected, assignments(protocol, 4))
        audit.check(arm + " balanced subset provenance", provenance["source_games"] == len(source_rows)
                    and provenance["selected_reference_games"] == 288 and provenance["new_reference_games_executed"] == 0
                    and provenance["identical_actor_reuse"] is True and provenance["actor_sha256"] == wanted["actor_sha256"]
                    and provenance["requested_checkpoint_weights_sha256"] == wanted["weights_sha256"]
                    and provenance["source_checkpoint_weights_sha256"] == before["weights_sha256"]
                    and provenance["assignment_selection"] == "parent_repetition_0_1_2_3_all_pairs_both_seats"
                    and provenance["reference_kind"] == ("subset_derived_from_576" if source_repetitions == 8 else "amended_288")
                    and provenance["selected_rows_sha256"] == digest(selected))
        audit.check(arm + " independently reconstructed paired contrast", same(contrast(rows, selected), result["contrasts"][arm]))
    worker_path = args.evaluation / "worker-result.json"
    if worker_path.exists():
        worker = read(worker_path)
        charged = worker["budget"]["admitted_learner_microsteps"]
        audit.check("worker accounting and recorded-choice lower bound", result["worker_budget"] == worker["budget"]
                    and result["charged_evaluation_learner_microsteps"] == charged
                    and result["evaluation_learner_microstep_bounds"] == [charged, charged]
                    and result["summary"]["recorded_learner_choices"] <= charged <= manifest["limits"]["admitted_learner_microsteps"]
                    and result["frozen_player_unchanged"] == worker["frozen_player_unchanged"])
        audit.check("worker records a frozen player and zero teacher queries", worker["frozen_player_unchanged"] is True
                    and worker["teacher_model_loads"] == worker["new_teacher_queries"] == 0
                    and result["teacher_model_loads"] == result["new_teacher_queries"] == 0)
    else:
        audit.check("missing-worker cost remains explicitly unknown", result["charged_evaluation_learner_microsteps"] is None
                    and result["frozen_player_unchanged"] is None
                    and result["evaluation_learner_microstep_bounds"] == [result["summary"]["recorded_learner_choices"], manifest["limits"]["admitted_learner_microsteps"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.evaluation, args.output = args.evaluation.resolve(strict=True), args.output.resolve()
    if args.output.is_relative_to(ROOT) or args.output.is_relative_to(args.evaluation):
        parser.error("write a new private receipt outside the checkout and original evaluation")
    if args.output.exists():
        raise FileExistsError(args.output)
    audit, error = Audit(), None
    try:
        run(args, audit)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    receipt = {"schema": "86-tcg-agent/native-evaluation-288-audit/1", "status": "passed" if error is None else "issues",
               "audit_source_sha256": sha(__file__), "evaluation": str(args.evaluation), "checks": audit.checks,
               "passed_checks": sum(row["passed"] for row in audit.checks), "error": error,
               "models_loaded": 0, "native_games_started": 0, "optimizer_updates": 0,
               "limitations": ["Checks saved receipts and current bytes. Does not observe historical model execution or independently prove actor visibility.",
                               "Worker immutability and complete choice charging retain their recorded attestation boundary.",
                               "Campaign-wide checkpoint count, scheduling and interrupted historical costs require the separate campaign receipts."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "passed_checks": receipt["passed_checks"], "error": error, "receipt_sha256": sha(args.output)}))
    return int(error is not None)


if __name__ == "__main__":
    raise SystemExit(main())
