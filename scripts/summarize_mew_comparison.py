"""Summarize audited offline comparison receipts without loading models or traces."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def spread(values):
    values = sorted(values)
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values), "invalid cost")
    if not values:
        return {"count": 0, "sum": 0, "min": None, "p50": None, "p95": None, "max": None}
    def percentile(fraction):
        position = (len(values) - 1) * fraction
        low = int(position)
        return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)
    return {"count": len(values), "sum": sum(values), "min": values[0], "p50": percentile(.5), "p95": percentile(.95), "max": values[-1]}


def scores(rows):
    observed = [score for _, score, _ in rows if score is not None]
    planned = len(rows)
    return {"planned": planned, "complete": len(observed), "unresolved": planned - len(observed),
            "wins": observed.count(1), "draws": observed.count(.5), "losses": observed.count(0),
            "planned_denominator_score_bounds": [sum(observed) / planned, (sum(observed) + planned - len(observed)) / planned]}


def contrast_summary(comparison, number, run, audited, final, version):
    rows, identities = [], {}
    statuses, errors = Counter(), Counter()
    for index, assignment in enumerate(comparison["games"]):
        name = f"{number:02d}-{index:03d}.json"
        path = run / name
        receipt = load(path) if path.exists() else None
        score = None
        if receipt is None:
            statuses["missing_receipt"] += 1
        else:
            identities[name] = sha(path)
            require(receipt["assignment"] == assignment, "receipt assignment mismatch")
            require(receipt["players_unchanged"] is True, "player state changed")
            status = receipt["status"]
            require(status in ("complete", "failed", "unrun_phase_cap"), "unknown receipt status")
            statuses[status] += 1
            if status == "complete":
                require(receipt["winner"] in (0, 1, 2), "invalid winner")
                score = .5 if receipt["winner"] == 2 else float(assignment["seat_arms"][receipt["winner"]] == comparison["candidate"])
            else:
                errors[receipt.get("error", {}).get("type", status)] += 1
        rows.append((assignment, score, receipt))
    aggregate = scores(rows)
    require(aggregate["planned"] == 72, "unexpected planned denominator")
    for ours, theirs in (("planned", "planned_games"), ("complete", "complete_games"), ("unresolved", "failed_or_unrun"), ("wins", "wins"), ("draws", "draws"), ("losses", "losses")):
        require(aggregate[ours] == audited[theirs], "audit snapshot differs from current receipts")
    require(dict(statuses) == audited["statuses"], "audit statuses differ from current receipts")
    bounds = aggregate["planned_denominator_score_bounds"]
    require(bounds == audited["score_bounds"], "audit score bounds differ")
    radius = math.sqrt(math.log(2 / .025) / (2 * 36))
    interval = [max(0, bounds[0] - radius), min(1, bounds[1] + radius)]
    require(interval == audited["simultaneous_hoeffding_interval"], "audit interval differs")
    complete = aggregate["complete"] == 72
    baseline = comparison["opponent"] == "baseline002_last_verified_endpoint"
    eligible = final and complete and not (version == 1 and baseline)
    verdict = "candidate_superior" if eligible and interval[0] > .5 else "candidate_inferior" if eligible and interval[1] < .5 else "inconclusive"
    executed = [receipt for _, _, receipt in rows if receipt is not None and receipt["status"] != "unrun_phase_cap"]
    complete_rows = [receipt for _, _, receipt in rows if receipt is not None and receipt["status"] == "complete"]
    failed_rows = [receipt for _, _, receipt in rows if receipt is not None and receipt["status"] == "failed"]
    timing_keys = sorted({key for receipt in executed for key in receipt["timing_seconds"]})
    model_calls, native_counts = Counter(), Counter()
    for receipt in executed:
        model_calls.update(receipt["mew_model_calls"])
        model_calls["baseline002_last_verified_endpoint"] += receipt["baseline_forward_calls"]
        native_counts.update(receipt["native_stats"])
    wall = spread([receipt["total_wall_seconds"] for receipt in executed])
    require(math.isclose(wall["sum"], audited["game_wall_seconds_including_failures"], rel_tol=1e-12, abs_tol=1e-9), "audit cost differs")
    by_seat = {str(seat): scores([row for row in rows if row[0]["seat_arms"][seat] == comparison["candidate"]]) for seat in (0, 1)}
    by_pair = {}
    for left in range(6):
        for right in range(6):
            subset = [row for row in rows if tuple(row[0]["seat_deck_indices"] if row[0]["leg"] == 0 else row[0]["seat_deck_indices"][::-1]) == (left, right)]
            by_pair[f"{left}-{right}"] = scores(subset)
    phase = None if final is None else final["comparisons"][number]["phase_wall_seconds"]
    require(phase is None or (math.isfinite(phase) and phase >= wall["sum"]), "invalid phase wall")
    return {"comparison_id": comparison["comparison_id"], **aggregate,
            "statuses": dict(statuses), "failure_types": dict(errors),
            "interval": {"method": "two_sided_Hoeffding_on_36_independent_swapped_seat_pair_means", "per_contrast_alpha": .025, "family_alpha": .05, "bounds": interval, "radius": radius},
            "confirmatory_eligible": bool(eligible), "verdict": verdict,
            "exclusion": "v1 baseline bridge schema faults require fresh v2 confirmation; never pool these scores" if version == 1 and baseline else None,
            "interpretation": "package contrast has unequal architecture, encoding, data and training compute" if baseline else "initialization contrast conditional on one predeclared matched random seed 860201",
            "candidate_seats": by_seat, "candidate_deck_opponent_deck": by_pair,
            "costs_seconds": {"executed_games_including_failures": wall,
                              "complete_games": spread([receipt["total_wall_seconds"] for receipt in complete_rows]),
                              "failed_games": spread([receipt["total_wall_seconds"] for receipt in failed_rows]),
                              "baseline_model_only_forward": spread([receipt["baseline_model_forward_seconds"] for receipt in executed]),
                              "mew_model_only_forward": spread([receipt["timing_seconds"]["forward"] for receipt in executed]),
                              "timed_components": {key: spread([receipt["timing_seconds"][key] for receipt in executed]) for key in timing_keys},
                              "phase_wall_including_between_game_checks": phase,
                              "between_game_and_phase_overhead": None if phase is None else phase - wall["sum"]},
            "counts": {"model_calls": dict(model_calls), "native_counters_including_failed_games": dict(native_counts),
                       "baseline_microsteps": sum(receipt["baseline_microsteps"] for receipt in executed),
                       "baseline_forced_microsteps": sum(receipt["baseline_forced_microsteps"] for receipt in executed),
                       "forced_mew_encoded_choices": sum(receipt["forced_mew_encoded_choices"] for receipt in executed)},
            "process_cumulative_peak_rss_bytes": max((receipt["process_peak_rss_bytes"] for receipt in executed), default=None),
            "receipt_sha256": identities}


def summarize(protocol_path, run, audit_path, allow_partial, comparison_id=None):
    protocol, audit = load(protocol_path), load(audit_path)
    require(audit.get("scope") != "schedule_only_no_game_evidence", "schedule-only audit has no game evidence")
    require(audit["schema"] == "mew-comparison-independent-audit/1", "unsupported audit schema")
    require(audit["status"] in ("pass", "incomplete"), "audit has not passed")
    require(audit["protocol_sha256"] == sha(protocol_path), "audit protocol mismatch")
    version = {"86-tcg-agent/mew-comparison/1": 1, "86-tcg-agent/mew-baseline-confirmation/2": 2}.get(protocol["schema"])
    require(version is not None and version == audit["protocol_version"], "unsupported protocol version")
    final_path = run / "result.json"
    final = load(final_path) if final_path.exists() else None
    require(allow_partial or (audit["status"] == "pass" and final is not None), "final readout needs final independent audit")
    require((final is not None) == audit["final_result_present"], "audit final-result snapshot mismatch")
    if final is not None:
        require(audit["status"] == "pass", "final result lacks passing audit")
        require(sha(final_path) == audit["final_result_sha256"], "final-result identity mismatch")
    launch, players = load(run / "launch.json"), load(run / "players.json")
    require(launch["protocol_sha256"] == sha(protocol_path) and launch["bindings_sha256"] == audit["bindings_sha256"], "launch audit identity mismatch")
    require(launch["runner_sha256"] == protocol["runner_sha256"] and launch["optimizer_updates"] == 0, "unexpected runner or training")
    require(len(protocol["comparisons"]) == len(audit["comparisons"]), "audit contrast count differs")
    comparisons = []
    for index, comparison in enumerate(protocol["comparisons"]):
        if comparison_id is not None and comparison["comparison_id"] != comparison_id:
            continue
        require(comparison["comparison_id"] == audit["comparisons"][index]["comparison_id"], "audit contrast identity differs")
        comparisons.append(contrast_summary(comparison, index, run, audit["comparisons"][index], final, version))
    require(comparisons, "requested contrast is absent")
    return {"schema": "mew-comparison-safe-readout/1", "status": "final_audited" if final is not None else "partial_audited_snapshot",
            "identities": {"summarizer_sha256": sha(__file__), "protocol_sha256": sha(protocol_path), "audit_sha256": sha(audit_path),
                           "auditor_sha256": audit["auditor_sha256"], "launch_sha256": sha(run / "launch.json"), "players_receipt_sha256": sha(run / "players.json"),
                           "result_sha256": sha(final_path) if final is not None else None, "bindings_sha256": audit["bindings_sha256"],
                           "runner_sha256": launch["runner_sha256"], "player_state_sha256": players["state_sha256"], "artifact_manifest_sha256": protocol["artifact_manifest_sha256"]},
            "runtime": launch["runtime"], "optimizer_updates": 0,
            "model_load_and_reader_warm_seconds": players["model_load_and_reader_warm_seconds"],
            "cost_quantiles": "linear interpolation at (n-1)*p across recorded games; missing and unrun games have no invented zero cost",
            "host_priority_scope": "v1 process priority changed from nice0 to nice19 during the run; all costs include uncontrolled concurrent host load" if version == 1 else "confirmation planned at Linux nice19 under concurrent host load; priority is not independently instrumented by this readout",
            "comparisons": comparisons,
            "limitations": ["Costs are descriptive under shared concurrent host load and low process priority, not isolated throughput, architecture speed, or training throughput estimates. Priority may have changed during v1.",
                            "Model load and reader warm time is shared across contrasts and is reported separately, not silently amortized.",
                            "Baseline forwards forced microsteps; Mew native auto-forced actions bypass model forward. Native counters include failed games.",
                            "Per-game model timings sum both seats where both use Mew; they cannot establish per-arm speed.",
                            "Peak RSS is cumulative process high water, not per-game allocation.",
                            "Observed deck and seat summaries are descriptive without additional multiplicity-adjusted inference.",
                            "Baseline002 was trained on its original frozen engine. Cross-engine behavioral parity with this common Mew engine is unqualified, so the package contrast also contains an environment-transfer difference.",
                            "Final adoption requires both the fresh v2 baseline confirmation and qualified v1 matched-random contrast; this single-run readout does not establish that conjunction.",
                            "Original pretraining corpus is absent. No pretraining was reproduced and no additional learning was performed.",
                            "Historical six-deck policy-only results do not establish ranked or full target coverage.",
                            *audit["limits"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--comparison-id", help="Read only this audited contrast, for a stable closed phase during a live run")
    args = parser.parse_args()
    try:
        result = summarize(args.protocol, args.run, args.audit, args.allow_partial, args.comparison_id)
    except Exception as error:
        result = {"schema": "mew-comparison-safe-readout/1", "status": "invalid", "error": {"type": type(error).__name__, "message": str(error)}, "summarizer_sha256": sha(__file__)}
    with args.receipt.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "error": result.get("error"), "contrasts": [{"comparison_id": row["comparison_id"], "complete": row["complete"], "verdict": row["verdict"]} for row in result.get("comparisons", [])]}))
    return 2 if result["status"] == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
