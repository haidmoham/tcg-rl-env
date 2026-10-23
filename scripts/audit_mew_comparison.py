"""Read-only audit of recorded comparison evidence, without importing either player."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_schedule(protocol):
    assignments = {}
    seeds = set()
    domain = protocol["seed_assignment"]["domain"]
    version = protocol_version(protocol)
    require(domain == ("mew-comparison-primary-v1" if version == 1 else "mew-baseline-confirmation-v2"), "unexpected seed domain")
    require(len(protocol["comparisons"]) == (2 if version == 1 else 1), "unexpected contrast count")
    if version == 2:
        comparison = protocol["comparisons"][0]
        require(comparison["candidate"] == "mew_pretrained" and comparison["opponent"] == "baseline002_last_verified_endpoint", "confirmation must use the original frozen player contrast")
    for comparison in protocol["comparisons"]:
        identity = comparison["comparison_id"]
        expected = []
        for left in range(6):
            for right in range(6):
                seed = int.from_bytes(hashlib.sha256(canonical([domain, identity, left, right])).digest()[:4], "big")
                require(seed != 0 and seed not in seeds, "zero or reused pair seed")
                seeds.add(seed)
                pair = f"{identity}/{left}-{right}"
                for leg in (0, 1):
                    expected.append({"effective_seed32": seed, "game_id": f"{pair}/leg-{leg}",
                                     "leg": leg, "pair_id": pair,
                                     "seat_arms": [comparison["candidate"], comparison["opponent"]][::1 if leg == 0 else -1],
                                     "seat_deck_indices": [left, right][::1 if leg == 0 else -1]})
        require(comparison["games"] == expected, f"derived schedule mismatch in {identity}")
        for row in expected:
            require(row["game_id"] not in assignments, "duplicate game identity")
            assignments[row["game_id"]] = row
    if version == 2:
        old_seeds = {
            int.from_bytes(hashlib.sha256(canonical(["mew-comparison-primary-v1", identity, left, right])).digest()[:4], "big")
            for identity in ("mew_pretrained_vs_baseline002_last_verified_endpoint", "mew_pretrained_vs_mew_matched_random_seed860201")
            for left in range(6) for right in range(6)
        }
        require(seeds.isdisjoint(old_seeds), "confirmation seed overlaps the first-attempt domain")
    return assignments


def protocol_version(protocol):
    schema = protocol.get("schema")
    if schema == "86-tcg-agent/mew-comparison/1":
        return 1
    require(schema == "86-tcg-agent/mew-baseline-confirmation/2", "unsupported protocol schema")
    require(protocol.get("purpose_kind") == "corrected_baseline_confirmation", "missing confirmation purpose kind")
    return 2


def audit_external_decision(decision, response, rec):
    require(set(decision) == {"tokens", "prompt", "selected"}, "unexpected ActorDecision fields")
    prompt = decision["prompt"]
    require(set(prompt) == {"options", "minimum", "maximum", "ordered", "allow_repeats", "allocation_total", "semantics"}, "unexpected LegalPrompt fields")
    require(prompt["semantics"] == "independent" and prompt["allocation_total"] is None, "unsupported physical response semantics")
    require(prompt["ordered"] is True and type(prompt["allow_repeats"]) is bool, "physical response order or repeat flag mismatch")
    require(type(prompt["minimum"]) is int and type(prompt["maximum"]) is int, "invalid prompt cardinality type")
    require(0 <= prompt["minimum"] <= prompt["maximum"], "invalid prompt cardinality")
    require((prompt["minimum"], prompt["maximum"]) == (rec["selectMin"], rec["selectMax"]), "actor and native cardinality mismatch")
    options = prompt["options"]
    identifiers = [option["option_id"] for option in options]
    require(identifiers == [str(index) for index in range(rec["optionCount"])], "physical option identity mismatch")
    for option in options:
        require(set(option) == {"option_id", "token", "state_indices"}, "unexpected Option fields")
        require(all(type(index) is int and 0 <= index < len(decision["tokens"]) for index in option["state_indices"]), "option points outside actor-visible tokens")
    selected = list(decision["selected"])
    require(len(selected) == rec["prefixLen"], "actor prefix differs from the native record")
    require(all(type(value) is str and value in identifiers for value in selected), "unknown initial selected option")
    require(len(selected) <= prompt["maximum"], "initial prefix exceeds maximum")
    require(prompt["allow_repeats"] or len(set(selected)) == len(selected), "repeated initial selection")
    require(prompt["allow_repeats"] or prompt["maximum"] <= len(options), "impossible distinct maximum")
    physical = [str(index) for index in response]
    require(physical[:len(selected)] == selected, "response changes the selected prefix")
    calls = forced = 0
    for option_id in physical[len(selected):]:
        available = [identity for identity in identifiers if len(selected) < prompt["maximum"] and (prompt["allow_repeats"] or identity not in selected)]
        choices = available + (["__submit__"] if len(selected) >= prompt["minimum"] else [])
        require(option_id in available, "illegal external PICK transition")
        calls += 1
        forced += len(choices) == 1
        selected.append(option_id)
    available = [identity for identity in identifiers if len(selected) < prompt["maximum"] and (prompt["allow_repeats"] or identity not in selected)]
    require(len(selected) >= prompt["minimum"], "illegal external SUBMIT transition")
    calls += 1
    forced += len(available) + 1 == 1
    return calls, forced


def audit_trace(path, receipt, version=1):
    require(sha(path) == receipt["trace_sha256"], f"trace bytes mismatch {path.name}")
    counts = Counter()
    model_calls = Counter()
    terminal = None
    previous_index = previous_turn = -1
    for line_number, line in enumerate(path.read_bytes().splitlines(), 1):
        row = json.loads(line)
        require(line == canonical(row), f"noncanonical trace line {line_number}")
        if "terminal_record" in row:
            require(terminal is None, "duplicate terminal record")
            terminal = row["terminal_record"]
            require(terminal["terminal"] and terminal["winner"] in (0, 1, 2), "invalid terminal record")
            continue
        require(terminal is None, "action after terminal")
        rec = row["record"]
        require(not rec["terminal"], "action marked terminal")
        require(rec["decisionIndex"] >= previous_index and rec["turn"] >= previous_turn, "nonmonotonic native record")
        previous_index, previous_turn = rec["decisionIndex"], rec["turn"]
        require(row["arm"] == receipt["assignment"]["seat_arms"][rec["selectPlayer"]], "wrong actor arm")
        require(re.fullmatch("[0-9a-f]{64}", row["semantic_input_sha256"]) is not None, "invalid semantic digest")
        if "selected_atomic_column" in row:
            mask = row["legal_mask"]
            choice = row["selected_atomic_column"]
            require(rec["batchRow"] >= 0 and row["arm"] != "baseline002_last_verified_endpoint", "wrong internal actor")
            require(all(type(value) is bool for value in mask), "invalid mask type")
            require(type(choice) is int and 0 <= choice < len(mask) and mask[choice], "illegal recorded atomic choice")
            require(all(not value or index < rec["visibleCount"] for index, value in enumerate(mask)), "legal padding column")
            require(choice <= rec["optionCount"] and (choice < rec["optionCount"] or rec["canSubmit"]), "invalid PICK or SUBMIT")
            counts["atomic_choices"] += 1
            counts["forced_encoded"] += sum(mask) == 1
            counts["max_legal"] = max(counts["max_legal"], sum(mask))
            model_calls[row["arm"]] += 1
        else:
            response = row["ordered_physical_response"]
            require(rec["batchRow"] < 0 and row["arm"] == "baseline002_last_verified_endpoint", "wrong external actor")
            require(rec["prefixLen"] == 0, "external response starts inside a prefix")
            require(rec["selectMin"] <= len(response) <= rec["selectMax"], "external response cardinality")
            require(all(type(index) is int and 0 <= index < rec["optionCount"] for index in response), "external option out of range")
            if version == 1:
                require(len(set(response)) == len(response), "duplicate physical option")
            else:
                require("initial_actor_decision" in row, "confirmation trace lacks the actor decision")
                calls, forced = audit_external_decision(row["initial_actor_decision"], response, rec)
                counts["external_legal_microsteps"] += calls
                counts["external_forced_microsteps"] += forced
            counts["external_actions"] += 1
    require(counts["atomic_choices"] == receipt["mew_atomic_choices"], "atomic count mismatch")
    require(dict(model_calls) == receipt["mew_model_calls"], "model call count mismatch")
    require(counts["forced_encoded"] == receipt["forced_mew_encoded_choices"], "forced count mismatch")
    require(counts["max_legal"] == receipt["max_legal_columns"], "legal column count mismatch")
    require(counts["external_actions"] == receipt["external_actions"], "external count mismatch")
    require(receipt["baseline_forward_calls"] == receipt["baseline_microsteps"], "baseline forward accounting mismatch")
    require(0 <= receipt["baseline_forced_microsteps"] <= receipt["baseline_microsteps"], "invalid forced microsteps")
    if receipt["status"] == "complete":
        require(terminal == receipt["last_record"] and terminal["winner"] == receipt["winner"], "terminal receipt mismatch")
        require(receipt["native_stats"]["gamesFinished"] == 1, "native completion counter mismatch")
        if version == 2:
            require(counts["external_legal_microsteps"] == receipt["baseline_microsteps"], "external microstep accounting mismatch")
            require(counts["external_forced_microsteps"] == receipt["baseline_forced_microsteps"], "external forced-choice accounting mismatch")
    else:
        require(terminal is None and receipt["winner"] is None, "unresolved game has winner")
        require("error" in receipt, "failed game lacks error")
    return dict(counts)


def audit(protocol_path, bindings_path, output):
    protocol, bindings = load(protocol_path), load(bindings_path)
    version = protocol_version(protocol)
    assignments = audit_schedule(protocol)
    require(sha(protocol_path) == bindings["protocol_sha256"], "protocol binding mismatch")
    require(hashlib.sha256(canonical(bindings["artifacts"])).hexdigest() == protocol["artifact_manifest_sha256"], "artifact manifest mismatch")
    for path, expected in bindings["artifacts"].items():
        require(sha(path) == expected, f"current artifact mismatch {path}")
    launch, players = load(output / "launch.json"), load(output / "players.json")
    require(launch["mode"] == "primary" and protocol["execution_authorized"] is True, "not authorized primary evidence")
    require(launch["protocol_sha256"] == sha(protocol_path) and launch["bindings_sha256"] == sha(bindings_path), "launch identity mismatch")
    require(launch["runner_sha256"] == protocol["runner_sha256"], "runner identity mismatch")
    require(launch["source_and_artifacts"] == bindings["artifacts"] and launch["optimizer_updates"] == 0, "launch artifact or optimizer mismatch")
    require(players["baseline_transport"]["optimizer_restored"] is False, "optimizer unexpectedly restored")
    require(players["baseline_transport"]["manifest_sha256"] == protocol["players"]["baseline002_last_verified_endpoint"]["checkpoint_manifest_sha256"], "baseline endpoint mismatch")
    require(len(players["state_sha256"]) == 3, "missing player state identities")
    require(sha(bindings["deck_manifest"]) == protocol["deck_manifest_sha256"], "deck manifest mismatch")
    decks = load(bindings["deck_manifest"])["decks"]
    for deck, expected in zip(decks, protocol["decks"]):
        require(len(deck["card_ids"]) == 60 and hashlib.sha256(canonical(deck["card_ids"])).hexdigest() == expected["ordered_card_ids_sha256"], "deck order mismatch")
    summaries, totals, failures = [], Counter(), Counter()
    expected_files, expected_traces = set(), set()
    for number, comparison in enumerate(protocol["comparisons"]):
        scores, cluster_scores, receipts = [], {}, []
        statuses = Counter()
        for index, assignment in enumerate(comparison["games"]):
            stem = f"{number:02d}-{index:03d}"
            receipt_path = output / f"{stem}.json"
            expected_files.add(receipt_path.name)
            if not receipt_path.exists():
                scores.append(None)
                statuses["missing_receipt"] += 1
                continue
            receipt = load(receipt_path)
            require(receipt["assignment"] == assignment, f"assignment mismatch {stem}")
            require(receipt["players_unchanged"] is True, f"changed players {stem}")
            require(receipt["status"] in ("complete", "failed", "unrun_phase_cap"), f"unknown receipt status {stem}")
            statuses[receipt["status"]] += 1
            receipts.append(receipt)
            if receipt["status"] != "unrun_phase_cap":
                expected_traces.add(f"{stem}.trace.jsonl")
                counts = audit_trace(output / f"{stem}.trace.jsonl", receipt, version)
                totals.update({key: value for key, value in counts.items() if key != "max_legal"})
                costs = [receipt["total_wall_seconds"], receipt["baseline_model_forward_seconds"], *receipt["timing_seconds"].values()]
                require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in costs), "invalid game costs")
                require(sum(receipt["timing_seconds"].values()) <= receipt["total_wall_seconds"] + 1e-6, "component times exceed game wall")
            else:
                require(receipt["winner"] is None, "unrun assignment has winner")
            if receipt["status"] == "complete":
                winner = receipt["winner"]
                score = .5 if winner == 2 else float(assignment["seat_arms"][winner] == comparison["candidate"])
                cluster_scores.setdefault(assignment["pair_id"], []).append(score)
            else:
                score = None
                error = receipt.get("error", {})
                failures[f"{receipt['status']}: {error.get('type', '')}: {error.get('message', '')}"] += 1
            scores.append(score)
        low = sum(value if value is not None else 0 for value in scores) / 72
        high = sum(value if value is not None else 1 for value in scores) / 72
        radius = math.sqrt(math.log(2 / .025) / (2 * 36))
        complete = sum(value is not None for value in scores)
        summary = {"comparison_id": comparison["comparison_id"], "planned_games": 72,
                   "complete_games": complete, "failed_or_unrun": 72-complete,
                   "wins": scores.count(1), "draws": scores.count(.5), "losses": scores.count(0),
                   "score_bounds": [low, high], "simultaneous_hoeffding_interval": [max(0, low-radius), min(1, high+radius)],
                   "independent_seat_pair_clusters": 36, "per_comparison_alpha": .025,
                   "verdict": "candidate_superior" if complete == 72 and low-radius > .5 else "inconclusive",
                   "statuses": dict(statuses), "complete_cluster_means": {key: sum(value)/2 for key, value in cluster_scores.items() if len(value) == 2},
                   "game_wall_seconds_including_failures": sum(row.get("total_wall_seconds", 0) for row in receipts),
                   "complete_game_wall_seconds": [row["total_wall_seconds"] for row in receipts if row["status"] == "complete"],
                   "baseline_model_forward_seconds": sum(row.get("baseline_model_forward_seconds", 0) for row in receipts),
                   "mew_model_forward_seconds": sum(row.get("timing_seconds", {}).get("forward", 0) for row in receipts)}
        summaries.append(summary)
    actual_receipts = {path.name for path in output.glob("[0-9][0-9]-[0-9][0-9][0-9].json")}
    require(not actual_receipts-expected_files, "unexpected game receipt")
    result_path = output / "result.json"
    if result_path.exists():
        require(actual_receipts == expected_files, "final result lacks scheduled receipts")
        require({path.name for path in output.glob("*.trace.jsonl")} == expected_traces, "final trace set differs from executed assignments")
        result = load(result_path)
        require(result["mode"] == "primary" and result["optimizer_updates"] == 0 and result["players_and_artifacts_unchanged"] is True, "final identity attestation mismatch")
        require(len(result["comparisons"]) == len(protocol["comparisons"]), "result contrast count mismatch")
        for number, (observed, expected) in enumerate(zip(result["comparisons"], summaries)):
            require(observed["comparison_id"] == expected["comparison_id"], "result contrast mismatch")
            require(observed["receipts"] == [f"{number:02d}-{index:03d}.json" for index in range(72)], "result receipt schedule mismatch")
            require(math.isfinite(observed["phase_wall_seconds"]) and observed["phase_wall_seconds"] >= expected["game_wall_seconds_including_failures"], "invalid phase wall accounting")
            for key in ("planned_games", "complete_games", "failed_or_unrun", "wins", "draws", "losses", "score_bounds", "simultaneous_hoeffding_interval", "independent_seat_pair_clusters", "per_comparison_alpha", "verdict", "complete_game_wall_seconds"):
                require(observed["result"][key] == expected[key], f"reported analysis mismatch {key}")
        require(result["all_scheduled_games_complete"] == all(row["complete_games"] == 72 for row in summaries), "completion claim mismatch")
    return {"schema": "mew-comparison-independent-audit/1", "status": "pass" if result_path.exists() else "incomplete",
            "protocol_sha256": sha(protocol_path), "bindings_sha256": sha(bindings_path), "auditor_sha256": sha(__file__),
            "artifact_count_verified": len(bindings["artifacts"]), "derived_assignments": len(assignments),
            "comparisons": summaries, "trace_counts": dict(totals), "failures": dict(failures),
            "final_result_present": result_path.exists(), "final_result_sha256": sha(result_path) if result_path.exists() else None,
            "protocol_version": version,
            "limits": ["External response order, cardinality, unique physical indices, and native option bounds are checked. External microstep legality and greedy equivalence cannot be reconstructed without omitted projected decisions." if version == 1 else "Each recorded external response is checked against the recorded initial ActorDecision, with exact option identity, selected prefix, repeat policy, legal PICK transitions, and final legal SUBMIT. Greedy equivalence and the projection of omitted source wire are not independently reconstructed.",
                       "Recorded semantic input digests are shape-checked, not recomputed because inputs are omitted.",
                       "Zero optimizer updates and unchanged in-memory tensors use runner attestations plus independently rehashed file identities. No independent process instrumentation or tensor reload is performed.",
                       "This audit never replays a game and does not establish engine rule parity, full RNG identity, or ranked coverage."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--bindings", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--schedule-only", action="store_true", help="Validate only the frozen protocol schedule; never claims game evidence")
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.schedule_only:
            protocol = load(args.protocol)
            assignments = audit_schedule(protocol)
            receipt = {"schema": "mew-comparison-independent-audit/1", "status": "pass", "scope": "schedule_only_no_game_evidence",
                       "protocol_sha256": sha(args.protocol), "auditor_sha256": sha(__file__), "protocol_version": protocol_version(protocol),
                       "derived_assignments": len(assignments), "unique_pair_seeds": len({row["effective_seed32"] for row in assignments.values()}),
                       "domain": protocol["seed_assignment"]["domain"], "execution_authorized": protocol["execution_authorized"]}
        else:
            require(args.bindings is not None and args.run is not None, "bindings and run paths are required for artifact audit")
            receipt = audit(args.protocol, args.bindings, args.run)
    except Exception as error:
        receipt = {"schema": "mew-comparison-independent-audit/1", "status": "invalid", "error": {"type": type(error).__name__, "message": str(error)}, "auditor_sha256": sha(__file__)}
    with args.receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: receipt[key] for key in ("status", "error", "trace_counts", "failures") if key in receipt}, sort_keys=True))
    return 0 if receipt["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
