"""Qualify a frozen actor-visible heuristic on a pinned native game cohort."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tcg.decision import SUBMIT
from tcg.heuristic import ACTION_CAP, IDENTITY, MAIN_PRIORITY, NEUTRAL_RULES, RULE_TABLE, choose, selection

ENGINE = "a9792857abafb44709bb9062a6d7cb5aa9cd1148c5168ea3214b20a91b62c0f7"
DECKS = "93150420ccbd048c83f8d5c6c0123c27585dec8ea9996080aa737279b64e3f31"
PROJECTION = "fa45c24cbe3f356d21b04e2f240c07fd664e349fcf76c1f7ffd0a42fd1f85d42"
REPLAY = (0, 13, 28, 43, 58, 71)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"schema": "86-tcg-agent/frozen-development-heuristic/1", "identity": IDENTITY,
                "engine_sha256": ENGINE, "projection_sha256": PROJECTION,
                "deck_manifest_sha256": DECKS, "action_cap": ACTION_CAP,
                "main_priority": [list(row) for row in MAIN_PRIORITY], "rules": RULE_TABLE,
                "neutral_rules": sorted(NEUTRAL_RULES), "neutral_rule_wire_context_domain": list(range(49)),
                "input": "ActorDecision_only", "selection_order": "native_option_tuple_order",
                "qualification": {"ordered_deck_pairs": 36, "heuristic_seats_per_pair": [0, 1],
                                  "native_seed_base": 128849019740001,
                                  "uniform_seed_base": 171798692700001,
                                  "replay_indices": list(REPLAY), "max_submissions": 4096,
                                  "max_submissions_per_turn": 512, "game_seconds": 30,
                                  "process_seconds": 300},
                "status": "future_campaign_comparator_not_baseline_002_preregistration"}
    if {key: value for key, value in manifest.items() if key != "source_sha256"} != expected:
        raise ValueError("heuristic manifest does not match the frozen implementation contract")
    files = ("tcg/heuristic.py", "tcg/decision.py", "tcg/cabt.py", "tcg/cabt_observation.py", "tcg/policy.py")
    actual = {name: sha(ROOT / name) for name in files}
    if manifest["source_sha256"] != actual or actual["tcg/cabt_observation.py"] != PROJECTION:
        raise ValueError("heuristic implementation or projection identity mismatch")
    return manifest


def schedule():
    return [{"index": (left * 6 + right) * 2 + seat, "deck_indices": [left, right],
             "heuristic_seat": seat, "native_seed": 128849019740001 + (left * 6 + right) * 2 + seat,
             "uniform_seed": 171798692700001 + (left * 6 + right) * 2 + seat}
            for left in range(6) for right in range(6) for seat in range(2)]


def game(args, spec, decks):
    from tcg.cabt import CabtBattle, EngineError
    from tcg.cabt_observation import CabtObservationAdapter
    started = time.monotonic()
    counts = {name: Counter() for name in ("heuristic_rules", "rule_families", "prompt_types", "wire_contexts",
                                          "type_context", "legal_choice_counts", "actor_microsteps")}
    totals = Counter()
    decisions_digest = hashlib.sha256()
    sampler = random.Random(spec["uniform_seed"])
    trace_path = args.output / ("trace-%03d.jsonl" % spec["index"])
    result = {**spec, "status": "error", "raw_trace": trace_path.name}
    try:
        with trace_path.open("x", encoding="utf-8", newline="\n") as trace, CabtBattle(
                args.binary, ENGINE, seed_api="BattleStartSeeded64") as battle:
            pair = tuple(decks[index]["card_ids"] for index in spec["deck_indices"])
            adapter = CabtObservationAdapter(tuple(str(card) for card in battle.card_ids()))
            adapter.begin_episode(pair)
            frame = battle.start(*pair, engine_seed=spec["native_seed"])
            per_turn = Counter()
            while frame.result == -1:
                if time.monotonic() - started >= 30 or totals["submissions"] >= 4096:
                    raise TimeoutError("game seconds or submission cap")
                decision = adapter.project(frame.wire_observation, frame.actor)
                initial = decision
                turn = next(token.features[0] for token in decision.tokens if token.role == "state.turn")
                per_turn[turn] += 1
                if per_turn[turn] > 512:
                    raise TimeoutError("per-turn submission cap")
                heuristic = frame.actor == spec["heuristic_seat"]
                actor = "heuristic" if heuristic else "uniform_legal_diagnostic"
                prompt_type = next(token.features[0] for token in decision.tokens if token.role == "prompt.type")
                context = next(token.features[0] for token in decision.tokens if token.role == "prompt.context")
                if heuristic:
                    totals["heuristic_frames"] += 1
                    counts["prompt_types"][str(prompt_type)] += 1
                    counts["wire_contexts"][str(context)] += 1
                    counts["type_context"][f"{prompt_type}/{context}"] += 1
                    totals["ordered_multi_frames"] += int(decision.prompt.ordered and decision.prompt.maximum > 1)
                steps = []
                while True:
                    legal = decision.choices()
                    if heuristic:
                        choice, rule = selection(decision)
                        if choice != choose(decision):
                            raise ValueError("rule accounting and public choose disagree")
                        counts["heuristic_rules"][rule] += 1
                        family = "forced" if rule == "forced" else "neutral" if rule in NEUTRAL_RULES else "semantic"
                        counts["rule_families"][family] += 1
                        counts["legal_choice_counts"][str(len(legal))] += 1
                        totals["forced_microsteps" if len(legal) == 1 else "genuine_microsteps"] += 1
                        totals["cap_activation_microsteps"] += int(rule.startswith("main_cap_"))
                    else:
                        choice, rule = sampler.choice(legal), "uniform_legal_diagnostic"
                    if choice not in legal:
                        totals["illegal_microchoices"] += 1
                        raise ValueError("actor produced an illegal microchoice")
                    counts["actor_microsteps"][actor] += 1
                    steps.append({"choice": choice, "rule": rule, "legal_count": len(legal)})
                    if choice == SUBMIT:
                        response = decision.response()
                        break
                    decision = decision.pick(choice)
                if any(type(index) is not str for index in response):
                    raise ValueError("this cohort has no qualified repeated-unit allocation")
                indices = [int(index) for index in response]
                record = {"actor": frame.actor, "controller": actor, "wire": frame.wire_observation,
                          "actor_decision_sha256": digest(asdict(initial)), "microsteps": steps,
                          "response": indices}
                encoded = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                trace.write(encoded)
                decisions_digest.update(encoded.encode())
                try:
                    frame = battle.submit(indices)
                except EngineError:
                    totals["native_rejections"] += 1
                    raise
                totals["submissions"] += 1
            terminal = {"terminal_actor": frame.actor, "terminal_wire": frame.wire_observation,
                        "result": frame.result}
            encoded = json.dumps(terminal, sort_keys=True, separators=(",", ":")) + "\n"
            trace.write(encoded)
            decisions_digest.update(encoded.encode())
            result.update(status="complete", result=frame.result,
                          heuristic_score=0.5 if frame.result == 2 else int(frame.result == spec["heuristic_seat"]))
    except Exception as error:
        result.update(status="censored" if isinstance(error, TimeoutError) else "error",
                      error_type=type(error).__name__, error=str(error))
    result.update(counts={key: dict(value) for key, value in counts.items()}, totals=dict(totals),
                  actor_frame_response_outcome_sha256=decisions_digest.hexdigest(),
                  elapsed_seconds=time.monotonic() - started)
    if trace_path.exists():
        result["raw_trace_sha256"] = sha(trace_path)
    return result


def worker(args):
    import torch
    torch.set_num_threads(1)
    manifest = validate_manifest(args.manifest)
    if sha(args.binary) != ENGINE or sha(ROOT / "experiments/cabt-v1-candidate-decks.json") != DECKS:
        raise ValueError("native artifact or exact deck manifest mismatch")
    decks = json.loads((ROOT / "experiments/cabt-v1-candidate-decks.json").read_text())["decks"]
    results = []
    selected = schedule() if args.phase == "cohort" else [row for row in schedule() if row["index"] in REPLAY]
    for spec in selected:
        write(args.output / ("started-%03d.json" % spec["index"]),
              {"index": spec["index"], "started_monotonic": time.monotonic()})
        result = game(args, spec, decks)
        results.append(result)
        write(args.output / ("game-%03d.json" % spec["index"]), result)
    if validate_manifest(args.manifest) != manifest or sha(args.binary) != ENGINE:
        raise ValueError("inputs changed during native cohort")
    write(args.output / "results.json", results)


def run_phase(args, phase, deadline):
    output = args.output / phase
    output.mkdir()
    command = [sys.executable, str(Path(__file__).resolve()), "--manifest", str(args.manifest),
               "--binary", str(args.binary), "--output", str(output), "--phase", phase, "--worker"]
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, cwd=ROOT)
        write(output / "process.json", {"pid": process.pid, "command": command})
        watchdog = None
        while process.poll() is None:
            starts = sorted(output.glob("started-*.json"))
            if time.monotonic() >= deadline:
                watchdog = "300 second whole qualification cap"
            elif starts:
                state = json.loads(starts[-1].read_text())
                finished = output / ("game-%03d.json" % state["index"])
                if not finished.exists() and time.monotonic() - state["started_monotonic"] > 30:
                    watchdog = "30 second active game cap"
            if watchdog:
                process.kill()
                process.wait()
                break
            time.sleep(0.1)
    completed = sorted(output.glob("game-*.json"))
    games = [json.loads(path.read_text()) for path in completed]
    planned = [row["index"] for row in schedule() if phase == "cohort" or row["index"] in REPLAY]
    started_indices = [json.loads(path.read_text())["index"] for path in sorted(output.glob("started-*.json"))]
    finished_indices = [row["index"] for row in games]
    return {"exit_code": process.returncode, "watchdog": watchdog,
            "planned_indices": planned, "started_indices": started_indices,
            "finished_indices": finished_indices,
            "unfinished_started_indices": sorted(set(started_indices) - set(finished_indices)),
            "not_started_indices": sorted(set(planned) - set(started_indices)), "games": games}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/development-heuristic-v1.json")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=("cohort", "replay"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return 0
    if not args.execute:
        parser.error("native execution requires --execute")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = validate_manifest(args.manifest)
    if sha(args.binary) != ENGINE:
        raise ValueError("native artifact mismatch")
    bindings = {"manifest_sha256": sha(args.manifest), "qualifier_sha256": sha(__file__),
                "engine_sha256": sha(args.binary), "source_sha256": manifest["source_sha256"],
                "deck_manifest_sha256": sha(ROOT / "experiments/cabt-v1-candidate-decks.json")}
    write(args.output / "launch.json", {**bindings, "schedule": schedule(),
          "runtime": sys.version, "scope": "native heuristic engineering qualification; no playing strength claim",
          "host_contention": "concurrent baseline 002 may be running; no comparative performance claim",
          "device": "cpu", "torch_threads": 1, "optimizer_updates": 0})
    started = time.monotonic()
    cohort = run_phase(args, "cohort", started + 300)
    replay = run_phase(args, "replay", started + 300) if time.monotonic() < started + 300 else None
    totals, counts = Counter(), {}
    for row in cohort["games"]:
        totals.update(row["totals"])
        for name, values in row["counts"].items():
            counts.setdefault(name, Counter()).update(values)
    originals = {row["index"]: row for row in cohort["games"]}
    replay_matches = [{"index": row["index"], "equal": row["status"] == "complete"
                       and row["actor_frame_response_outcome_sha256"] == originals.get(row["index"], {}).get(
                           "actor_frame_response_outcome_sha256")} for row in replay["games"]] if replay else []
    unchanged = (sha(args.manifest) == bindings["manifest_sha256"]
                 and sha(__file__) == bindings["qualifier_sha256"] and sha(args.binary) == ENGINE
                 and validate_manifest(args.manifest) == manifest
                 and sha(ROOT / "experiments/cabt-v1-candidate-decks.json") == DECKS)
    passed = (len(cohort["games"]) == 72 and all(row["status"] == "complete" for row in cohort["games"])
              and len(replay_matches) == len(REPLAY) and all(row["equal"] for row in replay_matches)
              and cohort["exit_code"] == 0 and replay["exit_code"] == 0 and unchanged
              and not totals["illegal_microchoices"] and not totals["native_rejections"])
    receipt = {"schema": "86-tcg-agent/frozen-heuristic-qualification/1", "passed": passed,
               "bindings": bindings, "inputs_unchanged": unchanged, "cohort": cohort, "replay": replay,
               "replay_comparisons": replay_matches, "totals": dict(totals),
               "counts": {name: dict(values) for name, values in counts.items()},
               "complete_games": sum(row["status"] == "complete" for row in cohort["games"]),
               "censored_games": sum(row["status"] == "censored" for row in cohort["games"]),
               "unfinished_started_games": len(cohort["unfinished_started_indices"]),
               "not_started_games": len(cohort["not_started_indices"]),
               "planned_games": 72, "started_games": len(cohort["started_indices"]),
               "error_games": sum(row["status"] == "error" for row in cohort["games"]),
               "elapsed_seconds": time.monotonic() - started, "optimizer_updates": 0,
               "limitations": ["bounded six candidate-deck coverage, not exhaustive context/effect coverage",
                               "uniform comparator is a legal-random diagnostic, not a formal heuristic",
                               "game results are not a strength estimate or candidate-selection criterion",
                               "formal eae artifact and future MatchSession are not qualified here"]}
    write(args.output / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("passed", "complete_games", "censored_games", "error_games", "elapsed_seconds")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
