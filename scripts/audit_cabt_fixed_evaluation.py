"""Audit retained fixed CABT evaluation without loading Torch, models or the engine."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


class AuditError(ValueError):
    pass


@dataclass(frozen=True)
class AuditReceipt:
    schema: str
    status: str
    recorded_at_utc: str
    checks: dict
    metrics: dict
    artifact_sha256: dict
    limitations: tuple[str, ...]


def require(condition, message):
    if not condition:
        raise AuditError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def document(path):
    return json.loads(path.read_bytes())


def schedule(seed):
    seed_bytes = hashlib.sha256(f"{seed}:evaluation-assignments".encode()).digest()[:8]
    rng = random.Random(int.from_bytes(seed_bytes, "little"))
    draws = []
    for stratum, (learner_pool, opponent_pool) in enumerate(((range(4), range(4)),
            (range(4), range(4, 6)), (range(4, 6), range(4)), (range(4, 6), range(4, 6)))):
        for index in range(8):
            draws.append((stratum, {"learner_seat": index % 2,
                "learner_deck_index": rng.choice(learner_pool),
                "opponent_deck_index": rng.choice(opponent_pool), "engine_seed": rng.getrandbits(64)}))
    return draws


def score_group(rows, planned):
    scores = [row["score"] for row in rows if row["score"] is not None]
    total = sum(scores)
    return {"planned_games": planned, "started_games": len(rows), "natural_terminal_games": len(scores),
            "wins": scores.count(1), "draws": scores.count(0.5), "losses": scores.count(0),
            "censored_or_failed_games": len(rows) - len(scores), "unstarted_games": planned - len(rows),
            "observed_terminal_mean_score": total / len(scores) if scores else None,
            "missing_score_bounds": [total / planned, (total + planned - len(scores)) / planned]}


def read_progress(path, final):
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        require(not final, "final progress has an unterminated record")
        lines.pop()
    return [json.loads(line) for line in lines], hashlib.sha256(raw).hexdigest()


def audit(directory, checks, hashes, metrics):
    launch = document(directory / "launch.json")
    protocol_path = directory / "protocol.json"
    protocol = document(protocol_path)
    hashes["launch"] = digest(directory / "launch.json")
    hashes["protocol"] = digest(protocol_path)
    require(hashes["protocol"] == launch["protocol_sha256"], "retained protocol differs from launch")
    for name, expected in launch["source_sha256"].items():
        require((ROOT / name).resolve().is_relative_to(ROOT), "source path escapes repository")
        require(digest(ROOT / name) == expected, "current source differs from retained launch")
    checks["launch_source_bindings"] = True
    require(protocol["evaluation"]["games"] == 32 and protocol["evaluation"]["games_per_stratum"] == 8
            and protocol["evaluation"]["native_pairing"] is False
            and protocol["evaluation"]["checkpoint_selection"] is False
            and protocol["evaluation"]["seat_assignment"] == "balanced"
            and protocol["evaluation"]["opponent"] == "retained_initial_random_checkpoint"
            and protocol["evaluation"]["deck_strata"] == ["training/training", "training/heldout", "heldout/training", "heldout/heldout"],
            "unsupported evaluation protocol")
    decks_path = ROOT / protocol["deck_manifest"]
    require(decks_path.resolve().is_relative_to(ROOT), "deck manifest escapes repository")
    require(digest(decks_path) == protocol["deck_manifest_sha256"], "deck manifest hash mismatch")
    decks = document(decks_path)["decks"]
    require(len(decks) == 6 and all(len(row["card_ids"]) == 60 for row in decks), "expected six complete decks")
    deck_hashes = [hashlib.sha256(encoded(row["card_ids"])).hexdigest() for row in decks]
    initial = protocol["model"]["initial_weights_sha256"]
    require(digest(Path(launch["initial_checkpoint"]) / "weights.pt") == initial, "initial weight hash mismatch")
    hashes["initial_weights"] = initial
    require(digest(Path(launch["binary"])) == protocol["engine"]["sha256"], "engine binary hash mismatch")
    expected_identity = {"protocol_sha256": hashes["protocol"],
        "config_sha256": protocol["reference_proposal_sha256"], "engine_sha256": protocol["engine"]["sha256"],
        "decks_sha256": protocol["deck_manifest_sha256"], "projection_sha256": protocol["projection_sha256"],
        "initial_weights_sha256": initial}
    final_available = (directory / "result.json").is_file()
    progress, hashes["progress_snapshot"] = read_progress(directory / "progress.jsonl", final_available)
    games, accepted, evaluation_started = {}, [], False
    for row in progress:
        if row["event"] == "game":
            require(row["episode_id"] not in games, "duplicate progress episode")
            require(row["phase"] in ("training", "evaluation"), "unknown progress game phase")
            if row["phase"] == "evaluation":
                evaluation_started = True
            else:
                require(not evaluation_started, "training game follows evaluation")
                current = accepted[-1]["weights_sha256"] if accepted else initial
                expected_collection = {"learner_weights_sha256": current, "learner_policy_version": len(accepted),
                    "engine_sha256": expected_identity["engine_sha256"], "projection_sha256": expected_identity["projection_sha256"],
                    "protocol_sha256": hashes["protocol"]}
                require(row["identity"] == expected_collection, "training collection identity differs from accepted timeline")
            games[row["episode_id"]] = row
        elif row["event"] == "accepted_update":
            require(not evaluation_started, "accepted update follows evaluation")
            version = len(accepted) + 1
            require(row["counts"]["accepted_updates"] == version, "accepted updates are discontinuous")
            require(Path(row["checkpoint"]) == Path("checkpoints") / f"update-{version:04d}", "unexpected checkpoint location")
            checkpoint = directory / row["checkpoint"]
            receipt_path = directory / f"accepted-update-{version:04d}.json"
            require(document(receipt_path) == row, "accepted receipt differs from progress")
            manifest = document(checkpoint / "manifest.json")
            payload = manifest["payload"]
            require(hashlib.sha256(encoded(payload)).hexdigest() == manifest["sha256"], "checkpoint envelope hash mismatch")
            require(payload["identity"] == expected_identity, "checkpoint identity differs from protocol")
            require(payload["weights_sha256"] == row["weights_sha256"]
                    and payload["counters"]["policy_version"] == version, "checkpoint endpoint mismatch")
            require(set(payload["files"]) == {"weights.pt", "optimizer.pt", "population.json", "samplers.json"},
                    "checkpoint file inventory mismatch")
            for checkpoint_key, campaign_key in (("optimizer_steps", "optimizer_steps"),
                    ("learner_micro_decisions", "trained_decisions"), ("games_started", "games_started"), ("games_finished", "games_finished")):
                require(payload["counters"][checkpoint_key] == row["counts"][campaign_key], "checkpoint counters differ from acceptance")
            for name, expected in payload["files"].items():
                require(Path(name).name == name, "checkpoint file escapes directory")
                require(digest(checkpoint / name) == expected, "retained checkpoint file hash mismatch")
            require(payload["files"]["weights.pt"] == row["weights_sha256"], "checkpoint weights do not match receipt")
            rollout = row["rollout"]
            require((directory / rollout["path"]).resolve().is_relative_to(directory), "rollout path escapes run")
            require(digest(directory / rollout["path"]) == rollout["file_sha256"], "accepted rollout file hash mismatch")
            hashes[f"accepted_update_{version:04d}"] = digest(receipt_path)
            hashes[f"checkpoint_manifest_{version:04d}"] = digest(checkpoint / "manifest.json")
            accepted.append(row)
    checks["accepted_checkpoint_and_rollout_file_hashes"] = True
    metrics["accepted_updates_observed"] = len(accepted)
    metrics["training_games_observed"] = sum(row["phase"] == "training" for row in games.values())
    plan = schedule(protocol["model"]["seed"])
    seeds = {draw["engine_seed"] for _, draw in plan}
    require(len(seeds) == 32, "evaluation seeds are not unique")
    training_seeds = {row["draw"]["engine_seed"] for row in games.values() if row["phase"] == "training"}
    require(not seeds & training_seeds, "evaluation native seeds overlap observed training")
    checks["independently_reconstructed_schedule_seed_separation"] = True
    hashes["planned_schedule"] = hashlib.sha256(encoded(plan)).hexdigest()
    if not final_available:
        metrics["evaluation_games_observed"] = sum(row["phase"] == "evaluation" for row in games.values())
        checks["final_result_present"] = False
        checks["failure_artifact_present"] = (directory / "failure.json").is_file()
        if checks["failure_artifact_present"]:
            hashes["failure"] = digest(directory / "failure.json")
        return "failed_run" if checks["failure_artifact_present"] else "result_unavailable"
    require(accepted, "no accepted checkpoint for evaluation")
    hashes["result"] = digest(directory / "result.json")
    hashes["training_result"] = digest(directory / "training-result.json")
    final = document(directory / "result.json")
    training_final = document(directory / "training-result.json")
    endpoint = accepted[-1]
    training_games = [row for row in games.values() if row["phase"] == "training"]
    for result in (training_final, final):
        counts = result["counts"]
        for key in ("accepted_updates", "optimizer_steps", "trained_decisions"):
            require(counts[key] == endpoint["counts"][key], "final accepted counters differ from endpoint")
        require(counts["games_started"] == counts["games_finished"] == len(training_games), "final training game count mismatch")
        require(counts["raw_decisions"] == sum(row["learner_micro_decisions"] for row in training_games), "final raw decision count mismatch")
        require(counts["admitted_decisions"] == sum(row["learner_micro_decisions"] for row in training_games if row["end"] == "terminal"),
                "final admitted decision count mismatch")
        require(counts["truncated_games"] == sum(row["end"] == "administrative_truncation" for row in training_games), "final truncation count mismatch")
        require(result["last_good_weights_sha256"] == endpoint["weights_sha256"], "final endpoint is not last accepted update")
        require(Path(result["last_good_checkpoint"]).resolve() == (directory / endpoint["checkpoint"]).resolve(),
                "final checkpoint path mismatch")
    require(training_final["status"] == "training_complete" and final["status"] == "complete", "campaign did not complete")
    for key in training_final:
        if key != "status":
            require(final[key] == training_final[key], "final result changed training endpoint metadata")
    endpoint_identity = {"learner_weights_sha256": endpoint["weights_sha256"], "learner_policy_version": len(accepted),
        "engine_sha256": expected_identity["engine_sha256"], "projection_sha256": expected_identity["projection_sha256"],
        "protocol_sha256": hashes["protocol"]}
    hashes["episodes_before"] = digest(directory / "episodes.jsonl.gz")
    rows, seen, active, step_count, last_reward = [], set(), None, 0, 0
    with gzip.open(directory / "episodes.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            require(line.endswith("\n"), "episode artifact has an unterminated record")
            record = json.loads(line)
            if "schema" in record:
                require(active is None and record["schema"] == "86-tcg-agent/private-episode/1", "nested or unknown episode")
                active, step_count, last_reward = record, 0, 0
                assignment = record["assignment"]
                episode_id = assignment["episode_id"]
                require(episode_id not in seen and episode_id in games, "duplicate or uncorrelated episode header")
                seen.add(episode_id)
                game = games[episode_id]
                for key in ("identity", "opponent", "learner_deck_sha256", "opponent_deck_sha256"):
                    require(assignment[key] == game[key], "episode assignment differs from progress")
                require(record["phase"] == game["phase"] and record["draw"] == game["draw"], "episode draw differs from progress")
                draw = record["draw"]
                require(all(type(draw[key]) is int and 0 <= draw[key] < 6
                            for key in ("learner_deck_index", "opponent_deck_index")), "invalid deck index")
                require(assignment["learner_seat"] == draw["learner_seat"], "assignment seat differs from draw")
                require(assignment["learner_deck_sha256"] == deck_hashes[draw["learner_deck_index"]]
                        and assignment["opponent_deck_sha256"] == deck_hashes[draw["opponent_deck_index"]], "deck role hash mismatch")
                if record["phase"] == "evaluation":
                    index = len(rows)
                    require(index < 32 and episode_id == f"evaluation-{index:04d}", "evaluation sequence differs from fixed plan")
                    require(draw == plan[index][1], "evaluation draw differs from independently reconstructed plan")
                    require(assignment["identity"] == endpoint_identity, "evaluation learner endpoint mismatch")
                    require(assignment["opponent"] == {"accepted_update": 0, "weights_sha256": initial}, "evaluation opponent is not initial checkpoint")
            elif "step" in record:
                require(active is not None, "step outside episode")
                step = record["step"]
                assignment = active["assignment"]
                expected_provenance = {"episode_id": assignment["episode_id"], "learner_seat": assignment["learner_seat"],
                    "deck_hash": assignment["learner_deck_sha256"], "policy_version": assignment["identity"]["learner_policy_version"],
                    "opponent_hash": assignment["opponent"]["weights_sha256"]}
                require(step["provenance"] == expected_provenance and step["sequence"] == step_count, "step provenance or sequence mismatch")
                require(last_reward == 0, "nonterminal step has reward")
                last_reward = step["reward"]
                require(type(last_reward) in (int, float) and math.isfinite(last_reward), "nonfinite or invalid reward")
                step_count += 1
            elif "end" in record:
                require(active is not None, "ending outside episode")
                assignment = active["assignment"]
                game = games[assignment["episode_id"]]
                require(record["end"] in ("terminal", "administrative_truncation", "infrastructure_failure"), "unknown ending")
                require(record["end"] == game["end"] and record["reason"] == game["reason"]
                        and step_count == game["learner_micro_decisions"], "episode ending differs from progress")
                score = None
                if record["end"] == "terminal":
                    winner = record["terminal_result"]
                    require(type(winner) is int and winner in (0, 1, 2) and step_count > 0, "invalid terminal result")
                    reward = 0 if winner == 2 else (1 if winner == assignment["learner_seat"] else -1)
                    require(last_reward == reward and game["terminal_reward"] == reward, "terminal reward differs from absolute winner")
                    score = (reward + 1) / 2
                else:
                    require(last_reward == 0 and "terminal_reward" not in game, "censored game has terminal reward")
                if active["phase"] == "evaluation":
                    rows.append({"stratum": plan[len(rows)][0], "seat": assignment["learner_seat"],
                        "end": record["end"], "reason": record["reason"], "score": score})
                active = None
            else:
                raise AuditError("unknown episode record")
    require(active is None and seen == set(games), "episode artifact is incomplete or differs from progress")
    summary = score_group(rows, 32)
    evaluation = final["evaluation"]
    require(evaluation["planned_games"] == 32 and evaluation["native_pairing"] is False, "result evaluation protocol mismatch")
    for key in ("started_games", "natural_terminal_games", "observed_terminal_mean_score", "missing_score_bounds"):
        require(evaluation[key] == summary[key], "reported evaluation aggregate differs from retained episodes")
    require(evaluation["games"] == [{key: row[key] for key in ("stratum", "end", "reason", "score")} for row in rows],
            "reported per-game outcomes differ from retained episodes")
    complete = len(rows) == 32 and summary["natural_terminal_games"] == 32
    require(evaluation["complete_fixed_sample"] is complete, "reported fixed-sample completion differs from episodes")
    metrics["overall"] = summary
    metrics["by_stratum"] = {str(index): score_group([row for row in rows if row["stratum"] == index], 8) for index in range(4)}
    metrics["by_seat"] = {str(seat): score_group([row for row in rows if row["seat"] == seat], 16) for seat in range(2)}
    metrics["by_stratum_and_seat"] = {f"{index}/{seat}": score_group(
        [row for row in rows if row["stratum"] == index and row["seat"] == seat], 4)
        for index in range(4) for seat in range(2)}
    require(digest(directory / "result.json") == hashes["result"]
            and digest(directory / "training-result.json") == hashes["training_result"], "result artifacts changed while auditing")
    require(digest(directory / "episodes.jsonl.gz") == hashes["episodes_before"], "episode artifact changed while auditing")
    require(digest(directory / "progress.jsonl") == hashes["progress_snapshot"], "progress changed while auditing")
    require(all(digest(ROOT / name) == expected for name, expected in launch["source_sha256"].items()), "source changed while auditing")
    checks.update(final_result_present=True, streamed_episode_progress_result_agreement=True,
                  exact_initial_opponent_and_last_accepted_endpoint=True, complete_fixed_sample=complete)
    return "passed" if complete else "incomplete_fixed_sample"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks, hashes, metrics = {}, {"auditor_source": digest(Path(__file__))}, {}
    try:
        status = audit(args.run_directory.resolve(strict=True), checks, hashes, metrics)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        status = "invalid_artifacts"
        checks["failure"] = str(error) if isinstance(error, AuditError) else type(error).__name__
    receipt = AuditReceipt("86-tcg-agent/fixed-evaluation-audit/1", status,
        datetime.now(timezone.utc).isoformat(), checks, metrics, hashes,
        ("Descriptive 32-game sample against the retained initial random checkpoint; no strength or formal reproduction claim.",
         "Schedule independently reconstructs the pinned launcher formula; native RNG execution is evidenced by retained assignment metadata, not replayed here.",
         "Outcomes are independently recomputed from retained terminal metadata and learner rewards; this does not independently adjudicate native rules.",
         "Checkpoint bytes and accepted rollout file hashes are checked without deserializing tensors or proving full campaign resume."))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(asdict(receipt), stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": status, "accepted_updates_observed": metrics.get("accepted_updates_observed"),
                      "complete_fixed_sample": checks.get("complete_fixed_sample", False)}))
    return 0 if status == "passed" else (2 if status == "result_unavailable" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
