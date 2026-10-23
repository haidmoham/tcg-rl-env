"""Reproduce signed Energy-selection remainders using seeded native games."""

import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

from qualify_cabt_hp import ROOT, choose_response, file_hash, project_negative, write_record


ENGINE_SHA256 = "a9792857abafb44709bb9062a6d7cb5aa9cd1148c5168ea3214b20a91b62c0f7"
DECKS_SHA256 = "93150420ccbd048c83f8d5c6c0123c27585dec8ea9996080aa737279b64e3f31"
SAMPLER_SEED = 860431


def energy_response(wire, sampler):
    prompt = wire["select"]
    if prompt["type"] == 4:
        if prompt["minCount"] not in (0, 1) or prompt["maxCount"] != 1:
            raise ValueError("unsupported Energy cardinality")
        counts = [option["count"] for option in prompt["option"]]
        if any(type(count) is not int or count <= 0 for count in counts):
            raise ValueError("invalid provided Energy count")
        return [max(range(len(counts)), key=counts.__getitem__)]
    if prompt["type"] == 0:
        options = prompt["option"]
        active_attach = [index for index, option in enumerate(options)
                         if option["type"] == 8 and option["inPlayArea"] == 4]
        retreat = [index for index, option in enumerate(options) if option["type"] == 12]
        if active_attach:
            return [sampler.choice(active_attach)]
        if retreat:
            return [sampler.choice(retreat)]
    return choose_response(wire, sampler)


def run_probe(args):
    sys.path.insert(0, str(ROOT))
    from tcg.cabt import CabtBattle

    if file_hash(args.binary) != ENGINE_SHA256 or file_hash(args.decks) != DECKS_SHA256:
        raise ValueError("pinned engine or six-deck artifact mismatch")
    decks = json.loads(args.decks.read_text(encoding="utf-8"))["decks"]
    if len(decks) != 6 or any(len(deck["card_ids"]) != 60 for deck in decks):
        raise ValueError("expected six exact sixty-card decks")
    record = {
        "schema": "cabt-signed-energy-diagnostic/1",
        "scope": "DIRECTED_RESPONSE_DIAGNOSTIC_NOT_POLICY_PLAY",
        "status": "not_observed", "engine_sha256": ENGINE_SHA256,
        "deck_manifest_sha256": DECKS_SHA256,
        "native_seed_api": "BattleStartSeeded64", "action_sampler_seed": SAMPLER_SEED,
        "optimizer_updates": 0, "device": "cpu", "policy_forward_calls": 0,
        "full_prompt_coverage_verified": False, "ordinary_policy_play_verified": False,
        "implementation_sha256": {
            path.relative_to(ROOT).as_posix(): file_hash(path)
            for path in (*sorted((ROOT / "tcg").glob("*.py")), Path(__file__).resolve(),
                         ROOT / "scripts" / "qualify_cabt_hp.py")},
        "checkpoint_weights_sha256": file_hash(args.untrained_policy / "weights.pt"),
        "checkpoint_manifest_sha256": file_hash(args.untrained_policy / "manifest.json"),
        "games_requested": args.games, "max_seconds": args.max_seconds,
        "max_submissions_per_game": args.max_submissions, "games": [], "errors": [],
    }
    started = time.monotonic()
    sampler = random.Random(SAMPLER_SEED)
    for game_index in range(args.games):
        if time.monotonic() - started >= args.max_seconds:
            record["status"] = "incomplete"
            break
        # Alternate the failing pair's seats to encourage both players' Energy effects.
        pair = (2, 4)
        if game_index % 2:
            pair = pair[::-1]
        seed = sampler.getrandbits(64)
        game = {"game_index": game_index, "deck_indices": list(pair), "native_seed": seed,
                "submissions": 0, "action_prefix": []}
        record["games"].append(game)
        try:
            with CabtBattle(args.binary, ENGINE_SHA256, seed_api="BattleStartSeeded64") as battle:
                exact_decks = tuple(decks[index]["card_ids"] for index in pair)
                frame = battle.start(*exact_decks, engine_seed=seed)
                while frame.result < 0:
                    prompt = frame.wire_observation["select"]
                    remain = prompt["remainEnergyCost"]
                    if type(remain) is not int:
                        raise ValueError("Energy remainder must be a signed integer")
                    if remain < 0:
                        record["observed"] = {
                            "game_index": game_index, "native_seed": seed,
                            "submissions": game["submissions"], "actor": frame.actor,
                            "prompt_type": prompt["type"], "prompt_context": prompt["context"],
                            "min_count": prompt["minCount"], "max_count": prompt["maxCount"],
                            "remain_energy_cost": remain, "options": prompt["option"],
                            "wire_observation": frame.wire_observation,
                        }
                        record["status"] = "reproduced"
                        if args.check_projection:
                            try:
                                record["projection"] = project_negative(
                                    frame, battle, exact_decks, args.untrained_policy)
                                from tcg.cabt_observation import CabtObservationAdapter
                                adapter = CabtObservationAdapter(tuple(str(card) for card in battle.card_ids()))
                                adapter.begin_episode(exact_decks)
                                decision = adapter.project(frame.wire_observation, frame.actor)
                                token = next(token for token in decision.tokens if token.role == "prompt.remainEnergyCost")
                                if token.features != (remain,):
                                    raise ValueError("projection changed the signed native Energy remainder")
                                record["projection"]["signed_energy_features"] = token.features
                                record["policy_forward_calls"] = 1
                            except (ValueError, RuntimeError, KeyError, TypeError) as error:
                                record["projection"] = {"projection_passed": False,
                                                        "error": str(error), "error_type": type(error).__name__}
                        game["status"] = "stopped_at_signed_energy_remainder"
                        break
                    if (time.monotonic() - started >= args.max_seconds
                            or game["submissions"] >= args.max_submissions):
                        game["status"] = "censored"
                        record["status"] = "incomplete"
                        break
                    response = energy_response(frame.wire_observation, sampler)
                    game["action_prefix"].append(response)
                    frame = battle.submit(response)
                    game["submissions"] += 1
                else:
                    game.update(status="natural_terminal", result=frame.result)
        except (RuntimeError, ValueError, KeyError, IndexError, TypeError, OSError) as error:
            record["errors"].append({"game_index": game_index, "type": type(error).__name__, "message": str(error)})
            game["status"] = "failed"
            record["status"] = "incomplete"
        if record["status"] != "not_observed":
            break
    if record["status"] == "reproduced":
        observed = record["observed"]
        game = record["games"][-1]
        exact_decks = tuple(decks[index]["card_ids"] for index in game["deck_indices"])
        record["replays"] = []
        for response in ([], [0]):
            with CabtBattle(args.binary, ENGINE_SHA256, seed_api="BattleStartSeeded64") as battle:
                frame = battle.start(*exact_decks, engine_seed=game["native_seed"])
                for previous in game["action_prefix"]:
                    frame = battle.submit(previous)
                if frame.wire_observation != observed["wire_observation"]:
                    raise ValueError("seed and recorded action prefix failed to reproduce exact wire")
                if response and len(frame.wire_observation["select"]["option"]) < 1:
                    raise ValueError("missing optional Energy continuation")
                following = battle.submit(response)
                prompt = following.wire_observation.get("select")
                record["replays"].append({
                    "exact_actor_wire_match": True, "response": response,
                    "following_result": following.result,
                    "following_prompt_type": prompt.get("type") if prompt else None,
                    "following_prompt_context": prompt.get("context") if prompt else None,
                    "following_remain_energy_cost": prompt.get("remainEnergyCost") if prompt else None,
                })
    record["wall_seconds"] = time.monotonic() - started
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "decks", "untrained-policy", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--max-submissions", type=int, default=4096)
    parser.add_argument("--check-projection", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not 1 <= args.games <= 1024 or not math.isfinite(args.max_seconds)
            or not 0 < args.max_seconds <= 30 or not 1 <= args.max_submissions <= 4096):
        parser.error("requires 1..1024 games, at most 30 seconds and 1..4096 submissions per game")
    if args.output.exists():
        parser.error("preserve evidence; choose a fresh output")
    if not args._worker:
        try:
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_worker"],
                                   timeout=args.max_seconds + 10, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            write_record(args.output, {"schema": "cabt-signed-energy-diagnostic/1", "status": "incomplete",
                                       "reason": "worker_timeout", "optimizer_updates": 0})
            return 1
        sys.stdout.write(child.stdout)
        sys.stderr.write(child.stderr)
        if not args.output.exists():
            write_record(args.output, {"schema": "cabt-signed-energy-diagnostic/1", "status": "incomplete",
                                       "reason": "worker_exit", "returncode": child.returncode,
                                       "optimizer_updates": 0})
        return child.returncode
    record = run_probe(args)
    write_record(args.output, record)
    print(json.dumps({key: record[key] for key in ("status", "wall_seconds", "optimizer_updates")}))
    projection_ok = not args.check_projection or record.get("projection", {}).get("projection_passed") is True
    return 0 if record["status"] == "reproduced" and projection_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
