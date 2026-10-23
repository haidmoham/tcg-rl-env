"""Measure selected native response ordering with seeded counterfactual replay."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tcg.cabt import CabtBattle

ENGINE_SHA256 = "a9792857abafb44709bb9062a6d7cb5aa9cd1148c5168ea3214b20a91b62c0f7"
DECKS_SHA256 = "93150420ccbd048c83f8d5c6c0123c27585dec8ea9996080aa737279b64e3f31"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def frame_digest(frame):
    return digest({"actor": frame.actor, "wire": frame.wire_observation})


def continuing_response(frame, sampler):
    prompt = frame.wire_observation["select"]
    if prompt["type"] == 0:
        for index, option in enumerate(prompt["option"]):
            if option["type"] == 14:
                return [index]
        raise ValueError("main prompt has no end-turn choice")
    return sampler.sample(range(len(prompt["option"])), prompt["minCount"])


def distinct_choices(prompt, cards, area):
    first = None
    if not isinstance(cards, list):
        return None
    for index, option in enumerate(prompt["option"]):
        if option["type"] != 3 or option["area"] != area:
            continue
        identity = cards[option["index"]]["id"]
        if first is None:
            first = (index, identity)
        elif identity != first[1]:
            return [first[0], index], [first[1], identity]
    return None


def discover(args, decks, deadline):
    for game in range(128):
        seed = (args.seed + game) % 2**64
        sampler = random.Random(seed ^ 860991)
        actions, hashes = [], []
        played_by = None
        with CabtBattle(args.binary, ENGINE_SHA256, seed_api="BattleStartSeeded64") as battle:
            frame = battle.start(*decks, engine_seed=seed)
            for _ in range(1024):
                if time.monotonic() >= deadline:
                    raise TimeoutError("search deadline exceeded")
                if frame.result != -1:
                    break
                prompt = frame.wire_observation["select"]
                hashes.append(frame_digest(frame))
                choices = None
                if (args.probe == "ciphermaniac" and played_by == frame.actor
                        and prompt["type"] == 1 and prompt["context"] == 9
                        and prompt["minCount"] == prompt["maxCount"] == 2):
                    choices = distinct_choices(prompt, prompt["deck"], 1)
                elif (args.probe == "kofu" and played_by == frame.actor
                        and prompt["type"] == 1 and prompt["context"] == 10
                        and prompt["minCount"] == prompt["maxCount"] == 2):
                    hand = frame.wire_observation["current"]["players"][frame.actor]["hand"]
                    choices = distinct_choices(prompt, hand, 2)
                elif (args.probe == "bench" and prompt["type"] == 1 and prompt["context"] == 2
                        and prompt["minCount"] <= 2 <= prompt["maxCount"]):
                    hand = frame.wire_observation["current"]["players"][frame.actor]["hand"]
                    choices = distinct_choices(prompt, hand, 2)
                if choices is not None:
                    return {"seed": seed, "game_index": game, "actor": frame.actor,
                            "prefix": actions, "frame_hashes": hashes,
                            "branch_indices": choices[0], "branch_card_ids": choices[1]}
                response = None
                if prompt["type"] == 0:
                    played_by = None
                    hand = frame.wire_observation["current"]["players"][frame.actor]["hand"]
                    for index, option in enumerate(prompt["option"]):
                        target = 1200 if args.probe == "kofu" else 1188
                        if option["type"] == 7 and hand[option["index"]]["id"] == target:
                            response, played_by = [index], frame.actor
                            break
                if response is None:
                    response = continuing_response(frame, sampler)
                actions.append(response)
                frame = battle.submit(response)
    raise ValueError(f"no qualifying {args.probe} prompt within search cap")


def replay(args, decks, found, reverse, deadline):
    with CabtBattle(args.binary, ENGINE_SHA256, seed_api="BattleStartSeeded64") as battle:
        frame = battle.start(*decks, engine_seed=found["seed"])
        for index, response in enumerate(found["prefix"]):
            if time.monotonic() >= deadline:
                raise TimeoutError("replay deadline exceeded")
            if frame_digest(frame) != found["frame_hashes"][index]:
                raise ValueError("prebranch replay diverged")
            frame = battle.submit(response)
        if frame_digest(frame) != found["frame_hashes"][-1]:
            raise ValueError("branch frame replay diverged")
        response = list(reversed(found["branch_indices"])) if reverse else found["branch_indices"]
        adapter_check = None
        if args.check_adapter:
            from tcg.cabt_observation import CabtObservationAdapter
            adapter = CabtObservationAdapter(tuple(str(card) for card in battle.card_ids()))
            adapter.begin_episode(decks)
            try:
                decision = adapter.project(frame.wire_observation, frame.actor)
                ordered = decision.prompt.ordered
                for index in response:
                    decision = decision.pick(str(index))
                actual = [int(index) for index in decision.response()]
                adapter_check = {"requested": response, "actual": actual, "ordered": ordered,
                                 "preserved": actual == response}
            except ValueError as error:
                adapter_check = {"requested": response, "preserved": False,
                                 "projection_error": str(error)}
        existing_bench = [card["id"] for card in frame.wire_observation["current"]["players"][found["actor"]]["bench"]]
        frame = battle.submit(response)
        if args.probe == "kofu":
            return observe_bottom_draws(battle, frame, found["actor"], response, adapter_check, deadline)
        if args.probe == "bench":
            actual_bench = [card["id"] for card in frame.wire_observation["current"]["players"][found["actor"]]["bench"]]
            selected_ids = list(reversed(found["branch_card_ids"])) if reverse else found["branch_card_ids"]
            if actual_bench != existing_bench + selected_ids:
                raise ValueError("native bench positions do not preserve submitted order")
            return {"response": response, "public_bench_card_ids": actual_bench,
                    "postbranch_frame_sha256": frame_digest(frame), "adapter_check": adapter_check,
                    "native_response_accepted": True}
        records = []
        sampler = random.Random(860992)
        for _ in range(128):
            if time.monotonic() >= deadline:
                raise TimeoutError("postbranch deadline exceeded")
            records.append({"frame_sha256": frame_digest(frame), "actor": frame.actor})
            for event in frame.wire_observation["logs"]:
                if (event["type"] == 4 and event["playerIndex"] == found["actor"]
                        and frame.actor == found["actor"] and event.get("cardId", 0) > 0):
                    return {"response": response, "first_public_draw_card_id": event["cardId"],
                            "draw_observation_actor": frame.actor, "draw_event": event,
                            "postbranch_frames": records, "native_response_accepted": True,
                            "adapter_check": adapter_check}
            if frame.result != -1:
                raise ValueError("game ended before the branch actor's next visible draw")
            frame = battle.submit(continuing_response(frame, sampler))
        raise ValueError("no visible draw within continuation cap")


def observe_bottom_draws(battle, frame, actor, response, adapter_check, deadline):
    draws, frames = [], []
    sampler = random.Random(860992)
    for _ in range(256):
        if time.monotonic() >= deadline:
            raise TimeoutError("bottom-draw continuation deadline exceeded")
        frames.append(frame_digest(frame))
        if frame.actor == actor:
            draws.extend(event["cardId"] for event in frame.wire_observation["logs"]
                         if event["type"] == 4 and event["playerIndex"] == actor and event.get("cardId", 0) > 0)
            players = frame.wire_observation["current"]["players"]
            if players[actor]["deckCount"] == 0 and len(draws) >= 2:
                return {"response": response, "actor_visible_own_draws": draws,
                        "last_two_draw_card_ids": draws[-2:], "postbranch_frame_sha256": frames,
                        "actor_deck_count": 0, "opponent_deck_count": players[1 - actor]["deckCount"],
                        "native_response_accepted": True, "adapter_check": adapter_check}
        if frame.result != -1:
            raise ValueError("game ended before both bottom cards were visibly drawn")
        frame = battle.submit(continuing_response(frame, sampler))
    raise ValueError("bottom cards were not drawn within continuation cap")


def worker(args):
    started = time.monotonic()
    raw = args.decks.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DECKS_SHA256:
        raise ValueError("candidate deck manifest hash mismatch")
    manifest = json.loads(raw)
    entries = manifest["decks"]
    if len(entries) != 6 or any(len(entry["card_ids"]) != 60 for entry in entries):
        raise ValueError("expected six exact 60-card decks")
    if entries[1]["card_ids"].count(1188) != 1:
        raise ValueError("expected one Ciphermaniac in candidate deck index1")
    diagnostic = [1200 if card == 1188 and args.probe == "kofu" else card for card in entries[1]["card_ids"]]
    decks = (diagnostic, diagnostic)
    found = discover(args, decks, started + 45)
    first = replay(args, decks, found, False, started + 60)
    second = replay(args, decks, found, True, started + 60)
    ids = found["branch_card_ids"]
    if (args.probe == "ciphermaniac" and (first["first_public_draw_card_id"] != ids[1]
            or second["first_public_draw_card_id"] != ids[0])):
        raise ValueError("observed draws do not follow the supplied reversed top-deck order")
    if args.probe == "kofu":
        if (first["actor_visible_own_draws"][:-2] != second["actor_visible_own_draws"][:-2]
                or first["last_two_draw_card_ids"] != list(reversed(second["last_two_draw_card_ids"]))
                or set(first["last_two_draw_card_ids"]) != set(ids)):
            raise ValueError("bottom-order counterfactual differs outside the expected two-card tail")
    adapter_passed = not args.check_adapter or all(branch["adapter_check"]["preserved"] for branch in (first, second))
    sources = [Path(__file__), ROOT / "tcg/cabt.py"]
    if args.check_adapter:
        sources += [ROOT / "tcg/cabt_observation.py", ROOT / "tcg/decision.py", ROOT / "tcg/policy.py"]
    return {"schema": "cabt-native-ordering/1", "status": "passed" if adapter_passed else "failed",
            "probe": args.probe, "adapter_checked": args.check_adapter,
            "adapter_order_preserved": adapter_passed if args.check_adapter else None,
            "engine_sha256": ENGINE_SHA256, "seed_api": "BattleStartSeeded64", "seed_bits": 64,
            "deck_manifest_sha256": hashlib.sha256(raw).hexdigest(), "deck_indices": [1, 1],
            "executed_deck_pair_sha256": digest(decks),
            "diagnostic_deck_substitution": {"from_id": 1188, "to_id": 1200, "quantity_each_deck": 1}
                if args.probe == "kofu" else None,
            "deck_names": [entries[1]["name"] + ("_kofu_diagnostic" if args.probe == "kofu" else "")] * 2,
            "discovery": found,
            "branches": [first, second], "same_card_set": True, "identical_prebranch_frames": True,
            "different_actor_visible_first_draw": True if args.probe == "ciphermaniac" else None,
            "different_public_bench_positions": True if args.probe == "bench" else None,
            "different_actor_visible_bottom_draw_order": True if args.probe == "kofu" else None,
            "optimizer_updates": 0,
            "elapsed_seconds": time.monotonic() - started,
            "scope": "selected_native_prompt_order_and_optional_adapter_not_full_prompt_coverage",
            "implementation_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sources}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--decks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2**32 + 860993)
    parser.add_argument("--probe", choices=("ciphermaniac", "bench", "kofu"), default="ciphermaniac")
    parser.add_argument("--check-adapter", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing evidence; choose a fresh output")
    if not 0 <= args.seed < 2**64:
        parser.error("seed must be an unsigned 64-bit integer")
    if not args._worker:
        try:
            return subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_worker"],
                                  timeout=70, check=False).returncode
        except subprocess.TimeoutExpired:
            result = {"schema": "cabt-native-ordering/1", "status": "timeout",
                      "reason": "native worker exceeded 70-second watchdog", "optimizer_updates": 0}
    else:
        try:
            result = worker(args)
        except Exception as error:
            result = {"schema": "cabt-native-ordering/1", "status": "failed",
                      "reason": f"{type(error).__name__}: {error}", "optimizer_updates": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "optimizer_updates")}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
