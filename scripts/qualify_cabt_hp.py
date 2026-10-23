"""Probe pending-KO negative HP with directed native responses and one CPU inference."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SAMPLER_SEED = 860303
SCOPE = "DIRECTED_RESPONSE_DIAGNOSTIC_NOT_POLICY_PLAY"


def integer(value, name, minimum=0):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"invalid {name}")
    return value


def target_hp(wire, option):
    if type(option) is not dict or option.get("type") != 3 or type(option.get("type")) is not int:
        raise ValueError("damage target must be a card option")
    area = integer(option.get("area"), "target area")
    seat = integer(option.get("playerIndex"), "target seat")
    index = integer(option.get("index"), "target index")
    if area not in (4, 5) or seat not in (0, 1):
        raise ValueError("damage target must be a visible active or bench pokemon")
    try:
        zone = wire["current"]["players"][seat]["active" if area == 4 else "bench"]
        if type(zone) is not list or index >= len(zone) or type(zone[index]) is not dict:
            raise ValueError("damage target is not visible")
        return integer(zone[index].get("hp"), "visible HP", None)
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("missing visible damage target") from error


def choose_response(wire, sampler):
    prompt = wire.get("select")
    if type(prompt) is not dict or type(prompt.get("option")) is not list:
        raise ValueError("missing legal prompt")
    options = prompt["option"]
    lower = integer(prompt.get("minCount"), "minimum")
    upper = integer(prompt.get("maxCount"), "maximum")
    if not lower <= upper <= len(options):
        raise ValueError("unsupported cardinality or repetition semantics")
    context = integer(prompt.get("context"), "context")
    if context == 14:
        if upper > 1:
            raise ValueError("unsupported multi-choice damage allocation")
        values = [target_hp(wire, option) for option in options]
        if lower == upper == 1:
            return [min(range(len(options)), key=values.__getitem__)]
    count = sampler.randint(lower, upper)
    return sampler.sample(range(len(options)), count)


def negative_hp(wire):
    players = wire["current"]["players"]
    if type(players) is not list or len(players) != 2:
        raise ValueError("expected two players")
    result = []
    for seat, player in enumerate(players):
        for area, zone in ((4, "active"), (5, "bench")):
            cards = player[zone]
            if type(cards) is not list:
                raise ValueError("invalid pokemon zone")
            for index, card in enumerate(cards):
                if card is None:
                    continue
                hp = integer(card["hp"], "visible HP", None)
                if hp < 0:
                    result.append({"seat": seat, "area": area, "index": index, "hp": hp,
                                   "max_hp": integer(card["maxHp"], "maximum HP")})
    return result


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_negative(frame, battle, decks, checkpoint):
    import torch
    from tcg.cabt_observation import CabtObservationAdapter
    from tcg.checkpoint import load_checkpoint

    torch.set_num_threads(1)
    adapter = CabtObservationAdapter(tuple(str(card) for card in battle.card_ids()))
    adapter.begin_episode(decks)
    policy = load_checkpoint(checkpoint, adapter.vocabulary()).cpu().eval()
    initial = {name: tensor.clone() for name, tensor in policy.state_dict().items()}
    decision = adapter.project(frame.wire_observation, frame.actor)
    with torch.inference_mode():
        prediction = policy([decision])
    legal_logits = prediction.logits[prediction.legal_mask]
    if not legal_logits.numel() or not torch.isfinite(legal_logits).all() or not torch.isfinite(prediction.values).all():
        raise ValueError("nonfinite or empty legal policy output")
    unchanged = all(torch.equal(initial[name], tensor) for name, tensor in policy.state_dict().items())
    if not unchanged:
        raise ValueError("policy parameters changed during inference")
    return {"projection_passed": True, "finite_legal_logits": True, "finite_value": True,
            "legal_choice_count": int(legal_logits.numel()), "parameters_unchanged": unchanged,
            "initialization": "random", "device": "cpu", "policy_forward_calls": 1}


def run_probe(args):
    sys.path.insert(0, str(ROOT))
    from tcg.cabt import CabtBattle

    started = time.monotonic()
    deadline = started + args.max_seconds
    actual_hash = file_hash(args.binary)
    if actual_hash != args.sha256:
        raise ValueError("CABT artifact hash mismatch")
    manifest = json.loads(args.decks.read_text(encoding="utf-8"))
    decks = manifest["decks"]
    if type(decks) is not list or len(decks) != 6:
        raise ValueError("probe requires the six candidate decks")
    for deck in decks:
        if type(deck.get("name")) is not str or not deck["name"]:
            raise ValueError("candidate deck requires a name")
        cards = deck.get("card_ids")
        if type(cards) is not list or len(cards) != 60:
            raise ValueError("candidate deck must contain 60 card IDs")
        for card in cards:
            integer(card, "card ID")
    if len({deck["name"] for deck in decks}) != 6:
        raise ValueError("candidate deck names must be distinct")
    record = {"schema": "cabt-hp-regression/1", "scope": SCOPE,
              "status": "not_observed", "engine_sha256": actual_hash,
              "deck_manifest_sha256": file_hash(args.decks),
              "checkpoint_manifest_sha256": file_hash(args.untrained_policy / "manifest.json"),
              "checkpoint_weights_sha256": file_hash(args.untrained_policy / "weights.pt"),
              "implementation_sha256": {path.relative_to(ROOT).as_posix(): file_hash(path)
                                        for path in (*sorted((ROOT / "tcg").glob("*.py")), Path(__file__).resolve())},
              "engine_seeded": False, "action_sampler_seed": SAMPLER_SEED,
              "optimizer_updates": 0, "training_authorized": False, "ready_to_train": False,
              "complete_prompt_coverage_verified": False, "ordinary_policy_play_verified": False,
              "policy_forward_calls": 0, "games_requested": args.games,
              "max_seconds": args.max_seconds, "max_submissions_per_game": args.max_submissions,
              "games": [], "errors": []}
    sampler = random.Random(SAMPLER_SEED)
    for game_index in range(args.games):
        if time.monotonic() >= deadline:
            record["status"] = "incomplete"
            break
        first = decks[game_index % 6]
        second = decks[(game_index // 6) % 6]
        game = {"game_index": game_index, "deck0": first["name"], "deck1": second["name"], "submissions": 0}
        record["games"].append(game)
        try:
            with CabtBattle(args.binary, args.sha256) as battle:
                frame = battle.start(first["card_ids"], second["card_ids"])
                while True:
                    fixture = negative_hp(frame.wire_observation)
                    if fixture:
                        record["observed"] = {"game_index": game_index, "submissions": game["submissions"],
                                              "negative_hp": fixture}
                        prompt = frame.wire_observation["select"]
                        if type(prompt) is not dict:
                            raise ValueError("negative HP frame has no policy decision")
                        record["observed"].update(prompt_context=prompt["context"], prompt_type=prompt["type"],
                                                  remain_damage_counter=prompt["remainDamageCounter"])
                        record.update(project_negative(frame, battle, (first["card_ids"], second["card_ids"]),
                                                       args.untrained_policy))
                        game["status"] = "stopped_at_negative_hp"
                        record["status"] = "reproduced_and_projected"
                        break
                    if frame.result >= 0:
                        game.update(status="natural_terminal", result=frame.result)
                        break
                    if time.monotonic() >= deadline or game["submissions"] >= args.max_submissions:
                        game["status"] = "censored"
                        record["status"] = "incomplete"
                        break
                    frame = battle.submit(choose_response(frame.wire_observation, sampler))
                    game["submissions"] += 1
        except (RuntimeError, ValueError, KeyError, IndexError, TypeError, OSError) as error:
            record["errors"].append({"game_index": game_index, "type": type(error).__name__})
            game["status"] = "failed"
            record["status"] = "incomplete"
        if record["status"] != "not_observed":
            break
    record["wall_seconds"] = time.monotonic() - started
    return record


def write_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "decks", "untrained-policy", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--max-submissions", type=int, default=4096)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not 1 <= args.games <= 64 or not math.isfinite(args.max_seconds)
            or not 0 < args.max_seconds <= 60 or not 1 <= args.max_submissions <= 4096):
        parser.error("probe requires 1..64 games, 0..60 seconds, and 1..4096 submissions per game")
    if args.output.exists():
        parser.error("preserve existing evidence; choose a fresh output")
    if not args._worker:
        try:
            worker = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_worker"],
                                    timeout=args.max_seconds + 10, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            record = {"schema": "cabt-hp-regression/1", "scope": SCOPE, "status": "incomplete",
                      "reason": "worker_timeout", "hard_deadline_seconds": args.max_seconds + 10,
                      "engine_seeded": False, "optimizer_updates": 0,
                      "training_authorized": False, "ready_to_train": False}
            if not args.output.exists():
                write_record(args.output, record)
            print(json.dumps(record))
            return 1
        sys.stdout.write(worker.stdout)
        sys.stderr.write(worker.stderr)
        if worker.returncode and not args.output.exists():
            write_record(args.output, {"schema": "cabt-hp-regression/1", "scope": SCOPE,
                                      "status": "incomplete", "reason": "worker_exit",
                                      "returncode": worker.returncode, "engine_seeded": False,
                                      "optimizer_updates": 0, "training_authorized": False,
                                      "ready_to_train": False})
        return worker.returncode
    try:
        record = run_probe(args)
    except (RuntimeError, ValueError, KeyError, TypeError, OSError) as error:
        record = {"schema": "cabt-hp-regression/1", "scope": SCOPE, "status": "incomplete",
                  "errors": [{"type": type(error).__name__}], "engine_seeded": False,
                  "optimizer_updates": 0, "training_authorized": False, "ready_to_train": False}
    write_record(args.output, record)
    print(json.dumps({key: record[key] for key in ("status", "scope", "optimizer_updates")}))
    return 0 if record["status"] == "reproduced_and_projected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
