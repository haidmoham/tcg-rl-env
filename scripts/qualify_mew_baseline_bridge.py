"""Bounded actual-artifact qualification of the historical policy on Mew frames."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tcg.cabt_observation import CabtObservationAdapter
from tcg.decision import ActorDecision, LegalPrompt, Option, Token
from tcg.mew_baseline_bridge import (
    BRIDGE_SCHEMA, GreedyBaselineActor, project_mew_observation, transport_inference_checkpoint,
)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def decision_from_record(record):
    def token(value):
        return Token(**{**value, "features": tuple(value["features"])})
    prompt = record["prompt"]
    options = tuple(Option(value["option_id"], token(value["token"]), tuple(value["state_indices"]))
                    for value in prompt["options"])
    return ActorDecision(tuple(token(value) for value in record["tokens"]),
                         LegalPrompt(**{**prompt, "options": options}), tuple(record["selected"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--decks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--library-sha256")
    parser.add_argument("--seed", type=int, default=941780001)
    parser.add_argument("--max-seconds", type=float, default=90)
    parser.add_argument("--deck0", type=int, default=0)
    parser.add_argument("--deck1", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    if (not 0 < args.seed < 2**32 or not 0 < args.max_seconds <= 90
            or sum(value is not None for value in (args.frame, args.library, args.reference)) != 1):
        raise ValueError("choose one frame, forward reference, or native library and bounded seed/deadline")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    adapter = CabtObservationAdapter(tuple(str(index) for index in range(1, 1268)))
    deck_payload = json.loads(args.decks.read_text())
    decks = tuple(tuple(deck["card_ids"]) for deck in deck_payload["decks"])
    pair = (decks[args.deck0], decks[args.deck1])
    adapter.begin_episode(pair)
    sources = {str(path): digest(path) for path in (Path(__file__), *Path(__file__).parents[1].joinpath("tcg").glob("*.py"))}
    checkpoint_before = {path.name: digest(path) for path in args.checkpoint.iterdir()}
    receipt = {"schema": "mew-baseline-bridge-qualification/1", "bridge_schema": BRIDGE_SCHEMA,
               "status": "failed", "source_sha256": sources, "checkpoint_files": checkpoint_before,
               "decks_sha256": digest(args.decks), "deck_indices": [args.deck0, args.deck1],
               "seed": args.seed, "mode": "forward-parity" if args.reference else "captured-frame" if args.frame else "native-game",
               "frames": [], "games_started": 0, "games_complete": 0, "optimizer_updates": 0,
               "scope": ("frozen real-decision cross-runtime forward parity; no optimizer resume claim" if args.reference
                         else "engineering qualification; no primary comparative score or runtime parity claim")}
    start = time.monotonic()
    actor = None
    try:
        policy = transport_inference_checkpoint(args.checkpoint, adapter.vocabulary(), args.weights_sha256)
        receipt["transport"] = policy.receipt
        actor = GreedyBaselineActor(policy)

        def choose(wire):
            projected = project_mew_observation(wire)
            decision = adapter.project(projected, projected["current"]["yourIndex"])
            response = actor.choose(decision)
            receipt["frames"].append({"wire_sha256": sha256(json.dumps(wire, sort_keys=True).encode()).hexdigest(),
                                       "actor": projected["current"]["yourIndex"],
                                       "turn": projected["current"]["turn"],
                                       "select_type": projected["select"]["type"],
                                       "context": projected["select"]["context"],
                                       "option_count": len(projected["select"]["option"]),
                                       "response": response, "initial_decision": asdict(decision)})
            return response

        if args.reference:
            reference = json.loads(args.reference.read_text())
            if (reference["weights_sha256"] != args.weights_sha256
                    or reference["manifest_sha256"] != policy.receipt["manifest_sha256"]
                    or reference["strict_original_runtime_loader"] is not True
                    or reference["policy_frozen_unchanged"] is not True):
                raise ValueError("original-runtime reference identity mismatch")
            receipt["reference_sha256"] = digest(args.reference)
            receipt["parity_tolerance"] = {"rtol": 1e-5, "atol": 1e-6}
            for frame in reference["frames"]:
                decision = decision_from_record(frame["decision"])
                with torch.inference_mode():
                    output = policy.policy((decision,))
                ids = output.option_ids[0]
                mask = output.legal_mask[0].tolist()
                logits = {name: float(output.logits[0, index]) for index, name in enumerate(ids) if mask[index]}
                expected = torch.tensor([frame["legal_logits"][key] for key in logits], dtype=torch.float32)
                actual = torch.tensor(list(logits.values()), dtype=torch.float32)
                value = float(output.values[0])
                greedy = ids[int(output.logits[0].argmax())]
                passed = (list(ids) == frame["option_ids"] and mask == frame["legal_mask"]
                          and set(logits) == set(frame["legal_logits"])
                          and greedy == frame["greedy_choice"]
                          and bool(torch.isclose(actual, expected, rtol=1e-5, atol=1e-6).all())
                          and bool(torch.isclose(torch.tensor(value), torch.tensor(frame["value"]), rtol=1e-5, atol=1e-6)))
                receipt["frames"].append({"rollout_index": frame["rollout_index"], "passed": passed,
                                           "max_legal_logit_absolute_error": float((actual - expected).abs().max()),
                                           "value_absolute_error": abs(value - frame["value"]),
                                           "greedy_choice": greedy, "legal_logits": logits, "value": value,
                                           "option_ids": ids, "legal_mask": mask})
            receipt["parity_forward_calls"] = len(reference["frames"])
            if not receipt["frames"] or not all(frame["passed"] for frame in receipt["frames"]):
                raise ValueError("original versus destination runtime forward parity failed")
        elif args.frame:
            receipt["frame_sha256"] = digest(args.frame)
            choose(json.loads(args.frame.read_text()))
        else:
            if not args.library_sha256 or digest(args.library) != args.library_sha256:
                raise ValueError("native library identity mismatch")
            os.environ["FASTENC_LIB"] = str(args.library.resolve())
            from fastenc.games import Games
            receipt["library_sha256"] = args.library_sha256
            with Games([pair], seed=args.seed, seeds=[args.seed], threads=1) as games:
                receipt["games_started"] = 1
                games.set_external(0, 3)
                per_turn = {}
                for _ in range(4096):
                    if time.monotonic() - start > args.max_seconds:
                        raise TimeoutError("native game wall limit")
                    with games.encode() as encoded:
                        record = encoded.records[0]
                        if int(record["terminal"]):
                            receipt["games_complete"] = 1
                            receipt["winner"] = int(record["winner"])
                            receipt["terminal_turn"] = int(record["turn"])
                            break
                        if int(record["prefixLen"]) != 0:
                            raise ValueError("external native frame has a prefix")
                        seat = int(record["selectPlayer"])
                    wire = games.observe(0)
                    if wire["current"]["yourIndex"] != seat:
                        raise ValueError("record and observer actor mismatch")
                    turn = wire["current"]["turn"]
                    per_turn[turn] = per_turn.get(turn, 0) + 1
                    if per_turn[turn] > 512:
                        raise TimeoutError("per-turn submission limit")
                    games.act_external(0, choose(wire))
                else:
                    raise TimeoutError("native submission limit")
            if digest(args.library) != args.library_sha256:
                raise ValueError("native library changed")
        actor.assert_unchanged()
        receipt["status"] = "passed"
    except Exception as error:
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - start
        receipt["source_unchanged"] = all(digest(path) == expected for path, expected in sources.items())
        receipt["checkpoint_unchanged"] = checkpoint_before == {path.name: digest(path) for path in args.checkpoint.iterdir()}
        if actor is not None:
            receipt["policy_metrics"] = {name: getattr(actor, name) for name in ("forward_calls", "microsteps", "forced_microsteps")}
            try:
                actor.assert_unchanged()
                receipt["policy_unchanged"] = True
            except ValueError:
                receipt["policy_unchanged"] = False
        if not receipt["source_unchanged"] or not receipt["checkpoint_unchanged"] or receipt.get("policy_unchanged") is False:
            receipt["status"] = "failed"
        with args.output.open("x") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        print(json.dumps({key: value for key, value in receipt.items() if key not in ("frames", "source_sha256")}, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
