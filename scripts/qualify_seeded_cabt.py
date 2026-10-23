"""Check explicit uint64 CABT replay with complete games and interleaved noise."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tcg.cabt import CabtBattle

SEEDS = (0, 1, 17, 2**32, 2**32 + 1, 2**63, 2**64 - 2, 2**64 - 1)
STOCHASTIC_DECK = [726] * 4 + [1227] * 4 + [1235] * 4 + [3] * 48


def trace(binary, digest, decks, seed, sampler_seed, extra_reads=False, minimum_non_main=False):
    sampler = random.Random(sampler_seed)
    transcript, events = hashlib.sha256(), Counter()
    with CabtBattle(binary, digest, seed_api="BattleStartSeeded64") as battle:
        frame = battle.start(*decks, engine_seed=seed)
        for submissions in range(4097):
            wire = frame.wire_observation
            transcript.update(json.dumps({"actor": frame.actor, "observation": wire},
                                         sort_keys=True, separators=(",", ":")).encode())
            events.update(str(event["type"]) for event in wire["logs"])
            if frame.result != -1:
                if frame.result not in (0, 1, 2):
                    raise ValueError("unknown native terminal result")
                return {"transcript_sha256": transcript.hexdigest(), "result": frame.result,
                        "submissions": submissions, "coin_logs": events["22"], "shuffle_logs": events["0"]}
            if submissions == 4096:
                raise ValueError("replay trace exceeded submission cap")
            if extra_reads and submissions % 4 == 0:
                battle.observe()
                battle.card_ids()
            prompt = wire["select"]
            if minimum_non_main and prompt["type"] != 0:
                response = list(range(prompt["minCount"]))
            else:
                count = sampler.randint(prompt["minCount"], prompt["maxCount"])
                response = sampler.sample(range(len(prompt["option"])), count)
            transcript.update(json.dumps(response, separators=(",", ":")).encode())
            frame = battle.submit(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--decks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing evidence; choose a fresh output")
    if not args._worker:
        return subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_worker"],
                              timeout=70, check=False).returncode
    raw = args.decks.read_bytes()
    decks = json.loads(raw)["decks"]
    records = []
    for index, seed in enumerate(SEEDS):
        pair = (decks[index % 6]["card_ids"], decks[(index // 6) % 6]["card_ids"])
        first = trace(args.binary, args.sha256, pair, seed, 860500 + index)
        trace(args.binary, args.sha256, pair, seed ^ 0x9E3779B97F4A7C15, 860600 + index)
        second = trace(args.binary, args.sha256, pair, seed, 860500 + index, extra_reads=True)
        if first != second:
            raise ValueError(f"native replay differs for seed {seed}")
        stochastic_pair = (STOCHASTIC_DECK, STOCHASTIC_DECK)
        stochastic = trace(args.binary, args.sha256, stochastic_pair, seed, 860800 + index,
                           minimum_non_main=True)
        trace(args.binary, args.sha256, stochastic_pair, seed ^ 0xD1B54A32D192ED03, 860900 + index)
        stochastic_replay = trace(args.binary, args.sha256, stochastic_pair, seed, 860800 + index,
                                  extra_reads=True, minimum_non_main=True)
        if stochastic != stochastic_replay:
            raise ValueError(f"stochastic replay differs for seed {seed}")
        records.append({"seed": seed, "deck0": decks[index % 6]["name"],
                        "deck1": decks[(index // 6) % 6]["name"], **first, "stochastic_fixture": stochastic})
    common_pair = (decks[0]["card_ids"], decks[0]["card_ids"])
    low = trace(args.binary, args.sha256, common_pair, 0, 860700)
    high = trace(args.binary, args.sha256, common_pair, 2**32, 860700)
    if low["transcript_sha256"] == high["transcript_sha256"]:
        raise ValueError("high uint64 seed word had no observed effect")
    if (not sum(row["stochastic_fixture"]["coin_logs"] for row in records)
            or not sum(row["stochastic_fixture"]["shuffle_logs"] for row in records)):
        raise ValueError("replay fixture missed coins or post-setup shuffles")
    result = {"schema": "cabt-seeded-replay/1", "status": "passed", "engine_sha256": args.sha256,
              "deck_manifest_sha256": hashlib.sha256(raw).hexdigest(), "seed_api": "BattleStartSeeded64",
              "seed_bits": 64, "paired_traces": records, "high_word_effect_observed": True,
              "stochastic_deck_card_ids": STOCHASTIC_DECK,
              "interleaved_noise_and_reads": True, "optimizer_updates": 0,
              "scope": "current_windows_build_actor_json_and_responses_not_opaque_buffers_or_cross_toolchain",
              "implementation_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in (ROOT / "tcg/__init__.py", ROOT / "tcg/cabt.py", Path(__file__))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "paired_traces": len(records), "optimizer_updates": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
