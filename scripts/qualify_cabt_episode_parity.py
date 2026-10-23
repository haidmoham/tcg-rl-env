"""Measure eight seeded CPU complete games and optionally compare behavior receipts."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ((0, 1), (1, 2), (2, 3), (3, 0))
TOLERANCE = 2e-5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, receipt):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def compare(receipt, reference):
    if reference.get("status") != "passed" or len(reference.get("games", [])) != 8:
        raise ValueError("reference must contain eight successfully completed games")
    for key in ("schema", "identities", "runtime", "cohort", "tool_sha256"):
        if receipt[key] != reference[key]:
            raise ValueError(f"reference {key} differs")
    sources = lambda value: {k: v for k, v in value["sources"].items() if k != "tcg/policy.py"}
    if sources(receipt) != sources(reference):
        raise ValueError("non-policy implementation identity differs")
    maxima = {"log_probability": 0.0, "value": 0.0}
    for actual, expected in zip(receipt["games"], reference["games"], strict=True):
        for key in ("index", "transcript_sha256", "end", "winner", "submissions", "ordered_checks",
                    "multi_selections", "action_ids", "parameters_unchanged", "reason"):
            if actual[key] != expected[key]:
                raise ValueError(f"game {actual['index']} differs: {key}")
        for key in maxima:
            if not all(math.isfinite(value) for value in (*actual[key], *expected[key])):
                raise ValueError("nonfinite policy statistics")
            differences = [abs(a - b) for a, b in zip(actual[key], expected[key], strict=True)]
            maxima[key] = max(maxima[key], max(differences, default=0.0))
    if max(maxima.values()) > TOLERANCE:
        raise ValueError(f"policy statistics exceed tolerance: {maxima}")
    return {"passed": True, "absolute_tolerance": TOLERANCE, "maximum_absolute_error": maxima}


def measure(args, receipt):
    sys.path.insert(0, str(ROOT))
    import platform
    import torch
    from tcg.cabt import CabtBattle
    from tcg.cabt_observation import CabtObservationAdapter
    from tcg.collection import CollectionIdentity, GameAssignment
    from tcg.episode_runner import EpisodeLimits, FrozenPolicy, deck_sha256, projection_sha256, run_episode
    from tcg.population import Snapshot

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(args.protocol.read_bytes())
    deck_path = ROOT / protocol["deck_manifest"]
    identities = {"engine": sha(args.binary), "protocol": sha(args.protocol),
                  "projection": projection_sha256(), "weights": sha(args.checkpoint / "weights.pt"),
                  "decks": sha(deck_path)}
    required = {"engine": protocol["engine"]["sha256"], "projection": protocol["projection_sha256"],
                "weights": protocol["model"]["initial_weights_sha256"],
                "decks": protocol["deck_manifest_sha256"]}
    if any(identities[key] != value for key, value in required.items()):
        raise ValueError("artifact or actual projection source differs from protocol pins")
    if protocol["engine"].get("seed_api") != "BattleStartSeeded64":
        raise ValueError("protocol must declare the explicit uint64 seeded ABI")
    receipt.update(identities=identities,
                   runtime={"python": platform.python_version(), "torch": torch.__version__,
                            "platform": platform.platform(), "cpu_threads": 1, "device": "cpu"},
                   sources={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / "tcg").glob("*.py"))},
                   tool_sha256=sha(__file__))
    decks = [tuple(row["card_ids"]) for row in json.loads(deck_path.read_bytes())["decks"]]
    identity = CollectionIdentity(identities["weights"], 0, identities["engine"],
                                  identities["projection"], identities["protocol"])
    write(args.output, receipt)

    class TracedBattle(CabtBattle):
        def __init__(self):
            self.transcript = hashlib.sha256()
            super().__init__(args.binary, identities["engine"], seed_api="BattleStartSeeded64")

        def observe(self):
            frame = super().observe()
            self.transcript.update(json.dumps(["frame", frame.actor, frame.wire_observation],
                                               sort_keys=True, separators=(",", ":")).encode())
            return frame

        def submit(self, indices):
            self.transcript.update(json.dumps(["response", list(indices)], separators=(",", ":")).encode())
            return super().submit(indices)

    policy = None
    for row in receipt["cohort"]:
        learner_deck, opponent_deck = (decks[index] for index in row["pair"])
        seat = row["learner_seat"]
        exact_decks = (learner_deck, opponent_deck) if seat == 0 else (opponent_deck, learner_deck)
        with TracedBattle() as battle:
            adapter = CabtObservationAdapter(tuple(str(card) for card in battle.card_ids()))
            if policy is None:
                policy = FrozenPolicy.from_initial_checkpoint(args.checkpoint, identities["weights"], adapter.vocabulary())
            assignment = GameAssignment(f"parity-{row['index']}", seat, deck_sha256(learner_deck),
                                        deck_sha256(opponent_deck), Snapshot(0, identities["weights"]), identity)
            result = run_episode(battle, assignment, policy, policy, adapter, exact_decks,
                                 torch.Generator().manual_seed(row["learner_seed"]),
                                 torch.Generator().manual_seed(row["opponent_seed"]),
                                 protocol_path=args.protocol, engine_seed=row["native_seed"],
                                 limits=EpisodeLimits(max_seconds=120.0))
            game = {"index": row["index"], "transcript_sha256": battle.transcript.hexdigest(),
                    "end": result.episode.end.value, "winner": result.terminal_result,
                    "reason": result.reason, "submissions": result.submissions_accepted,
                    "ordered_checks": result.ordered_response_checks,
                    "multi_selections": result.multi_selection_submissions,
                    "wall_seconds": result.wall_seconds, "parameters_unchanged": result.parameters_unchanged,
                    "action_ids": [step.action_id for step in result.episode.steps],
                    "log_probability": [step.old_log_probability for step in result.episode.steps],
                    "value": [step.old_value for step in result.episode.steps]}
            receipt["games"].append(game)
            write(args.output, receipt)
            if (game["end"] != "terminal" or not game["parameters_unchanged"]
                    or game["ordered_checks"] != game["submissions"]):
                raise ValueError(f"game {row['index']} failed complete-game admission: {result.reason}")
    if receipt["sources"] != {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / "tcg").glob("*.py"))}:
        raise ValueError("implementation changed during cohort execution")
    if identities != {"engine": sha(args.binary), "protocol": sha(args.protocol),
                      "projection": projection_sha256(), "weights": sha(args.checkpoint / "weights.pt"),
                      "decks": sha(deck_path)}:
        raise ValueError("pinned artifacts changed during cohort execution")
    if args.compare:
        receipt["comparison"] = compare(receipt, json.loads(args.compare.read_bytes()))
        receipt["reference_sha256"] = sha(args.compare)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "protocol", "checkpoint", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args._worker:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump({"status": "reserved", "games": []}, stream)
        try:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_worker"],
                                     timeout=240, check=False)
            receipt = json.loads(args.output.read_bytes())
            if process.returncode and receipt.get("status") != "failed":
                receipt.update(status="failed", error=f"worker_exit_{process.returncode}")
                write(args.output, receipt)
            return process.returncode
        except subprocess.TimeoutExpired:
            receipt = json.loads(args.output.read_bytes())
            receipt.update(status="failed", error="parent_watchdog_240_seconds")
            write(args.output, receipt)
            return 1
    if json.loads(args.output.read_bytes()) != {"status": "reserved", "games": []}:
        raise ValueError("worker output must be freshly reserved by its parent")
    receipt = {"schema": "cabt-episode-behavior-parity/1", "status": "running", "games": [],
               "optimizer_updates": 0, "scope": "eight_fixed_complete_cpu_games_not_exhaustive_legality_or_strength",
               "cohort": [{"index": i * 2 + seat, "pair": list(pair), "learner_seat": seat,
                           "native_seed": (1 << 63) + 860100 + i * 2 + seat,
                           "learner_seed": 861100 + i * 2 + seat,
                           "opponent_seed": 862100 + i * 2 + seat}
                          for i, pair in enumerate(PAIRS) for seat in (0, 1)]}
    started = time.monotonic()
    try:
        measure(args, receipt)
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
    receipt["wall_seconds"] = time.monotonic() - started
    write(args.output, receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "wall_seconds")}))
    return int(receipt["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
