"""Private, frozen, policy-only comparison on the qualified common engine."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time


BASELINE = "baseline002_last_verified_endpoint"
PRETRAINED = "mew_pretrained"
RANDOM = "mew_matched_random_seed860201"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


@dataclass(frozen=True)
class GameAssignment:
    game_id: str
    pair_id: str
    leg: int
    effective_seed32: int
    seat_arms: tuple[str, str]
    seat_deck_indices: tuple[int, int]

    @classmethod
    def parse(cls, row):
        result = cls(**{**row, "seat_arms": tuple(row["seat_arms"]),
                        "seat_deck_indices": tuple(row["seat_deck_indices"])})
        if (not 0 < result.effective_seed32 < 2**32 or result.leg not in (0, 1)
                or len(result.seat_arms) != 2 or len(result.seat_deck_indices) != 2
                or any(arm not in (BASELINE, PRETRAINED, RANDOM) for arm in result.seat_arms)
                or any(index not in range(6) for index in result.seat_deck_indices)):
            raise ValueError("invalid game assignment")
        return result


def tensor_digest(tensors):
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(encoded([name, str(value.dtype), list(value.shape)]))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FrozenComparisonPlayers:
    def __init__(self, bindings):
        import torch
        from fastenc.model import load_model
        from fastenc.wide import Evaluator, Expander
        from tcg.cabt_observation import CabtObservationAdapter
        from tcg.mew_baseline_bridge import transport_inference_checkpoint

        self.bindings = bindings
        self.device = torch.device("cpu")
        self.models, self.evaluators, self.model_load_seconds = {}, {}, {}
        for arm in (PRETRAINED, RANDOM):
            start = time.monotonic()
            model = load_model(bindings["checkpoints"][arm], self.device, "none")
            model.float().eval().requires_grad_(False)
            if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
                   for parameter in model.parameters()):
                raise ValueError("Mew model is not CPU float32")
            self.models[arm] = model
            self.evaluators[arm] = Evaluator(model, self.device, bf16=False)
            self.model_load_seconds[arm] = time.monotonic() - start
        static = [tensor_digest(dict(model.named_buffers())) for model in self.models.values()]
        if len(set(static)) != 1:
            raise ValueError("Mew arms have different static buffers")
        self.static_buffers_sha256 = static[0]
        manifest = json.loads((Path(bindings["baseline_checkpoint"]) / "manifest.json").read_text())
        vocabulary = manifest["payload"]["vocabulary"]["card_ids"]
        self.card_ids = tuple(vocabulary)
        adapter = CabtObservationAdapter(self.card_ids)
        start = time.monotonic()
        self.baseline = transport_inference_checkpoint(bindings["baseline_checkpoint"], adapter.vocabulary(),
                                                       bindings["baseline_weights_sha256"])
        self.model_load_seconds[BASELINE] = time.monotonic() - start
        self.baseline_forward_seconds = 0.0
        self._baseline_forward_started = None
        def before_forward(module, inputs):
            self._baseline_forward_started = time.monotonic()
        def after_forward(module, inputs, output):
            self.baseline_forward_seconds += time.monotonic() - self._baseline_forward_started
        self.baseline.policy.register_forward_pre_hook(before_forward)
        self.baseline.policy.register_forward_hook(after_forward)
        self.expander = Expander(self.device, pin=False)
        self.before = self.identities()

    def identities(self):
        return {**{arm: tensor_digest(model.state_dict()) for arm, model in self.models.items()},
                BASELINE: tensor_digest(self.baseline.policy.state_dict())}

    def assert_unchanged(self):
        self.baseline.assert_unchanged()
        if self.identities() != self.before or any(
                module.training for model in self.models.values() for module in model.modules()):
            raise ValueError("frozen player state changed")


def payload_digest(payload, row):
    from fastenc.wide import ARRAYS, FAMILY_ORDER
    counts = payload["family_count"][row]
    tensors = {"family_count": counts}
    for family, array in ARRAYS:
        count = int(counts[FAMILY_ORDER.index(family)])
        tensors[array] = payload[array][row, :count]
    return hashlib.sha256(encoded([tensor_digest(tensors), payload["seq_len"]])).hexdigest()


def run_game(assignment, players, decks, limits, trace_path):
    import torch
    from fastenc.games import Games
    from fastenc.lib import F4_ELIGIBLE_SLOT
    from tcg.cabt_observation import CabtObservationAdapter
    from tcg.mew_baseline_bridge import GreedyBaselineActor, project_mew_observation

    started = time.monotonic()
    receipt = {"assignment": assignment.__dict__, "status": "failed", "winner": None,
               "timing_seconds": {key: 0.0 for key in ("initialize", "encode", "tensorize", "forward", "external_observe_project_choose", "step")},
               "mew_model_calls": {}, "mew_atomic_choices": 0, "external_actions": 0,
               "max_legal_columns": 0, "forced_mew_encoded_choices": 0}
    trace_digest = hashlib.sha256()
    games = None
    baseline_actor = GreedyBaselineActor(players.baseline)
    baseline_forward_before = players.baseline_forward_seconds
    adapter = CabtObservationAdapter(players.card_ids)
    pair = [decks[index]["card_ids"] for index in assignment.seat_deck_indices]
    adapter.begin_episode(pair)
    turn_decisions = {}
    previous_decision = 0
    def deadline(signum, frame):
        raise TimeoutError("per-game wall cap reached")
    old_handler = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, limits["per_game_wall_seconds"])
    with Path(trace_path).open("x") as trace:
        def record(value):
            data = encoded(value) + b"\n"
            trace_digest.update(data)
            trace.write(data.decode())
        try:
            tick = time.monotonic()
            games = Games([pair], seed=assignment.effective_seed32, seeds=[assignment.effective_seed32], threads=1)
            mask = sum(1 << seat for seat, arm in enumerate(assignment.seat_arms) if arm == BASELINE)
            games.set_external(0, mask)
            receipt["timing_seconds"]["initialize"] += time.monotonic() - tick
            while True:
                tick = time.monotonic()
                owner = games.encode()
                receipt["timing_seconds"]["encode"] += time.monotonic() - tick
                external = False
                with owner:
                    rec = {name: int(owner.records[0][name]) for name in owner.records.dtype.names}
                    receipt["last_record"] = rec
                    turn = rec["turn"]
                    delta = rec["decisionIndex"] - previous_decision
                    turn_decisions[turn] = turn_decisions.get(turn, 0) + delta
                    previous_decision = rec["decisionIndex"]
                    if rec["terminal"]:
                        receipt.update(status="complete", winner=rec["winner"])
                        record({"terminal_record": rec})
                        break
                    if (rec["decisionIndex"] >= limits["submissions_per_game"]
                            or turn_decisions[turn] >= limits["submissions_per_turn"]):
                        raise TimeoutError("native submission cap reached")
                    seat = rec["selectPlayer"]
                    arm = assignment.seat_arms[seat]
                    if rec["batchRow"] < 0:
                        if arm != BASELINE or owner.batch is not None:
                            raise ValueError("unexpected external native record")
                        external = True
                    else:
                        if arm == BASELINE or owner.batch is None:
                            raise ValueError("baseline unexpectedly encoded internally")
                        tick = time.monotonic()
                        payload = players.expander.tensors(owner.batch)
                        receipt["timing_seconds"]["tensorize"] += time.monotonic() - tick
                        row = rec["batchRow"]
                        input_sha = payload_digest(payload, row)
                        tick = time.monotonic()
                        logits, _ = players.evaluators[arm](payload)
                        receipt["timing_seconds"]["forward"] += time.monotonic() - tick
                        scores = logits[row]
                        legal = torch.arange(scores.shape[0]) < rec["visibleCount"]
                        legal &= payload["f4_int8"][row, :scores.shape[0], F4_ELIGIBLE_SLOT] > 0
                        if not bool(legal.any()) or not bool(torch.isfinite(scores[legal]).all()):
                            raise ValueError("empty legal mask or nonfinite legal logits")
                        choice = int(scores.masked_fill(~legal, float("-inf")).argmax())
                        count = int(legal.sum())
                        receipt["max_legal_columns"] = max(receipt["max_legal_columns"], count)
                        receipt["forced_mew_encoded_choices"] += count == 1
                        receipt["mew_model_calls"][arm] = receipt["mew_model_calls"].get(arm, 0) + 1
                        receipt["mew_atomic_choices"] += 1
                        record({"arm": arm, "record": rec, "semantic_input_sha256": input_sha,
                                "legal_mask": legal.tolist(), "selected_atomic_column": choice})
                tick = time.monotonic()
                if external:
                    wire = project_mew_observation(games.observe(0))
                    decision = adapter.project(wire, seat)
                    action = baseline_actor.choose(decision)
                    receipt["timing_seconds"]["external_observe_project_choose"] += time.monotonic() - tick
                    record({"arm": arm, "record": rec, "semantic_input_sha256": hashlib.sha256(encoded(wire)).hexdigest(),
                            "initial_actor_decision": asdict(decision),
                            "ordered_physical_response": list(action)})
                    tick = time.monotonic()
                    games.act_external(0, action)
                    receipt["external_actions"] += 1
                else:
                    games.step([choice])
                receipt["timing_seconds"]["step"] += time.monotonic() - tick
        except Exception as error:
            receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
            if games is not None:
                receipt["native_stats"] = games.stats()
                games.close()
    receipt.update(trace_sha256=trace_digest.hexdigest(), total_wall_seconds=time.monotonic() - started,
                   baseline_forward_calls=baseline_actor.forward_calls,
                   baseline_model_forward_seconds=players.baseline_forward_seconds-baseline_forward_before,
                   baseline_microsteps=baseline_actor.microsteps,
                   baseline_forced_microsteps=baseline_actor.forced_microsteps,
                   process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    return receipt


def analyze(comparison, receipts):
    candidate = comparison["candidate"]
    scores = []
    complete = wins = draws = losses = 0
    for assignment, receipt in zip(comparison["games"], receipts):
        if receipt["status"] != "complete":
            scores.append(None)
            continue
        complete += 1
        winner = receipt["winner"]
        if winner not in (0, 1, 2):
            raise ValueError("unknown terminal winner")
        score = 0.5 if winner == 2 else float(assignment["seat_arms"][winner] == candidate)
        wins += score == 1
        draws += score == 0.5
        losses += score == 0
        scores.append(score)
    n = len(scores)
    low = sum(score or 0 for score in scores) / n
    high = sum(1 if score is None else score for score in scores) / n
    radius = math.sqrt(math.log(2 / 0.025) / (2 * (n / 2)))
    return {"planned_games": n, "complete_games": complete, "failed_or_unrun": n-complete,
            "wins": wins, "draws": draws, "losses": losses, "score_bounds": [low, high],
            "simultaneous_hoeffding_interval": [max(0, low-radius), min(1, high+radius)],
            "independent_seat_pair_clusters": n//2, "per_comparison_alpha": 0.025,
            "verdict": "candidate_superior" if complete == n and low-radius > 0.5 else "inconclusive",
            "complete_game_wall_seconds": [r["total_wall_seconds"] for r in receipts if r["status"] == "complete"]}


def validate_schedule(protocol):
    seeds, game_ids = set(), set()
    if protocol.get("purpose_kind") == "corrected_baseline_confirmation":
        if (len(protocol["comparisons"]) != 1
                or protocol["comparisons"][0]["candidate"] != PRETRAINED
                or protocol["comparisons"][0]["opponent"] != BASELINE):
            raise ValueError("confirmation must contain only pretrained versus baseline")
    elif len(protocol["comparisons"]) != 2:
        raise ValueError("two primary comparisons required")
    for comparison in protocol["comparisons"]:
        games = [GameAssignment.parse(row) for row in comparison["games"]]
        if len(games) != 72:
            raise ValueError("exactly 72 scheduled games required per contrast")
        pairs = {}
        for game in games:
            if game.game_id in game_ids:
                raise ValueError("duplicate game ID")
            game_ids.add(game.game_id)
            pairs.setdefault(game.pair_id, []).append(game)
        if len(pairs) != 36:
            raise ValueError("36 independent seat pairs required")
        deck_pairs = set()
        for pair in pairs.values():
            if len(pair) != 2:
                raise ValueError("incomplete scheduled seat pair")
            first, second = sorted(pair, key=lambda row: row.leg)
            if (first.leg != 0 or second.leg != 1 or first.effective_seed32 != second.effective_seed32
                    or first.seat_arms != second.seat_arms[::-1]
                    or first.seat_deck_indices != second.seat_deck_indices[::-1]
                    or first.seat_arms != (comparison["candidate"], comparison["opponent"])):
                raise ValueError("invalid swapped-seat pair")
            if first.effective_seed32 in seeds:
                raise ValueError("reused independent pair seed")
            seeds.add(first.effective_seed32)
            deck_pairs.add(first.seat_deck_indices)
        if deck_pairs != {(a, b) for a in range(6) for b in range(6)}:
            raise ValueError("schedule does not cover all ordered deck pairs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--engineering", action="store_true", help="One nonprimary engineering game; never emits a primary result")
    parser.add_argument("--validate-only", action="store_true", help="Verify frozen identities and schedule without importing players or creating a run")
    args = parser.parse_args()
    bindings = json.loads(Path(args.bindings).read_text())
    protocol = json.loads(Path(args.protocol).read_text())
    if file_sha(args.protocol) != bindings["protocol_sha256"]:
        raise ValueError("protocol identity mismatch")
    if not args.engineering and not args.validate_only and protocol.get("execution_authorized") is not True:
        raise ValueError("primary comparison remains held")
    validate_schedule(protocol)
    if bindings.get("schema") != "mew-comparison-local-bindings/1":
        raise ValueError("unsupported local bindings schema")
    runner_sha256 = file_sha(__file__)
    if (runner_sha256 != protocol["runner_sha256"]
            or bindings["artifacts"].get(str(Path(__file__).resolve())) != runner_sha256):
        raise ValueError("the executing runner must match the frozen protocol and artifact manifest")
    if hashlib.sha256(encoded(bindings["artifacts"])).hexdigest() != protocol["artifact_manifest_sha256"]:
        raise ValueError("artifact manifest differs from the frozen protocol")
    for arm, field in ((PRETRAINED, "checkpoint_sha256"), (RANDOM, "weights_sha256")):
        if file_sha(bindings["checkpoints"][arm]) != protocol["players"][arm][field]:
            raise ValueError(f"checkpoint is mislabeled for {arm}")
    if bindings["baseline_weights_sha256"] != protocol["players"][BASELINE]["weights_sha256"]:
        raise ValueError("baseline checkpoint differs from the frozen endpoint")
    for path, expected in bindings["artifacts"].items():
        if file_sha(path) != expected:
            raise ValueError(f"artifact identity mismatch: {path}")
    os.environ["FASTENC_LIB"] = bindings["engine_library"]
    if bindings["engine_library"] not in bindings["artifacts"]:
        raise ValueError("engine library must be pinned")
    if any(path not in bindings["artifacts"] for path in bindings["checkpoints"].values()):
        raise ValueError("both Mew checkpoints must be pinned")
    decks_path = bindings["deck_manifest"]
    if file_sha(decks_path) != protocol["deck_manifest_sha256"]:
        raise ValueError("deck manifest mismatch")
    decks = json.loads(Path(decks_path).read_text())["decks"]
    for index, deck in enumerate(decks):
        if len(deck["card_ids"]) != 60 or hashlib.sha256(encoded(deck["card_ids"])).hexdigest() != protocol["decks"][index]["ordered_card_ids_sha256"]:
            raise ValueError("ordered deck identity mismatch")
    if args.validate_only:
        print(json.dumps({"status": "validated", "protocol_sha256": file_sha(args.protocol),
                          "bindings_sha256": file_sha(args.bindings), "runner_sha256": runner_sha256,
                          "artifacts": len(bindings["artifacts"]),
                          "scheduled_games": sum(len(row["games"]) for row in protocol["comparisons"]),
                          "model_loads": 0, "games_started": 0, "execution_authorized": protocol["execution_authorized"]}))
        return 0
    sys.path[:0] = [bindings["kit_python"], bindings["project_python"]]
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    launch = {"protocol_sha256": file_sha(args.protocol), "bindings_sha256": file_sha(args.bindings),
              "runner_sha256": file_sha(__file__), "mode": "engineering_nonprimary" if args.engineering else "primary",
              "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(),
                          "cpu_threads": 1, "mkldnn": False, "dtype": "float32", "inference": "eager_policy_only"},
              "forced_actions": "Mew native auto-forced bypasses model; baseline forwards every microstep",
              "baseline_forward_timer": "read-only module pre/post hooks; trace hashing and actor projection excluded",
              "source_and_artifacts": bindings["artifacts"], "optimizer_updates": 0}
    write_new(output / "launch.json", launch)
    players = FrozenComparisonPlayers(bindings)
    write_new(output / "players.json", {"state_sha256": players.before, "static_buffers_sha256": players.static_buffers_sha256,
                                       "model_load_and_reader_warm_seconds": players.model_load_seconds,
                                       "baseline_transport": players.baseline.receipt})
    comparisons = protocol["comparisons"]
    if args.engineering:
        comparisons = [{"comparison_id": "engineering_nonprimary", "candidate": PRETRAINED,
                        "games": [{"game_id": "engineering/nonprimary/0", "pair_id": "engineering/nonprimary", "leg": 0,
                                   "effective_seed32": 860401, "seat_arms": [PRETRAINED, BASELINE], "seat_deck_indices": [0, 1]}]}]
    results = []
    all_complete = True
    for comparison in comparisons:
        start = time.monotonic()
        receipts = []
        for index, raw in enumerate(comparison["games"]):
            assignment = GameAssignment.parse(raw)
            stem = f"{len(results):02d}-{index:03d}"
            if time.monotonic() - start >= bindings.get("phase_wall_seconds", 3600):
                receipt = {"assignment": assignment.__dict__, "status": "unrun_phase_cap", "winner": None}
            else:
                receipt = run_game(assignment, players, decks, protocol["limits"], output / f"{stem}.trace.jsonl")
            players.assert_unchanged()
            receipt["players_unchanged"] = True
            write_new(output / f"{stem}.json", receipt)
            receipts.append(receipt)
            print(json.dumps({"game_id": assignment.game_id, "status": receipt["status"], "winner": receipt.get("winner"),
                              "seconds": receipt.get("total_wall_seconds")}), flush=True)
        results.append({"comparison_id": comparison["comparison_id"],
                        "result": None if args.engineering else analyze(comparison, receipts),
                        "phase_wall_seconds": time.monotonic() - start,
                        "receipts": [f"{len(results):02d}-{i:03d}.json" for i in range(len(receipts))]})
        all_complete &= all(row["status"] == "complete" for row in receipts)
    for path, expected in bindings["artifacts"].items():
        if file_sha(path) != expected:
            raise ValueError(f"artifact changed during run: {path}")
    players.assert_unchanged()
    write_new(output / "result.json", {"mode": launch["mode"], "comparisons": results,
                                      "all_scheduled_games_complete": all_complete,
                                      "players_and_artifacts_unchanged": True, "optimizer_updates": 0})
    return 0 if all_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
