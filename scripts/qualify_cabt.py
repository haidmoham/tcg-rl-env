"""Run bounded, untrained CABT v1 engineering games. Never performs optimization."""

import argparse
from collections import Counter
import hashlib
import json
import platform
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tcg.cabt import CabtBattle


def play(library, artifact_hash, deck0, deck1, sampler_seed, max_submissions, deadline, policy_context=None):
    sampler = random.Random(sampler_seed)
    prompts = Counter()
    shapes = Counter()
    transcript = hashlib.sha256()
    started = time.perf_counter()
    submissions = 0
    policy_calls = 0
    if policy_context:
        policy, adapter, torch, generator = policy_context
        adapter.begin_episode((deck0, deck1))
        generator.manual_seed(sampler_seed)
    with CabtBattle(library, artifact_hash) as battle:
        frame = battle.start(deck0, deck1)
        while frame.result < 0:
            if submissions >= max_submissions or time.monotonic() >= deadline:
                return {"status": "censored", "reason": "engineering_cap", "submissions": submissions,
                        "wall_seconds": time.perf_counter() - started, "prompts": dict(prompts)}
            prompt = frame.wire_observation["select"]
            if not isinstance(prompt, dict):
                raise ValueError("nonterminal state has no selection prompt")
            options = prompt["option"]
            lower, upper = prompt["minCount"], prompt["maxCount"]
            if any(type(value) is not int for value in (lower, upper)) or not 0 <= lower <= upper <= len(options):
                raise ValueError("prompt requires unsupported cardinality or repetition semantics")
            if policy_context is None:
                count = sampler.randint(lower, upper)
                response = sampler.sample(range(len(options)), count)
            else:
                decision = adapter.project(frame.wire_observation, frame.actor)
                while True:
                    choices = decision.choices()
                    if len(choices) == 1:
                        choice = choices[0]
                    else:
                        with torch.inference_mode():
                            prediction = policy([decision])
                            probabilities = prediction.logits[0].softmax(-1)
                            chosen = int(torch.multinomial(probabilities, 1, generator=generator))
                            choice = prediction.option_ids[0][chosen]
                        policy_calls += 1
                    if choice == "__submit__":
                        response = [int(option) for option in decision.response()]
                        break
                    decision = decision.pick(choice)
            prompts[str(prompt.get("type"))] += 1
            shape = (prompt.get("type"), prompt.get("context"), lower, upper,
                     prompt.get("remainDamageCounter"), prompt.get("remainEnergyCost"))
            shapes[json.dumps(shape, separators=(",", ":"))] += 1
            transcript.update(json.dumps({"actor": frame.actor, "response": response,
                                          "observation": frame.wire_observation}, sort_keys=True).encode())
            frame = battle.submit(response)
            submissions += 1
        return {"status": "natural_terminal", "result": frame.result, "submissions": submissions,
                "wall_seconds": time.perf_counter() - started, "prompts": dict(prompts),
                "prompt_shapes": dict(shapes),
                "policy_forward_calls_both_seats": policy_calls,
                "transcript_sha256": transcript.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--decks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--game-offset", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=60)
    parser.add_argument("--max-submissions", type=int, default=4096)
    parser.add_argument("--untrained-policy", type=Path,
                        help="create or load a strict random-initialization checkpoint; CPU inference only")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not 1 <= args.games <= 64 or not 0 < args.max_seconds <= 1800
            or not 1 <= args.max_submissions <= 4096 or args.game_offset < 0):
        parser.error("qualification must fit 1..64 games, 0..1800 seconds, and 1..4096 submissions")
    if args.output.exists():
        parser.error("output exists; preserve evidence and choose a fresh path")
    if not args._worker:
        try:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                        *sys.argv[1:], "--_worker"],
                                       timeout=args.max_seconds + 10, capture_output=True,
                                       text=True, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            failure = {"schema": "cabt-untrained-engineering/1", "status": "worker_timeout",
                       "optimizer_updates": 0, "hard_deadline_seconds": args.max_seconds + 10}
            output = args.output if not args.output.exists() else args.output.with_name(args.output.stem + "-timeout.json")
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                json.dump(failure, stream, indent=2)
            print(json.dumps(failure))
            return 1
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        return completed.returncode
    manifest_bytes = args.decks.read_bytes()
    manifest = json.loads(manifest_bytes)
    decks = manifest["decks"]
    if not decks:
        parser.error("no candidate decks supplied")
    policy_context = None
    initial_state = None
    if args.untrained_policy:
        import torch
        from tcg.cabt_observation import CabtObservationAdapter
        from tcg.checkpoint import load_checkpoint, save_initial_checkpoint
        torch.set_num_threads(1)
        with CabtBattle(args.binary, args.sha256) as battle:
            adapter = CabtObservationAdapter(tuple(str(card) for card in battle.card_ids()))
        vocabulary = adapter.vocabulary()
        if args.untrained_policy.exists():
            policy = load_checkpoint(args.untrained_policy, vocabulary)
        else:
            policy = save_initial_checkpoint(args.untrained_policy, vocabulary)
        initial_state = {name: tensor.clone() for name, tensor in policy.state_dict().items()}
        policy_context = (policy, adapter, torch, torch.Generator(device="cpu"))
    result = {"schema": "cabt-untrained-engineering/1", "scope": "uniform_response_diagnostic_not_policy_or_training",
              "engine_version": 1, "best_of": 1, "engine_sha256": args.sha256,
              "deck_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "engine_seeded": False, "policy_inference": False, "optimizer_updates": 0,
              "python_version": platform.python_version(), "platform": platform.platform(),
              "implementation_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in (*sorted((ROOT / "tcg").glob("*.py")), Path(__file__).resolve())},
              "complete_prompt_coverage_verified": False, "games": [], "errors": []}
    if policy_context:
        result.update({"scope": "untrained_policy_engineering_not_training", "policy_inference": True,
                       "policy_parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
                       "checkpoint_manifest_sha256": hashlib.sha256((args.untrained_policy / "manifest.json").read_bytes()).hexdigest(),
                       "initialization": "random", "device": "cpu"})
    deadline = time.monotonic() + args.max_seconds
    start = time.perf_counter()
    for index in range(args.games):
        if time.monotonic() >= deadline:
            break
        game_index = args.game_offset + index
        first = decks[game_index % len(decks)]
        second = decks[(game_index // len(decks)) % len(decks)]
        try:
            game = play(args.binary, args.sha256, first["card_ids"], second["card_ids"],
                        860100 + game_index, args.max_submissions, deadline, policy_context)
            game.update({"deck0": first["name"], "deck1": second["name"], "sampler_seed": 860100 + game_index})
            result["games"].append(game)
        except (RuntimeError, ValueError, KeyError, TypeError, OSError) as error:
            result["errors"].append({"game_index": index, "type": type(error).__name__, "message": str(error)})
            break
    result["wall_seconds"] = time.perf_counter() - start
    result["natural_games"] = sum(game["status"] == "natural_terminal" for game in result["games"])
    if policy_context:
        result["parameters_unchanged"] = all(torch.equal(initial_state[name], tensor)
                                             for name, tensor in policy.state_dict().items())
        if not result["parameters_unchanged"]:
            result["errors"].append({"type": "UnexpectedWeightChange"})
    result["status"] = "diagnostic_passed" if result["natural_games"] == args.games and not result["errors"] else "incomplete_or_failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "natural_games", "errors", "optimizer_updates")}, indent=2))
    return 0 if result["status"] == "diagnostic_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
