"""JSONL inference for a frozen PPO checkpoint and its state-only critic."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

try:
    from .masked_ppo import CHECKPOINT_SCHEMA, ModelConfig, PPOTrainer, bc
except ImportError:
    from masked_ppo import CHECKPOINT_SCHEMA, ModelConfig, PPOTrainer, bc


def respond(trainer, request):
    if not isinstance(request, dict) or type(request.get("id")) is not int:
        raise ValueError("integer correlated request id required")
    operation = request.get("op")
    if operation == "ping":
        config = trainer.model_config
        return {"schema": config.feature_schema, "checkpointSchema": CHECKPOINT_SCHEMA,
                "stateDim": config.state_dim, "candidateDim": config.candidate_dim,
                "contractSha256": bc.CONTRACT_SHA256 if config == ModelConfig() else None,
                "printingAliases": trainer.printing_aliases, "device": str(trainer.device),
                "updateCount": trainer.update_count, "torch": str(torch.__version__),
                "objective": "masked categorical PPO with state-only critic",
                "selectionPolicy": "external frozen BC model"}
    if operation in ("infer", "infer_batch"):
        deterministic = request.get("deterministic")
        if type(deterministic) is not bool:
            raise ValueError("explicit boolean deterministic required")
        rows = [request] if operation == "infer" else request.get("items")
        if not isinstance(rows, list) or not 1 <= len(rows) <= trainer.config.max_records:
            raise ValueError("nonempty bounded inference items required")
        decisions = trainer.act(rows, deterministic=deterministic)
        mode = "deterministic" if deterministic else "sampled"
        if operation == "infer":
            return {**decisions[0], "index": decisions[0]["actionIndex"], "selectionMode": mode}
        return {"decisions": decisions, "indices": [d["actionIndex"] for d in decisions],
                "oldLogProbs": [d["oldLogProb"] for d in decisions],
                "oldValues": [d["oldValue"] for d in decisions], "selectionMode": mode}
    if operation == "value":
        return {"value": trainer.values([request.get("state")])[0]}
    if operation == "values":
        states = request.get("states")
        if not isinstance(states, list) or not 1 <= len(states) <= trainer.config.max_records:
            raise ValueError("nonempty bounded states required")
        return {"values": trainer.values(states)}
    if operation == "close":
        return {}
    raise ValueError("unsupported operation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sampling-seed", "--seed", type=int, dest="sampling_seed")
    args = parser.parse_args()
    torch.set_num_threads(1)
    trainer = PPOTrainer.load(args.checkpoint, device=args.device)
    checkpoint = Path(args.checkpoint).resolve()
    identity = {"checkpoint": str(checkpoint),
                "checkpointSha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "samplingSeed": args.sampling_seed}
    if args.sampling_seed is not None:
        if not 0 <= args.sampling_seed <= 0xffffffff:
            parser.error("sampling seed must be an integer from 0 through 4294967295")
        trainer.generator.manual_seed(args.sampling_seed)
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            result = respond(trainer, request)
            if request["op"] == "ping":
                result.update(identity)
            response = {"id": request["id"], "ok": True, **result}
        except Exception as error:
            request_id = request.get("id") if isinstance(request, dict) else None
            response = {"id": request_id if type(request_id) is int else None,
                        "ok": False, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response, allow_nan=False), flush=True)
        if response["ok"] and request["op"] == "close":
            break


if __name__ == "__main__":
    main()
