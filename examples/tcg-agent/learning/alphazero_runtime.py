"""Pinned local AlphaZero score service and immutable search/outcome updates."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from learning import alphazero as az
from learning import model6_torch as features
from learning import entity_ppo as native

BASELINE_SHA = "c175603c04df29723998efcd02dc7dc18ae9c1c2ad9953fae520adf0e7c515d9"
SOURCE_PATHS = ("learning/alphazero.py", "learning/alphazero_runtime.py", "learning/alphazero-search.mjs",
                "learning/legal-choice-sets.mjs", "learning/joint_choice_sets.py", "learning/entity_ppo.py",
                "learning/entity_torch.py", "learning/model6_torch.py", "learning/model6-contract.json",
                "learning/model6-features.mjs", "learning/entity-contract.json", "learning/entity-features.mjs",
                "learning/card_action_torch.py", "learning/masked_ppo.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {name: sha256(REPO / name) for name in SOURCE_PATHS}


def records(path):
    path = Path(path)
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("search records exceed byte budget")
    result = []
    with path.open("rb") as stream:
        for line in stream:
            if len(line) > 4 * 1024 * 1024 or len(result) >= 32768 or not line.strip():
                raise ValueError("search record line/count budget exceeded")
            result.append(json.loads(line))
    if not result:
        raise ValueError("actual encoded records required")
    return result


def fresh_output(path):
    output = features.private_checkpoint(path)
    if output.exists() or output.with_suffix(".receipt.json").exists():
        raise ValueError("fresh immutable AlphaZero checkpoint required")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def save(trainer, output, probe, receipt):
    expected = trainer.score(probe)
    with output.open("xb") as stream:
        torch.save(trainer.payload(), stream)
    loaded = az.AlphaZeroTrainer.load(output, trainer.catalog, str(trainer.device))
    if loaded.score(probe) != expected:
        raise ValueError("immutable AlphaZero reload changed policy or value outputs")
    receipt = {**receipt, "schema": "native-alphazero-receipt/1", "checkpoint": str(output),
               "checkpointSha256": sha256(output), "updateCount": trainer.update_count,
               "sourceHashes": source_hashes(), "reloadPolicyValueExact": True, "config": trainer.config}
    with output.with_suffix(".receipt.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def initialize(baseline, output, *, catalog, probe_records, seed=81729, device="cpu"):
    if sha256(baseline) != BASELINE_SHA:
        raise ValueError("exact frozen first-baseline actor/critic required")
    frozen = source_hashes()
    saved = torch.load(baseline, map_location=device, weights_only=True)
    metadata = features.catalog_metadata(catalog)
    if saved.get("catalog") != metadata:
        raise ValueError("baseline catalog differs from AlphaZero catalog")
    trainer = az.AlphaZeroTrainer(metadata, saved["architecture"], seed, device)
    trainer.model.load_state_dict(saved["state_dict"], strict=True)
    if not all(torch.equal(value, trainer.model.state_dict()[name]) for name, value in saved["state_dict"].items()):
        raise ValueError("AlphaZero warm start changed a baseline tensor")
    if trainer.optimizer.state:
        raise ValueError("new objective requires a fresh Adam optimizer")
    probe = [az.feature_record(row) if row.get("schema") == az.RECORD_SCHEMA else row for row in records(probe_records)[:64]]
    trainer.metadata = {"sourceHashes": frozen, "baseCheckpointSha256": BASELINE_SHA,
                        "optimizerResetForNewObjective": True, "warmStartAllTensorsExact": True,
                        "probeRecordsSha256": sha256(probe_records)}
    if source_hashes() != frozen or sha256(baseline) != BASELINE_SHA:
        raise ValueError("initialization inputs changed")
    return save(trainer, fresh_output(output), probe, trainer.metadata)


def update(checkpoint, record_path, output, *, catalog, device="cuda"):
    trainer = az.AlphaZeroTrainer.load(checkpoint, features.catalog_metadata(catalog), device)
    frozen = source_hashes()
    if trainer.metadata.get("sourceHashes") != frozen:
        raise ValueError("AlphaZero parent source freeze mismatch")
    parent_sha, record_sha = sha256(checkpoint), sha256(record_path)
    rows = records(record_path)
    metrics = trainer.update(rows)
    if source_hashes() != frozen or sha256(checkpoint) != parent_sha or sha256(record_path) != record_sha:
        raise ValueError("AlphaZero update inputs changed")
    trainer.metadata = {**trainer.metadata, "parentCheckpointSha256": parent_sha, "recordsSha256": record_sha}
    return save(trainer, fresh_output(output), [az.feature_record(row) for row in rows[:64]],
                {**trainer.metadata, "metrics": metrics})


def serve(checkpoint, *, catalog, device="cpu", seed=81729):
    trainer = az.AlphaZeroTrainer.load(checkpoint, features.catalog_metadata(catalog), device)
    if trainer.metadata.get("sourceHashes") != source_hashes():
        raise ValueError("AlphaZero service source freeze mismatch")
    identity = {"schema": az.SCHEMA, "trainingMethod": az.METHOD, "featureContract": features.CONTRACT,
                "contractSha256": features.CONTRACT_SHA256, **trainer.catalog, "checkpoint": str(Path(checkpoint).resolve()),
                "checkpointSha256": sha256(checkpoint), "samplingSeed": seed, "seed": seed,
                "device": str(trainer.device), "sourceHashes": source_hashes(), "updateCount": trainer.update_count,
                "architecture": trainer.architecture, "valueTransform": "tanh/1"}
    for line in sys.stdin:
        request = None
        try:
            if len(line) > 32 * 1024 * 1024:
                raise ValueError("AlphaZero service request exceeds byte budget")
            request = json.loads(line)
            if not isinstance(request, dict) or type(request.get("id")) is not int:
                raise ValueError("correlated integer request required")
            operation = request.get("op")
            if operation == "ping":
                result = identity
            elif operation == "score":
                result = trainer.score(request.get("records"))
            elif operation == "seed":
                if type(request.get("seed")) is not int or not 0 <= request["seed"] <= 0xffffffff:
                    raise ValueError("unsigned search sampling seed required")
                identity["samplingSeed"] = identity["seed"] = request["seed"]
                result = {"samplingSeed": request["seed"], "seed": request["seed"]}
            elif operation == "close":
                result = {}
            else:
                raise ValueError("unsupported AlphaZero service operation")
            response = {"id": request["id"], "ok": True, **result}
        except Exception as error:
            response = {"id": request.get("id") if isinstance(request, dict) else None,
                        "ok": False, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response, allow_nan=False), flush=True)
        if response["ok"] and request["op"] == "close":
            return


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("baseline"); init.add_argument("output"); init.add_argument("--probe-records", required=True)
    init.add_argument("--seed", type=int, default=81729)
    fit = commands.add_parser("update")
    fit.add_argument("checkpoint"); fit.add_argument("records"); fit.add_argument("output")
    service = commands.add_parser("serve")
    service.add_argument("--checkpoint", required=True); service.add_argument("--seed", type=int, default=81729)
    for command, default in ((init, "cpu"), (fit, "cuda"), (service, "cpu")):
        command.add_argument("--catalog", required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default=default)
    args = parser.parse_args()
    if args.command == "serve":
        serve(args.checkpoint, catalog=args.catalog, device=args.device, seed=args.seed)
        return
    result = initialize(args.baseline, args.output, catalog=args.catalog, probe_records=args.probe_records,
                        seed=args.seed, device=args.device) if args.command == "init" else update(
                            args.checkpoint, args.records, args.output, catalog=args.catalog, device=args.device)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
