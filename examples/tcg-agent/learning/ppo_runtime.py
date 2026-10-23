"""Private checkpoint initialization, frozen JSONL inference, and one PPO update."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from learning import card_action_torch as bc
from learning import masked_ppo as ppo

MAX_RECORDS = 32768
MAX_RECEIPT_BYTES = 4 * 1024 * 1024
MAX_RECORD_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
SHA = re.compile(r"^[0-9a-f]{64}$")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    return {"learning/masked_ppo.py": sha256(ppo.__file__),
            "learning/ppo_runtime.py": sha256(__file__),
            "learning/card_action_torch.py": sha256(bc.__file__),
            "learning/card-action-features.mjs": sha256(REPO / "learning/card-action-features.mjs")}


def read_json(path, maximum=MAX_RECEIPT_BYTES):
    path = Path(path)
    if path.stat().st_size > maximum:
        raise ValueError("JSON file exceeds admission budget")
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite metric or metadata")
    if isinstance(value, dict):
        for item in value.values():
            finite_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_json(item)


def fresh_output(value):
    path = bc.private_checkpoint(value)
    if path.exists() or path.with_suffix(".receipt.json").exists():
        raise ValueError("fresh checkpoint and receipt paths required")
    return path


def bounded_runtime(trainer):
    config = trainer.config
    if (trainer.model_config != ppo.ModelConfig() or not 1 <= config.epochs <= 10
            or not 1 <= config.minibatch_size <= 512 or config.max_records > MAX_RECORDS
            or config.max_padded_candidates > 131072):
        raise ValueError("bounded established PPO model and allocation limits required")


def probe(trainer):
    rows = [{"schema": bc.SCHEMA, "kind": "ordinary", "state": [0.] * 224,
             "candidates": [[0.] * 191, [0.25] * 191]}]
    trainer.model.eval()
    with torch.inference_mode():
        distribution, values = trainer.model(*ppo.pad_records(rows, trainer.model_config, trainer.device))
    return distribution.logits.detach().clone(), values.detach().clone()


def save_verified(trainer, output, metadata):
    expected = probe(trainer)
    trainer.save(str(output), metadata=metadata)
    reloaded = ppo.PPOTrainer.load(str(output), device=str(trainer.device))
    actual = probe(reloaded)
    if not all(torch.equal(left, right) for left, right in zip(expected, actual)):
        raise ValueError("saved PPO actor logits or critic values changed after reload")
    return {"checkpoint": str(output), "checkpointSha256": sha256(output),
            "parameters": sum(parameter.numel() for parameter in trainer.model.parameters()),
            "reloadLogitsExact": True, "reloadValuesExact": True,
            "updateCount": trainer.update_count}


def write_receipt(output, receipt):
    finite_json(receipt)
    path = output.with_suffix(".receipt.json")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return {"receipt": str(path), "checkpoint": str(output),
            "checkpointSha256": receipt["checkpointSha256"], "updateCount": receipt["updateCount"]}


def initialize(bc_checkpoint, output, *, seed=1729, device="cpu", config=None):
    output = fresh_output(output)
    device = str(bc.configure(device, seed))
    bc_checkpoint = bc.private_checkpoint(bc_checkpoint)
    # Restricted loading and strict warm-start validation also preserve alias checks.
    payload = torch.load(bc_checkpoint, map_location=device, weights_only=True)
    trainer = ppo.PPOTrainer(config=ppo.PPOConfig(**(config or {})), seed=seed, device=device)
    bounded_runtime(trainer)
    trainer.warm_start(payload)
    if any(not torch.equal(value, trainer.model.actor.state_dict()[key])
           for key, value in payload["state_dict"].items()):
        raise ValueError("BC actor changed during warm start")
    parent_sha = sha256(bc_checkpoint)
    metadata = {"bcCheckpointSha256": parent_sha, "frozenSelectionSha256": parent_sha,
                "sourceHashes": source_hashes(), "featureContract": bc.CONTRACT,
                "contractSha256": bc.CONTRACT_SHA256, "printingAliases": trainer.printing_aliases}
    receipt = {"schema": "native-masked-ppo-init/1", **metadata,
               **save_verified(trainer, output, metadata), "device": device, "seed": seed}
    return write_receipt(output, receipt)


def respond(trainer, request, identity):
    if not isinstance(request, dict) or type(request.get("id")) is not int:
        raise ValueError("integer correlated request id required")
    operation = request.get("op")
    if operation == "ping":
        return {"schema": ppo.CHECKPOINT_SCHEMA, "featureSchema": bc.SCHEMA,
                "stateDim": 224, "candidateDim": 191, "contractSha256": bc.CONTRACT_SHA256,
                "printingAliases": trainer.printing_aliases, "device": str(trainer.device),
                "parameters": sum(parameter.numel() for parameter in trainer.model.parameters()),
                "updateCount": trainer.update_count, **identity}
    if operation == "seed":
        seed = request.get("seed")
        if type(seed) is not int or not 0 <= seed <= 0xffffffff:
            raise ValueError("sampling seed must be an integer from 0 through 4294967295")
        trainer.generator.manual_seed(seed)
        identity["samplingSeed"] = seed
        return {"samplingSeed": seed}
    if operation == "act":
        records = request.get("records")
        if not isinstance(records, list) or not 1 <= len(records) <= 64:
            raise ValueError("act requires 1 through 64 complete records")
        if type(request.get("deterministic")) is not bool:
            raise ValueError("explicit boolean deterministic required")
        return {"actions": trainer.act(records, deterministic=request["deterministic"])}
    if operation == "values":
        states = request.get("states")
        if not isinstance(states, list) or not 1 <= len(states) <= 64:
            raise ValueError("values requires 1 through 64 states")
        return {"values": trainer.values(states)}
    if operation == "close":
        return {}
    raise ValueError("unsupported operation")


def serve(checkpoint, *, device="cpu", seed=1729):
    bc.configure(device, seed)
    checkpoint = bc.private_checkpoint(checkpoint)
    trainer = ppo.PPOTrainer.load(str(checkpoint), device=device)
    bounded_runtime(trainer)
    trainer.generator.manual_seed(seed)
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True).get("metadata", {})
    identity = {"checkpoint": str(checkpoint), "checkpointSha256": sha256(checkpoint),
                "samplingSeed": seed, "sourceHashes": source_hashes(),
                "frozenSelectionSha256": metadata.get("frozenSelectionSha256")}
    while True:
        line = sys.stdin.readline(32 * 1024 * 1024 + 1)
        if not line:
            return
        request = None
        try:
            if len(line) > 32 * 1024 * 1024:
                raise ValueError("request exceeds JSONL budget")
            request = json.loads(line)
            result = respond(trainer, request, identity)
            response = {"id": request["id"], "ok": True, **result}
            finite_json(response)
        except Exception as error:
            request_id = request.get("id") if isinstance(request, dict) else None
            response = {"id": request_id if type(request_id) is int else None,
                        "ok": False, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response, allow_nan=False), flush=True)
        if len(line) > 32 * 1024 * 1024 or response["ok"] and request["op"] == "close":
            return


def rollout_file(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("rollout files must use relative paths")
    path = (root / value).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("rollout path escapes its directory")
    return path


def checked_hash(path, expected):
    if not isinstance(expected, str) or not SHA.fullmatch(expected) or sha256(path) != expected:
        raise ValueError("raw SHA256 mismatch")


def admit_rollout(directory, checkpoint, parent_metadata):
    root = Path(directory).resolve()
    receipt_path = root / "receipt.json"
    receipt = read_json(receipt_path)
    selection_sha = parent_metadata.get("frozenSelectionSha256")
    if (not isinstance(selection_sha, str) or not SHA.fullmatch(selection_sha)
            or receipt.get("schema") != "native-ppo-rollout/1" or receipt.get("complete") is not True
            or receipt.get("fatal") is not False or receipt.get("featureContract") != bc.CONTRACT
            or receipt.get("checkpointsUnchanged") is not True
            or receipt.get("parentCheckpointSha256") != sha256(checkpoint)
            or receipt.get("selectionCheckpointSha256") != selection_sha):
        raise ValueError("rollout authority, completion, or checkpoint identity mismatch")
    sources = receipt.get("sourceHashes")
    current_sources = source_hashes()
    if not isinstance(sources, dict) or any(sources.get(key) != value for key, value in current_sources.items()):
        raise ValueError("rollout runtime source freeze mismatch")
    for relative, expected in sources.items():
        checked_hash(rollout_file(REPO, relative), expected)
    pins = receipt.get("pins")
    if not isinstance(pins, dict) or not pins or any(not isinstance(value, str) or not SHA.fullmatch(value)
                                                   for value in pins.values()):
        raise ValueError("nonempty SHA256 native pins required")
    counts = receipt.get("counts")
    if not isinstance(counts, dict) or any(type(counts.get(key)) is not int or counts[key] != 0
                                          for key in ("fatalErrors", "errors", "rejected")):
        raise ValueError("zero fatal errors, errors, and rejected counts required")
    for key, value in counts.items():
        if any(token in key.lower() for token in ("error", "reject")) and (type(value) is not int or value != 0):
            raise ValueError("rollout contains rejected or erroneous decisions")
    path = rollout_file(root, receipt.get("recordsPath"))
    if path.stat().st_size > MAX_TOTAL_BYTES:
        raise ValueError("rollout exceeds raw byte budget")
    checked_hash(path, receipt.get("recordsSha256"))
    records = []
    parsed_hash = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            parsed_hash.update(line)
            if len(line) > MAX_RECORD_BYTES or not line.strip() or len(records) >= MAX_RECORDS:
                raise ValueError("rollout record or allocation budget exceeded")
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("kind") != "ordinary":
                raise ValueError("only attributed ordinary decisions enter PPO")
            records.append(row)
    if parsed_hash.hexdigest() != receipt["recordsSha256"]:
        raise ValueError("parsed rollout raw SHA256 mismatch")
    if (not records or type(receipt.get("recordsCount")) is not int
            or len(records) != receipt["recordsCount"] or counts.get("records") != len(records)):
        raise ValueError("rollout record count mismatch")
    episodes = receipt.get("episodes")
    if not isinstance(episodes, list) or not episodes or len(episodes) > MAX_RECORDS:
        raise ValueError("bounded episode manifest required")
    admitted = {}
    totals = {"terminated": 0, "truncated": 0, "excluded": 0}
    for episode in episodes:
        episode_id = episode.get("episodeId", episode.get("id")) if isinstance(episode, dict) else None
        status = episode.get("status") if isinstance(episode, dict) else None
        if not isinstance(episode_id, str) or not episode_id or episode_id in admitted or status not in totals:
            raise ValueError("invalid or duplicate episode manifest")
        totals[status] += 1
        admitted[episode_id] = episode
        if status == "excluded":
            if not isinstance(episode.get("exclusionReason"), str) or not episode["exclusionReason"]:
                raise ValueError("excluded episode requires an explicit reason")
            continue
        if episode.get("terminated") is not (status == "terminated") or episode.get("truncated") is not (status == "truncated"):
            raise ValueError("episode ending flags contradict status")
        if "recordsPath" in episode:
            checked_hash(rollout_file(root, episode["recordsPath"]), episode.get("recordsSha256"))
    if counts.get("episodes") != len(episodes) or any(counts.get(key) != value for key, value in totals.items()):
        raise ValueError("episode count mismatch")
    grouped = {}
    for row in records:
        episode_id = row.get("episodeId")
        if episode_id not in admitted or admitted[episode_id]["status"] == "excluded":
            raise ValueError("record has no admitted episode")
        grouped.setdefault(episode_id, []).append(row)
    for episode_id, episode in admitted.items():
        if episode["status"] == "excluded":
            continue
        rows = grouped.get(episode_id, [])
        if not rows or episode.get("recordsCount") != len(rows):
            raise ValueError("episode record count mismatch")
        terminal = episode["status"] == "terminated"
        last = rows[-1]
        if last.get("terminated") is not terminal or last.get("truncated") is not (not terminal) or last.get("segmentEnd") is not True:
            raise ValueError("episode final record contradicts actual ending")
        if any(row.get("terminated") or row.get("truncated") or row.get("reward") != 0 for row in rows[:-1]):
            raise ValueError("episode contains premature endings or nonterminal rewards")
        if terminal and (last.get("reward") not in (-1, 0, 1) or last.get("nextValue") != 0):
            raise ValueError("natural result requires bounded terminal reward and zero bootstrap")
        if terminal and (episode.get("result") not in ("win", "loss", "draw")
                         or last["reward"] != {"win": 1, "loss": -1, "draw": 0}[episode["result"]]):
            raise ValueError("terminal reward does not match the native result")
        if not terminal and episode.get("result") is not None:
            raise ValueError("truncation cannot claim a terminal result")
        if not terminal and last.get("reward") != 0:
            raise ValueError("truncation cannot invent terminal reward")
    return records, receipt, sha256(receipt_path)


def update_once(checkpoint, rollout_directory, output, *, device="cpu"):
    output = fresh_output(output)
    checkpoint = bc.private_checkpoint(checkpoint)
    torch.set_num_threads(1)
    trainer = ppo.PPOTrainer.load(str(checkpoint), device=device)
    bounded_runtime(trainer)
    parent = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = parent.get("metadata", {})
    frozen_sources = source_hashes()
    if metadata.get("sourceHashes") != frozen_sources:
        raise ValueError("parent checkpoint source freeze mismatch")
    records, data_receipt, receipt_sha = admit_rollout(rollout_directory, checkpoint, metadata)
    initial_count = trainer.update_count
    metrics = trainer.update(records)
    finite_json(metrics)
    if source_hashes() != frozen_sources:
        raise ValueError("runtime sources changed during the update")
    if trainer.update_count != initial_count + 1 or metrics.get("update_count") != trainer.update_count:
        raise ValueError("exactly one PPO update required")
    metadata = {**metadata, "sourceHashes": source_hashes(), "parentCheckpointSha256": sha256(checkpoint),
                "dataSourceSha256": data_receipt["recordsSha256"], "collectionReceiptSha256": receipt_sha,
                "pins": data_receipt["pins"]}
    receipt = {"schema": "native-masked-ppo-update/1", **metadata, "metrics": metrics,
               **save_verified(trainer, output, metadata), "device": device,
               "featureContract": bc.CONTRACT, "contractSha256": bc.CONTRACT_SHA256}
    return write_receipt(output, receipt)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    service = commands.add_parser("serve")
    service.add_argument("--checkpoint", required=True)
    service.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    service.add_argument("--seed", type=int, default=1729)
    init = commands.add_parser("init")
    init.add_argument("bc_checkpoint")
    init.add_argument("output")
    init.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    init.add_argument("--seed", type=int, default=1729)
    init.add_argument("--config", help="JSON object containing PPOConfig fields")
    update = commands.add_parser("update")
    update.add_argument("checkpoint")
    update.add_argument("rollout_directory")
    update.add_argument("output")
    update.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.command == "serve":
        serve(args.checkpoint, device=args.device, seed=args.seed)
    elif args.command == "init":
        print(json.dumps(initialize(args.bc_checkpoint, args.output, seed=args.seed, device=args.device,
                                    config=read_json(args.config) if args.config else None), allow_nan=False))
    elif args.command == "update":
        print(json.dumps(update_once(args.checkpoint, args.rollout_directory, args.output,
                                     device=args.device), allow_nan=False))


if __name__ == "__main__":
    main()
