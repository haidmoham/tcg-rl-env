"""Immutable entity PPO initialization, sampling service, and bounded updates."""

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
from learning import model6_torch as bc
from learning import entity_ppo as ppo

SHA = re.compile(r"^[a-f0-9]{64}$")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_hash(path, expected):
    if not isinstance(expected, str) or not SHA.fullmatch(expected) or sha256(path) != expected:
        raise ValueError("raw-byte SHA256 mismatch")


def read_json(path, maximum=4 * 1024 * 1024):
    path = Path(path)
    if path.stat().st_size > maximum:
        raise ValueError("JSON byte budget exceeded")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def source_hashes():
    names = ("learning/entity_ppo.py", "learning/entity_ppo_runtime.py", "learning/model6_torch.py",
             "learning/train_model6.py", "learning/model6-contract.json", "learning/model6-features.mjs",
             "learning/entity_torch.py", "learning/entity-contract.json", "learning/entity-features.mjs", "learning/masked_ppo.py",
             "learning/card_action_torch.py", "integrations/local-worker/entity-ppo-model.mjs")
    return {name: sha256(REPO / name) for name in names}


def relative_file(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("relative artifact required")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root or not path.is_file():
        raise ValueError("artifact missing or escaped root")
    return path


def read_records(path, maximum=16384):
    path = Path(path)
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("records byte budget exceeded")
    records = []
    with path.open("rb") as stream:
        for line in stream:
            if len(line) > 4 * 1024 * 1024 or not line.strip() or len(records) >= maximum:
                raise ValueError("record byte or count budget exceeded")
            records.append(json.loads(line))
    if not records:
        raise ValueError("nonempty actual encoded records required")
    return records


def fresh_output(value):
    output = bc.private_checkpoint(value)
    if output.exists() or output.with_suffix(".receipt.json").exists():
        raise ValueError("fresh immutable checkpoint and receipt required")
    return output


def probe(trainer, records):
    trainer.model.eval()
    batch, _ = bc.padded_batch(records, trainer.device, trainer.catalog)
    with torch.inference_mode():
        logits = trainer.model.actor(**batch).detach().clone()
        _, values = trainer.model(batch)
    return logits, values.detach().clone()


def save_verified(trainer, output, records, receipt):
    expected = probe(trainer, records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(trainer.payload(), stream)
    reloaded = ppo.EntityPPOTrainer.load(output, trainer.catalog, str(trainer.device))
    actual = probe(reloaded, records)
    if not all(torch.equal(left, right) for left, right in zip(expected, actual)):
        raise ValueError("immutable checkpoint reload changed actor raw logits or critic values")
    receipt = {**receipt, "checkpoint": str(output), "checkpointSha256": sha256(output),
               "updateCount": trainer.update_count, "architecture": trainer.architecture, **trainer.counts(), "reloadLogitsExact": True, "reloadValuesExact": True}
    with output.with_suffix(".receipt.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return {key: receipt[key] for key in ("checkpoint", "checkpointSha256", "updateCount", "actorParameters", "criticExtraParameters", "trainableParameters")}


def initialize(baseline, output, *, catalog, probe_records, bc_sha256, seed=1729, device="cpu", config=None, selection_sha256=None):
    output = fresh_output(output)
    baseline = bc.private_checkpoint(baseline)
    checked_hash(baseline, bc_sha256)
    records = read_records(probe_records)[:64]
    metadata = bc.catalog_metadata(catalog)
    baseline_metadata = torch.load(baseline, map_location="cpu", weights_only=True)
    if baseline_metadata.get("architecture") is None:
        raise ValueError("explicit pre-PPO actor architecture required")
    architecture = bc.resolve_architecture(baseline_metadata["architecture"])
    trainer = ppo.EntityPPOTrainer(metadata, config, seed, device, architecture=architecture)
    warm = trainer.warm_start(str(baseline), records)
    if selection_sha256 is not None and not SHA.fullmatch(selection_sha256):
        raise ValueError("explicit selection checkpoint digest must be SHA256")
    trainer.metadata = {"bcCheckpointSha256": bc_sha256, "frozenSelectionSha256": selection_sha256 or bc_sha256,
                        "sourceHashes": source_hashes(), "probeRecordsSha256": sha256(probe_records),
                        "catalogSha256": metadata["catalogSha256"], "defaultRationale": ppo.DEFAULT_RATIONALE, **warm}
    checked_hash(baseline, bc_sha256)
    return save_verified(trainer, output, records, {"schema": "entity-masked-ppo-init/1", **trainer.metadata,
                         "featureContract": bc.CONTRACT, "contractSha256": bc.CONTRACT_SHA256,
                         "ppoConfig": trainer.payload()["ppoConfig"], "device": str(trainer.device), "seed": seed})


def admit_rollout(directory, checkpoint, trainer):
    root = Path(directory).resolve()
    receipt = read_json(root / "receipt.json")
    metadata = trainer.metadata
    if (receipt.get("schema") != "entity-ppo-rollout/1" or receipt.get("complete") is not True
            or receipt.get("fatal") is not False or receipt.get("checkpointsUnchanged") is not True
            or receipt.get("featureContract") != bc.CONTRACT
            or receipt.get("parentCheckpointSha256") != sha256(checkpoint)
            or receipt.get("selectionCheckpointSha256") != metadata["frozenSelectionSha256"]
            or receipt.get("catalogSha256") != trainer.catalog["catalogSha256"]
            or receipt.get("deterministic") is not False or receipt.get("teacherOverrides") != 0):
        raise ValueError("rollout identity, contract or completion mismatch")
    sources = receipt.get("sourceHashes")
    if not isinstance(sources, dict) or any(sources.get(key) != value for key, value in source_hashes().items()):
        raise ValueError("rollout runtime source freeze mismatch")
    for name, expected in sources.items():
        checked_hash(relative_file(REPO, name), expected)
    pins = receipt.get("pins")
    if not isinstance(pins, dict) or not pins or any(not isinstance(value, str) or not SHA.fullmatch(value) for value in pins.values()):
        raise ValueError("native pinned hashes required")
    path = relative_file(root, receipt.get("recordsPath"))
    checked_hash(path, receipt.get("recordsSha256"))
    records = read_records(path, trainer.config.max_records)
    if type(receipt.get("recordsCount")) is not int or len(records) != receipt["recordsCount"]:
        raise ValueError("rollout count mismatch")
    episodes = receipt.get("episodes")
    if not isinstance(episodes, list) or not episodes or len(episodes) > 16384:
        raise ValueError("all episode outcomes including excluded episodes required")
    admitted, grouped = {}, {}
    for episode in episodes:
        identity = episode.get("episodeId") if isinstance(episode, dict) else None
        if not isinstance(identity, str) or not identity or identity in admitted or type(episode.get("seed")) is not int:
            raise ValueError("unique episode identities and raw seeds required")
        status = episode.get("status")
        if status not in ("terminated", "truncated", "excluded"):
            raise ValueError("explicit episode status required")
        if status == "excluded" and (not isinstance(episode.get("exclusionReason"), str) or not episode["exclusionReason"]):
            raise ValueError("excluded episode requires a reason")
        if status != "excluded" and (episode.get("rejected") != 0 or episode.get("errors") != 0
                or episode.get("auditMode") != "sampled-native-training/1"
                or type(episode.get("privacyChecks")) is not int or episode["privacyChecks"] < 2):
            raise ValueError("admitted episode failed legality or privacy gates")
        admitted[identity] = episode
    for row in records:
        identity = row.get("episodeId")
        if identity not in admitted or admitted[identity]["status"] == "excluded":
            raise ValueError("rollout row has no admitted episode")
        if row.get("behaviorPolicy") != "entity-ppo-categorical/1" or row.get("executedActionIndex") != row.get("actionIndex"):
            raise ValueError("PPO labels must be exact sampled and executed categorical action")
        grouped.setdefault(identity, []).append(row)
    for identity, episode in admitted.items():
        if episode["status"] == "excluded":
            if identity in grouped:
                raise ValueError("excluded episode cannot contribute gradients")
            continue
        rows = grouped.get(identity, [])
        if not rows or episode.get("recordsCount") != len(rows):
            raise ValueError("episode record count mismatch")
        last = rows[-1]
        terminal = episode["status"] == "terminated"
        if last.get("terminated") is not terminal or last.get("truncated") is not (not terminal) or last.get("segmentEnd") is not True:
            raise ValueError("episode final boundary mismatch")
        if terminal:
            if (episode.get("completedNaturally") is not True or episode.get("result") not in ("win", "loss", "draw")
                    or last.get("reward") != {"win": 1, "loss": -1, "draw": 0}[episode["result"]] or last.get("nextValue") != 0):
                raise ValueError("reward must match natural win, loss or draw only")
        else:
            if episode.get("bootstrapReliable") is not True or episode.get("result") is not None or last.get("reward") != 0:
                raise ValueError("truncation requires reliable visible bootstrap or explicit exclusion")
            next_record = episode.get("bootstrapRecord")
            value = trainer.values([next_record])[0]
            if not math.isclose(value, last.get("nextValue", math.inf), abs_tol=1e-4, rel_tol=0):
                raise ValueError("truncation bootstrap differs from actual parent critic")
    checked_hash(path, receipt["recordsSha256"])
    return records, receipt, sha256(root / "receipt.json")


def update_once(checkpoint, directory, output, *, catalog, device="cuda"):
    output = fresh_output(output)
    checkpoint = bc.private_checkpoint(checkpoint)
    trainer = ppo.EntityPPOTrainer.load(checkpoint, bc.catalog_metadata(catalog), device)
    frozen = source_hashes()
    if trainer.metadata.get("sourceHashes") != frozen:
        raise ValueError("parent source freeze mismatch")
    records, receipt, receipt_sha = admit_rollout(directory, checkpoint, trainer)
    parent_sha = sha256(checkpoint)
    metrics = trainer.update(records)
    if source_hashes() != frozen:
        raise ValueError("runtime changed during update")
    checked_hash(checkpoint, parent_sha)
    trainer.metadata = {**trainer.metadata, "parentCheckpointSha256": parent_sha,
                        "rolloutRecordsSha256": receipt["recordsSha256"], "rolloutReceiptSha256": receipt_sha,
                        "pins": receipt["pins"]}
    return save_verified(trainer, output, records[:64], {"schema": "entity-masked-ppo-update/1", **trainer.metadata,
                         "featureContract": bc.CONTRACT, "contractSha256": bc.CONTRACT_SHA256,
                         "ppoConfig": trainer.payload()["ppoConfig"], "metrics": metrics, "device": str(trainer.device)})


def serve(checkpoint, *, catalog, device="cpu", seed=1729):
    checkpoint = bc.private_checkpoint(checkpoint)
    trainer = ppo.EntityPPOTrainer.load(checkpoint, bc.catalog_metadata(catalog), device)
    if trainer.metadata.get("sourceHashes") != source_hashes():
        raise ValueError("service source freeze mismatch")
    trainer.generator.manual_seed(seed)
    identity = {"checkpoint": str(checkpoint), "checkpointSha256": sha256(checkpoint), "samplingSeed": seed,
                "frozenSelectionSha256": trainer.metadata["frozenSelectionSha256"]}
    while True:
        line = sys.stdin.readline(32 * 1024 * 1024 + 1)
        if not line:
            return
        request = None
        try:
            if len(line) > 32 * 1024 * 1024:
                raise ValueError("request byte budget exceeded")
            request = json.loads(line)
            if not isinstance(request, dict) or type(request.get("id")) is not int:
                raise ValueError("correlated integer request required")
            operation = request.get("op")
            if operation == "ping":
                result = {"schema": ppo.SCHEMA, "featureContract": bc.CONTRACT, "contractSha256": bc.CONTRACT_SHA256,
                          **trainer.catalog, **trainer.counts(), **identity, "device": str(trainer.device),
                          "sourceHashes": source_hashes(), "updateCount": trainer.update_count, "architecture": trainer.architecture}
            elif operation == "seed":
                value = request.get("seed")
                if type(value) is not int or not 0 <= value <= 0xffffffff:
                    raise ValueError("unsigned sampling seed required")
                trainer.generator.manual_seed(value); identity["samplingSeed"] = value
                result = {"samplingSeed": value}
            elif operation == "act":
                result = {"actions": trainer.act(request.get("records"), request.get("deterministic"))}
            elif operation == "values":
                result = {"values": trainer.values(request.get("records"))}
            elif operation == "close":
                result = {}
            else:
                raise ValueError("unsupported operation")
            response = {"id": request["id"], "ok": True, **result}
        except Exception as error:
            response = {"id": request.get("id") if isinstance(request, dict) else None,
                        "ok": False, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response, allow_nan=False), flush=True)
        if len(line) > 32 * 1024 * 1024 or response["ok"] and request["op"] == "close":
            return


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("bc_checkpoint"); init.add_argument("output")
    init.add_argument("--probe-records", required=True); init.add_argument("--bc-sha256", required=True)
    init.add_argument("--config"); init.add_argument("--seed", type=int, default=1729)
    init.add_argument("--selection-sha256")
    update = commands.add_parser("update")
    update.add_argument("checkpoint"); update.add_argument("rollout_directory"); update.add_argument("output")
    service = commands.add_parser("serve")
    service.add_argument("--checkpoint", required=True); service.add_argument("--seed", type=int, default=1729)
    for command, default in ((init, "cpu"), (update, "cuda"), (service, "cpu")):
        command.add_argument("--catalog", required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default=default)
    args = parser.parse_args()
    if args.command == "init":
        result = initialize(args.bc_checkpoint, args.output, catalog=args.catalog, probe_records=args.probe_records,
                            bc_sha256=args.bc_sha256, seed=args.seed, device=args.device, config=read_json(args.config) if args.config else None,
                            selection_sha256=args.selection_sha256)
    elif args.command == "update":
        result = update_once(args.checkpoint, args.rollout_directory, args.output, catalog=args.catalog, device=args.device)
    else:
        serve(args.checkpoint, catalog=args.catalog, device=args.device, seed=args.seed)
        return
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
