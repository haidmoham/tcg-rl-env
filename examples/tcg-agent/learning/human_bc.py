"""Explicit ordinary human-label admission and exact PPO actor BC initialization."""

import hashlib
import json
from pathlib import Path
import torch
from learning import model6_torch as bc
from learning.entity_ppo import EntityActorCritic, finite_parameters

REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(value, catalog):
    path = Path(value).resolve()
    if path.is_dir():
        path /= "receipt.json"
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if (manifest.get("schema") != "model6-human-bc-dataset/1" or manifest.get("scope") != "localized-fit-only/1"
            or manifest.get("featureContract") != bc.CONTRACT or manifest.get("contractSha256") != bc.CONTRACT_SHA256
            or manifest.get("catalogSha256") != catalog["catalogSha256"] or manifest.get("validationRecords") != 0
            or manifest.get("eligibleForPpo") is not False or manifest.get("heldOutTransferClaim") is not False):
        raise ValueError("explicit human-only localized dataset required")
    for name, digest in manifest["sourceHashes"].items():
        source = (REPO / name).resolve()
        if not source.is_relative_to(REPO) or sha256(source) != digest:
            raise ValueError("human BC execution source changed")
    records_file = (path.parent / manifest["recordsPath"]).resolve()
    if not records_file.is_relative_to(path.parent) or sha256(records_file) != manifest["recordsSha256"]:
        raise ValueError("admitted human records changed")
    records = [json.loads(line) for line in records_file.read_text(encoding="utf-8").splitlines()]
    if not 1 <= len(records) <= 512 or len(records) != manifest["trainingRecords"]:
        raise ValueError("bounded one-game human collection required")
    decisions = set()
    for row in records:
        bc.record(row, catalog, labelled=True)
        label = row.get("humanLabel", {})
        if (row["kind"] != "ordinary" or label.get("actor") != "human" or label.get("gameId") != manifest["gameId"]
                or label.get("eligibleForPpo") is not False or label.get("behaviorLogProbability") is not None
                or label.get("decisionId") in decisions):
            raise ValueError("unique ordinary human labels required")
        decisions.add(label["decisionId"])
    provenance = {"datasetSha256": sha256(path), "sourceHashes": manifest["sourceHashes"],
                  "gameIds": [manifest["gameId"]], "seeds": [manifest["seed"]],
                  "trainingRecords": len(records), "validationRecords": 0,
                  "measurementScope": "fitted-examples-only; no held-out validation or transfer",
                  "baselineSha256": manifest["baselineSha256"], "admission": manifest}
    # Reuse the existing BC measurement/selection loop on fit examples, explicitly
    # labeled as such in its receipt. These are never described as validation.
    return records, records, provenance


def warm_start(checkpoint, metadata, records, device, expected_sha256):
    if sha256(checkpoint) != expected_sha256:
        raise ValueError("PPO parent digest mismatch")
    saved = torch.load(bc.private_checkpoint(str(checkpoint)), map_location=device, weights_only=True)
    if (saved.get("schema") != "entity-masked-ppo/1" or saved.get("updateCount") != 23
            or saved.get("featureContract") != bc.CONTRACT or saved.get("contractSha256") != bc.CONTRACT_SHA256
            or saved.get("catalog") != metadata):
        raise ValueError("exact update23 PPO contract/catalog required")
    architecture = bc.resolve_architecture(saved.get("architecture"))
    frozen_sources = saved.get("metadata", {}).get("sourceHashes", {})
    core = ["learning/model6_torch.py", "learning/entity_torch.py", "learning/card_action_torch.py",
            "learning/model6-contract.json", "learning/entity-contract.json", "learning/entity_ppo.py"]
    if any(sha256(REPO / name) != frozen_sources.get(name) for name in core):
        raise ValueError("frozen actor/contract code changed")
    frozen = EntityActorCritic(metadata, architecture=architecture).to(device).eval()
    frozen.load_state_dict(saved["state_dict"], strict=True)
    finite_parameters(frozen)
    model = bc.EnergyEntityScorer(metadata, architecture=architecture).to(device).eval()
    model.load_state_dict({name.removeprefix("actor."): tensor for name, tensor in saved["state_dict"].items()
                           if name.startswith("actor.")}, strict=True)
    if any(not torch.equal(tensor, frozen.actor.state_dict()[name]) for name, tensor in model.state_dict().items()):
        raise ValueError("PPO actor tensor changed on BC initialization")
    with torch.inference_mode():
        for start in range(0, len(records), 32):
            batch, _ = bc.padded_batch(records[start:start + 32], device, metadata)
            if not torch.equal(model(**batch), frozen.actor(**batch)):
                raise ValueError("PPO actor logits changed on human examples before fit")
    base_sha = saved["metadata"].get("baseModel5CheckpointSha256")
    if not isinstance(base_sha, str) or len(base_sha) != 64:
        raise ValueError("original Model5 lineage missing")
    return model, {"initialization": "exact-frozen-update23-actor/1", "warmStartActorStateExact": True,
                   "warmStartActorLogitsExact": True, "warmStartProbeRecords": len(records),
                   "baseModel5CheckpointSha256": base_sha, "parentPpoCheckpointSha256": expected_sha256,
                   "parentPpoUpdateCount": saved["updateCount"], "criticUsedOrModified": False,
                   "ppoOptimizerUsedOrModified": False, "frozenActorSourceGuardPassed": True}
