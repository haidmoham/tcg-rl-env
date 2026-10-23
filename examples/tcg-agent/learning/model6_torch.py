"""Model6 energy residual over the unchanged trusted Model5 entity actor."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from learning import entity_torch as m5
import torch
from torch import nn

CONTRACT = json.loads((Path(__file__).parent / "model6-contract.json").read_text(encoding="utf-8-sig"))
CONTRACT_SHA256 = hashlib.sha256(json.dumps(CONTRACT, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
SCHEMA = CONTRACT["schema"]
if CONTRACT.get("baseSchema") != m5.SCHEMA or CONTRACT.get("baseContractSha256") != m5.CONTRACT_SHA256:
    raise ValueError("Model6 must name the unchanged Model5 feature contract")
ENERGY_ENTITY_DIM = CONTRACT["energyEntityDimensions"]
ENERGY_CANDIDATE_DIM = CONTRACT["energyCandidateDimensions"]
ENERGY_ATTACK_DIM = CONTRACT["energyAttackDimensions"]
MAX_ATTACKS = CONTRACT["maxEnergyAttacks"]
MAX_CANDIDATE_ATTACKS = CONTRACT["maxEnergyCandidateAttacks"]
ARCHITECTURE_V1 = {"schema": "model6-energy-residual/1", "base": m5.ARCHITECTURE, "width": 64,
                "attackActionEmbedding": 16, "attackAggregation": "entity-bound-mean-with-candidate-action-gather/1",
                "afterAttachmentAggregation": "complete-candidate-bound-action-encoded-mean/1",
                "energyHead": [128, 64, 1], "initialResidual": "zero-last-layer/1", "criticContextDimensions": 272}
ARCHITECTURE_V2 = {**ARCHITECTURE_V1, "schema": "model6-energy-residual/2",
                   "entityContext": "energy-attacks-visible-entity-card/1", "entityEncoderInputDimensions": 243,
                   "candidateContext": "energy-afterattacks-visible-candidate-globals-actions/1",
                   "candidateEncoderInputDimensions": 305}
ARCHITECTURE = ARCHITECTURE_V2


def resolve_architecture(value=None):
    if value is None:
        return ARCHITECTURE
    for supported in (ARCHITECTURE_V1, ARCHITECTURE_V2):
        if value == supported:
            return supported
    raise ValueError("unsupported Model6 actor architecture")


CHECKPOINT_SCHEMA = "model6-energy-policy/1"
configure, model_root_for, private_checkpoint = m5.configure, m5.model_root_for, m5.private_checkpoint
catalog_metadata, supervised_loss = m5.catalog_metadata, m5.supervised_loss
BASE_KEYS = ("globals", "entities", "entity_mask", "entity_cards", "parents", "candidates", "candidate_mask",
             "action_ids", "origin_source", "origin_action", "sources", "targets", "actors")


def base_record(value):
    return {**value, "schema": m5.SCHEMA}


def record(value, metadata, labelled=False):
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("Model6 energy schema required")
    labels = m5.record(base_record(value), metadata, labelled)
    n, k = len(value["entities"]), len(value["candidates"])
    for key, count, dimension in (("energyEntities", n, ENERGY_ENTITY_DIM),
                                  ("energyCandidates", k, ENERGY_CANDIDATE_DIM)):
        rows = value.get(key)
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError("energy rows must cover every retained entity or candidate")
        for row in rows:
            m5.vector(row, dimension)
    attacks = value.get("energyAttacks")
    if not isinstance(attacks, list) or not 0 <= len(attacks) <= MAX_ATTACKS:
        raise ValueError("complete energy attack list required without truncation")
    for attack in attacks:
        m5.vector(attack, ENERGY_ATTACK_DIM)
    m5.indices(value.get("energyAttackEntities"), len(attacks), n)
    m5.indices(value.get("energyAttackActionIds"), len(attacks), metadata["actionVocabularySize"])
    after_attacks = value.get("energyCandidateAttacks")
    if not isinstance(after_attacks, list) or not 0 <= len(after_attacks) <= MAX_CANDIDATE_ATTACKS:
        raise ValueError("complete candidate after-attachment attack list required without truncation")
    for attack in after_attacks:
        m5.vector(attack, ENERGY_ATTACK_DIM)
    m5.indices(value.get("energyCandidateAttackCandidates"), len(after_attacks), k)
    m5.indices(value.get("energyCandidateAttackActionIds"), len(after_attacks), metadata["actionVocabularySize"])
    return labels


def padded_batch(values, device, metadata, labelled=False):
    if not isinstance(values, list) or not 1 <= len(values) <= 64:
        raise ValueError("bounded complete Model6 batch required")
    for value in values:
        record(value, metadata, labelled)
    batch, targets = m5.padded_batch([base_record(value) for value in values], device, metadata, labelled)
    b, n, k = len(values), batch["entities"].shape[1], batch["candidates"].shape[1]
    a = max(1, max(len(row["energyAttacks"]) for row in values))
    after_a = max(1, max(len(row["energyCandidateAttacks"]) for row in values))
    additions = {"energy_entities": torch.zeros((b, n, ENERGY_ENTITY_DIM)),
                 "energy_candidates": torch.zeros((b, k, ENERGY_CANDIDATE_DIM)),
                 "energy_attacks": torch.zeros((b, a, ENERGY_ATTACK_DIM)),
                 "energy_attack_entities": torch.full((b, a), -1, dtype=torch.long),
                 "energy_attack_actions": torch.zeros((b, a), dtype=torch.long),
                 "energy_attack_mask": torch.zeros((b, a), dtype=torch.bool),
                 "energy_candidate_attacks": torch.zeros((b, after_a, ENERGY_ATTACK_DIM)),
                 "energy_candidate_attack_candidates": torch.zeros((b, after_a), dtype=torch.long),
                 "energy_candidate_attack_actions": torch.zeros((b, after_a), dtype=torch.long),
                 "energy_candidate_attack_mask": torch.zeros((b, after_a), dtype=torch.bool)}
    for index, value in enumerate(values):
        count, choices, attacks = len(value["entities"]), len(value["candidates"]), len(value["energyAttacks"])
        if count:
            additions["energy_entities"][index, :count] = torch.tensor(value["energyEntities"], dtype=torch.float32)
        additions["energy_candidates"][index, :choices] = torch.tensor(value["energyCandidates"], dtype=torch.float32)
        if attacks:
            additions["energy_attacks"][index, :attacks] = torch.tensor(value["energyAttacks"], dtype=torch.float32)
            additions["energy_attack_entities"][index, :attacks] = torch.tensor(value["energyAttackEntities"])
            additions["energy_attack_actions"][index, :attacks] = torch.tensor(value["energyAttackActionIds"])
            additions["energy_attack_mask"][index, :attacks] = True
        after_attacks = len(value["energyCandidateAttacks"])
        if after_attacks:
            additions["energy_candidate_attacks"][index, :after_attacks] = torch.tensor(value["energyCandidateAttacks"], dtype=torch.float32)
            additions["energy_candidate_attack_candidates"][index, :after_attacks] = torch.tensor(value["energyCandidateAttackCandidates"])
            additions["energy_candidate_attack_actions"][index, :after_attacks] = torch.tensor(value["energyCandidateAttackActionIds"])
            additions["energy_candidate_attack_mask"][index, :after_attacks] = True
    batch.update({key: value.to(device) for key, value in additions.items()})
    return batch, targets


class EnergyEntityScorer(nn.Module):
    def __init__(self, metadata, architecture=None):
        super().__init__()
        self.architecture = resolve_architecture(architecture)
        self.contextual = self.architecture == ARCHITECTURE_V2
        self.base = m5.EntityScorer(metadata)
        self.attack_action_embedding = nn.Embedding(metadata["actionVocabularySize"], 16, padding_idx=0)
        self.attack_encoder = nn.Sequential(nn.Linear(ENERGY_ATTACK_DIM + 16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.energy_entity_encoder = nn.Sequential(nn.Linear(ENERGY_ENTITY_DIM + 64 + (m5.ENTITY_DIM + 16 if self.contextual else 0), 64), nn.ReLU())
        self.energy_parent_encoder = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.energy_candidate_encoder = nn.Sequential(nn.Linear(ENERGY_CANDIDATE_DIM + 64 + (m5.CANDIDATE_DIM + m5.GLOBAL_DIM + 32 if self.contextual else 0), 64), nn.ReLU())
        self.energy_head = nn.Sequential(nn.Linear(448, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.energy_head[-1].weight)
        nn.init.zeros_(self.energy_head[-1].bias)

    def base_batch(self, batch):
        return {key: batch[key] for key in BASE_KEYS}

    def energy_context(self, batch):
        attacks = self.attack_encoder(torch.cat((batch["energy_attacks"], self.attack_action_embedding(batch["energy_attack_actions"])), -1))
        attacks = attacks * batch["energy_attack_mask"].unsqueeze(-1)
        membership = torch.nn.functional.one_hot(batch["energy_attack_entities"].clamp_min(0), batch["entities"].shape[1]).to(attacks.dtype)
        membership = membership * batch["energy_attack_mask"].unsqueeze(-1)
        totals = membership.transpose(1, 2) @ attacks
        attack_summary = totals / membership.sum(1).unsqueeze(-1).clamp_min(1)
        entity_inputs = [batch["energy_entities"], attack_summary]
        if self.contextual:
            entity_inputs.extend((batch["entities"], self.base.card_embedding(batch["entity_cards"])))
        entities = self.energy_entity_encoder(torch.cat(entity_inputs, -1))
        entities = entities * batch["entity_mask"].unsqueeze(-1)
        entities = self.energy_parent_encoder(torch.cat((entities, m5.gather_entities(entities, batch["parents"])), -1))
        entities = entities * batch["entity_mask"].unsqueeze(-1)
        summary = entities.sum(1) / batch["entity_mask"].sum(1, keepdim=True).clamp_min(1)
        return entities, attacks, summary

    def visible_context(self, batch):
        a = self.base
        entities, mask = batch["entities"], batch["entity_mask"]
        b, n, _ = entities.shape
        encoded = a.entity_encoder(torch.cat((entities, a.card_embedding(batch["entity_cards"])), -1)) * mask.unsqueeze(-1)
        board = a.parent_encoder(torch.cat((encoded, m5.gather_entities(encoded, batch["parents"])), -1))
        q, key, value = [part.reshape(b, n, 2, 32).transpose(1, 2) for part in a.qkv(board).chunk(3, -1)]
        attention = ((q @ key.transpose(-1, -2)) / math.sqrt(32)).masked_fill(~mask[:, None, None, :], -1e9).softmax(-1)
        related = (attention @ value).transpose(1, 2).reshape(b, n, 64)
        board = a.relation_norm(board + a.relation_output(related)) * mask.unsqueeze(-1)
        summary = board.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        origin = m5.gather_entities(board, batch["origin_source"][:, None]).squeeze(1)
        _, _, energy_summary = self.energy_context(batch)
        return torch.cat((a.global_encoder(batch["globals"]), summary, origin, a.action_embedding(batch["origin_action"]), energy_summary), -1)

    def forward(self, **batch):
        base_logits = self.base(**self.base_batch(batch))
        entities, attacks, summary = self.energy_context(batch)
        origin = m5.gather_entities(entities, batch["origin_source"][:, None]).expand(-1, batch["candidates"].shape[1], -1)
        matches_entity = ((batch["energy_attack_entities"][:, None] == batch["sources"][:, :, None])
                          | (batch["energy_attack_entities"][:, None] == batch["actors"][:, :, None]))
        matches_action = batch["energy_attack_actions"][:, None] == batch["action_ids"][:, :, None]
        matches = (matches_entity & matches_action & batch["energy_attack_mask"][:, None]).to(attacks.dtype)
        candidate_attacks = (matches @ attacks) / matches.sum(-1, keepdim=True).clamp_min(1)
        after_attacks = self.attack_encoder(torch.cat((batch["energy_candidate_attacks"],
            self.attack_action_embedding(batch["energy_candidate_attack_actions"])), -1))
        after_attacks = after_attacks * batch["energy_candidate_attack_mask"].unsqueeze(-1)
        after_indices = batch["energy_candidate_attack_candidates"].unsqueeze(-1)
        after_totals = torch.zeros_like(batch["candidates"][..., :1]).expand(-1, -1, 64).clone()
        after_totals.scatter_add_(1, after_indices.expand(-1, -1, 64), after_attacks)
        after_counts = torch.zeros_like(batch["candidates"][..., :1])
        after_counts.scatter_add_(1, after_indices, batch["energy_candidate_attack_mask"].unsqueeze(-1).to(after_attacks.dtype))
        after_summary = after_totals / after_counts.clamp_min(1)
        candidate_inputs = [batch["energy_candidates"], after_summary]
        if self.contextual:
            candidate_inputs.extend((batch["candidates"], batch["globals"][:, None].expand(-1, batch["candidates"].shape[1], -1),
                                     self.base.action_embedding(batch["origin_action"])[:, None].expand(-1, batch["candidates"].shape[1], -1),
                                     self.base.action_embedding(batch["action_ids"])))
        candidate_context = self.energy_candidate_encoder(torch.cat(candidate_inputs, -1))
        inputs = torch.cat((m5.gather_entities(entities, batch["sources"]), m5.gather_entities(entities, batch["actors"]),
                            m5.gather_entities(entities, batch["targets"]), candidate_context,
                            summary[:, None].expand(-1, batch["candidates"].shape[1], -1), origin, candidate_attacks), -1)
        residual = self.energy_head(inputs).squeeze(-1)
        return (base_logits + residual).masked_fill(~batch["candidate_mask"], float("-inf"))

    def warm_start_model5(self, checkpoint, metadata, probe, device):
        baseline, _ = m5.load_checkpoint(checkpoint, device, metadata)
        self.base.load_state_dict(baseline.state_dict(), strict=True)
        batch, _ = padded_batch(probe, device, metadata)
        with torch.inference_mode():
            expected = baseline(**self.base_batch(batch))
            actual = self(**batch)
        if not torch.equal(expected, actual) or any(not torch.equal(value, self.base.state_dict()[name])
                                                  for name, value in baseline.state_dict().items()):
            raise ValueError("zero residual initialization changed Model5 actor state or actual-record logits")
        return {"baseModel5StateExact": True, "baseModel5LogitsExactAtInitialization": True, "probeRecords": len(probe)}


EntityScorer = EnergyEntityScorer


def load_checkpoint(path, device, metadata):
    saved = torch.load(private_checkpoint(path), map_location=device, weights_only=True)
    if (saved.get("schema") != CHECKPOINT_SCHEMA or saved.get("featureContract") != CONTRACT
            or saved.get("contractSha256") != CONTRACT_SHA256
            or saved.get("catalog") != metadata or not isinstance(saved.get("baseModel5CheckpointSha256"), str)):
        raise ValueError("Model6 checkpoint energy contract or baseline metadata mismatch")
    architecture = resolve_architecture(saved.get("architecture"))
    if saved.get("architecture") is None:
        raise ValueError("explicit saved Model6 architecture required")
    model = EnergyEntityScorer(metadata, architecture=architecture).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    if any(not torch.isfinite(p).all() or (p.abs() > 1e6).any() for p in model.parameters()):
        raise ValueError("invalid Model6 weights")
    return model.eval(), saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--catalog", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu"); parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    device = configure(args.device, args.seed); metadata = catalog_metadata(args.catalog)
    model, saved = load_checkpoint(args.checkpoint, device, metadata)
    for line in sys.stdin:
        request = None
        try:
            if len(line) > 32 * 1024 * 1024:
                raise ValueError("request exceeds byte budget")
            request = json.loads(line)
            if not isinstance(request, dict) or type(request.get("id")) is not int:
                raise ValueError("correlated request required")
            op = request.get("op")
            if op == "ping":
                result = {"schema": SCHEMA, "featureContract": CONTRACT, "contractSha256": CONTRACT_SHA256, **metadata,
                          "checkpoint": str(private_checkpoint(args.checkpoint)), "device": str(device),
                          "baseModel5CheckpointSha256": saved["baseModel5CheckpointSha256"],
                          "parameters": sum(p.numel() for p in model.parameters()), "architecture": model.architecture}
            elif op == "infer":
                batch, _ = padded_batch([request], device, metadata)
                with torch.inference_mode():
                    logits = model(**batch)[0, :len(request["candidates"])]
                if not torch.isfinite(logits).all():
                    raise ValueError("nonfinite candidate score")
                scores = logits.cpu().tolist(); ranking = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
                result = {"index": ranking[0], "ranking": ranking, "scores": scores}
            elif op == "close":
                print(json.dumps({"id": request["id"], "ok": True}), flush=True); break
            else:
                raise ValueError("unsupported operation")
            response = {"id": request["id"], "ok": True, **result}
        except Exception as error:
            response = {"id": request.get("id") if isinstance(request, dict) else None, "ok": False,
                        "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
