"""Model5 relational scoring over complete actor-visible entity decisions."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from learning.card_action_torch import configure, model_root_for, private_checkpoint
import torch
from torch import nn

CONTRACT = json.loads((Path(__file__).parent / "entity-contract.json").read_text(encoding="utf-8-sig"))
CONTRACT_SHA256 = hashlib.sha256(json.dumps(CONTRACT, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
SCHEMA = CONTRACT["schema"]
GLOBAL_DIM = CONTRACT["globalDimensions"]
ENTITY_DIM = CONTRACT["entityDimensions"]
CANDIDATE_DIM = CONTRACT["candidateDimensions"]
ARCHITECTURE = {"schema": "entity-relational-scorer/1", "width": 64, "cardEmbedding": 16,
                "actionEmbedding": 16, "attentionHeads": 2, "attentionLayers": 1,
                "head": [128, 64, 1], "explicitBindings": ["parent", "source", "actor", "target", "origin"]}


def catalog_metadata(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("catalog exceeds byte budget")
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict) or value.get("schema") != "model5-public-catalog/1":
        raise ValueError("Model5 private catalog required")
    sizes = {}
    for name in ("cards", "actions"):
        rows = value.get(name)
        if (not isinstance(rows, list) or not 1 <= len(rows) <= 100000
                or any(not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] != index
                       for index, row in enumerate(rows))):
            raise ValueError("catalog vocabulary must have contiguous ids including unknown zero")
        sizes[name] = len(rows)
    return {"catalogSha256": hashlib.sha256(raw).hexdigest(), "cardVocabularySize": sizes["cards"],
            "actionVocabularySize": sizes["actions"], "catalogSchema": value["schema"]}


def vector(value, size):
    if not isinstance(value, list) or len(value) != size or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or abs(x) > 1e6 for x in value):
        raise ValueError(f"expected {size} finite bounded numbers")
    return value


def indices(value, size, maximum, minimum=0):
    if not isinstance(value, list) or len(value) != size or any(
            type(index) is not int or not minimum <= index < maximum for index in value):
        raise ValueError("indices must cover the complete bounded list")
    return value


def record(value, metadata, labelled=False):
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("kind") not in ("ordinary", "selection"):
        raise ValueError("entity record schema or kind mismatch")
    vector(value.get("globals"), GLOBAL_DIM)
    entities, candidates = value.get("entities"), value.get("candidates")
    if not isinstance(entities, list) or not 0 <= len(entities) <= CONTRACT["maxEntities"]:
        raise ValueError("complete entity list exceeds contract")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= CONTRACT["maxCandidates"]:
        raise ValueError("complete candidate list exceeds contract")
    for row in entities:
        vector(row, ENTITY_DIM)
    for row in candidates:
        vector(row, CANDIDATE_DIM)
    n, k = len(entities), len(candidates)
    indices(value.get("entityCardIds"), n, metadata["cardVocabularySize"])
    parents = indices(value.get("entityParents"), n, n, -1)
    for index, parent in enumerate(parents):
        if parent == index:
            raise ValueError("entity cannot parent itself")
        seen = {index}
        while parent != -1:
            if parent in seen:
                raise ValueError("entity parent cycle")
            seen.add(parent)
            parent = parents[parent]
    for key in ("candidateSources", "candidateTargets", "candidateActors"):
        indices(value.get(key), k, n, -1)
    indices(value.get("candidateActionIds"), k, metadata["actionVocabularySize"])
    indices([value.get("originSourceIndex")], 1, n, -1)
    indices([value.get("originActionId")], 1, metadata["actionVocabularySize"])
    if not labelled:
        return None
    label, labels = value.get("label"), value.get("labels")
    if type(label) is not int or not 0 <= label < k:
        raise ValueError("label must index a legal candidate")
    if value["kind"] == "selection":
        if not isinstance(labels, list) or not labels or len(set(labels)) != len(labels):
            raise ValueError("selection needs unique chosen labels")
        indices(labels, len(labels), k)
        if label != labels[0]:
            raise ValueError("selection label must equal first chosen label")
        return labels
    if labels is not None:
        raise ValueError("multiple labels require selection kind")
    return [label]


def padded_batch(values, device, metadata, labelled=False):
    if not isinstance(values, list) or not 1 <= len(values) <= 64:
        raise ValueError("batch needs 1 through 64 decisions")
    labels = [record(value, metadata, labelled) for value in values]
    b, n, k = len(values), max(1, max(len(row["entities"]) for row in values)), max(len(row["candidates"]) for row in values)
    batch = {"globals": torch.tensor([row["globals"] for row in values], dtype=torch.float32),
             "entities": torch.zeros((b, n, ENTITY_DIM)), "entity_mask": torch.zeros((b, n), dtype=torch.bool),
             "entity_cards": torch.zeros((b, n), dtype=torch.long), "parents": torch.full((b, n), -1, dtype=torch.long),
             "candidates": torch.zeros((b, k, CANDIDATE_DIM)), "candidate_mask": torch.zeros((b, k), dtype=torch.bool),
             "action_ids": torch.zeros((b, k), dtype=torch.long),
             "origin_source": torch.tensor([row["originSourceIndex"] for row in values], dtype=torch.long),
             "origin_action": torch.tensor([row["originActionId"] for row in values], dtype=torch.long)}
    for key in ("sources", "targets", "actors"):
        batch[key] = torch.full((b, k), -1, dtype=torch.long)
    targets = torch.zeros((b, k)) if labelled else None
    for index, row in enumerate(values):
        count, choices = len(row["entities"]), len(row["candidates"])
        if count:
            batch["entities"][index, :count] = torch.tensor(row["entities"], dtype=torch.float32)
            batch["entity_cards"][index, :count] = torch.tensor(row["entityCardIds"])
            batch["parents"][index, :count] = torch.tensor(row["entityParents"])
            batch["entity_mask"][index, :count] = True
        batch["candidates"][index, :choices] = torch.tensor(row["candidates"], dtype=torch.float32)
        batch["candidate_mask"][index, :choices] = True
        batch["action_ids"][index, :choices] = torch.tensor(row["candidateActionIds"])
        for key in ("sources", "targets", "actors"):
            batch[key][index, :choices] = torch.tensor(row["candidate" + key.capitalize()])
        if labelled:
            targets[index, labels[index]] = 1 / len(labels[index])
    return {key: value.to(device) for key, value in batch.items()}, targets.to(device) if labelled else None


def gather_entities(entities, indices):
    gathered = entities.gather(1, indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, entities.shape[-1]))
    return gathered * (indices >= 0).unsqueeze(-1)


class EntityScorer(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        width = ARCHITECTURE["width"]
        self.card_embedding = nn.Embedding(metadata["cardVocabularySize"], 16, padding_idx=0)
        self.action_embedding = nn.Embedding(metadata["actionVocabularySize"], 16, padding_idx=0)
        self.entity_encoder = nn.Sequential(nn.Linear(ENTITY_DIM + 16, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.parent_encoder = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU())
        self.qkv = nn.Linear(width, width * 3)
        self.relation_output = nn.Linear(width, width)
        self.relation_norm = nn.LayerNorm(width)
        self.global_encoder = nn.Sequential(nn.Linear(GLOBAL_DIM, width), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(width * 6 + CANDIDATE_DIM + 32, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, globals, entities, entity_mask, entity_cards, parents, candidates, candidate_mask,
                action_ids, origin_source, origin_action, sources, targets, actors):
        b, n, _ = entities.shape
        encoded = self.entity_encoder(torch.cat((entities, self.card_embedding(entity_cards)), -1))
        encoded = encoded * entity_mask.unsqueeze(-1)
        board = self.parent_encoder(torch.cat((encoded, gather_entities(encoded, parents)), -1))
        q, key, value = [part.reshape(b, n, 2, 32).transpose(1, 2) for part in self.qkv(board).chunk(3, -1)]
        attention = (q @ key.transpose(-1, -2)) / math.sqrt(32)
        attention = attention.masked_fill(~entity_mask[:, None, None, :], -1e9).softmax(-1)
        related = (attention @ value).transpose(1, 2).reshape(b, n, 64)
        board = self.relation_norm(board + self.relation_output(related)) * entity_mask.unsqueeze(-1)
        summary = board.sum(1) / entity_mask.sum(1, keepdim=True).clamp_min(1)
        origin = gather_entities(board, origin_source[:, None]).squeeze(1)
        shared = torch.cat((self.global_encoder(globals), summary, origin, self.action_embedding(origin_action)), -1)
        shared = shared[:, None].expand(-1, candidates.shape[1], -1)
        inputs = torch.cat((shared, gather_entities(board, sources), gather_entities(board, actors),
                            gather_entities(board, targets), self.action_embedding(action_ids), candidates), -1)
        return self.head(inputs).squeeze(-1).masked_fill(~candidate_mask, float("-inf"))


def supervised_loss(logits, mask, targets, reduction="mean"):
    loss = -(torch.nn.functional.log_softmax(logits, 1).masked_fill(~mask, 0) * targets).sum(1)
    return loss.sum() if reduction == "sum" else loss.mean()


def load_checkpoint(path, device, metadata):
    path = private_checkpoint(path)
    saved = torch.load(path, map_location=device, weights_only=True)
    if (saved.get("featureContract") != CONTRACT or saved.get("contractSha256") != CONTRACT_SHA256
            or saved.get("architecture") != ARCHITECTURE or saved.get("catalog") != metadata
            or saved.get("initialization") not in ("random-local/1", "trusted-local-bc-warm-start/1")):
        raise ValueError("checkpoint entity, catalog or initialization contract mismatch")
    model = EntityScorer(metadata).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    if any(not torch.isfinite(p).all() or (p.abs() > 1e6).any() for p in model.parameters()):
        raise ValueError("invalid checkpoint weights")
    return model.eval(), saved


def rank(model, values, device, metadata):
    batch, _ = padded_batch(values, device, metadata)
    with torch.inference_mode():
        logits = model(**batch)
        if not torch.isfinite(logits[batch["candidate_mask"]]).all():
            raise ValueError("nonfinite legal candidate score")
        scores = [row[:len(item["candidates"])].cpu().tolist() for row, item in zip(logits, values)]
    rankings = [sorted(range(len(row)), key=lambda index: (-row[index], index)) for row in scores]
    return {"indices": [row[0] for row in rankings], "rankings": rankings, "scores": scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    device = configure(args.device, args.seed)
    metadata = catalog_metadata(args.catalog)
    model, _ = load_checkpoint(args.checkpoint, device, metadata)
    for line in sys.stdin:
        request = None
        try:
            if len(line) > 16 * 1024 * 1024:
                raise ValueError("request exceeds byte budget")
            request = json.loads(line)
            if not isinstance(request, dict) or type(request.get("id")) is not int:
                raise ValueError("correlated request required")
            operation = request.get("op")
            if operation == "ping":
                result = {"schema": SCHEMA, "featureContract": CONTRACT, "contractSha256": CONTRACT_SHA256,
                          **metadata, "architecture": ARCHITECTURE, "checkpoint": str(private_checkpoint(args.checkpoint)),
                          "device": str(device), "parameters": sum(p.numel() for p in model.parameters()), "torch": torch.__version__}
            elif operation == "infer":
                ranked = rank(model, [request], device, metadata)
                result = {"index": ranked["indices"][0], "ranking": ranked["rankings"][0], "scores": ranked["scores"][0]}
            elif operation == "infer_batch":
                result = rank(model, request.get("items"), device, metadata)
            elif operation == "close":
                print(json.dumps({"id": request["id"], "ok": True}), flush=True)
                break
            else:
                raise ValueError("unsupported operation")
            response = {"id": request["id"], "ok": True, **result}
        except Exception as error:
            response = {"id": request.get("id") if isinstance(request, dict) else None,
                        "ok": False, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
