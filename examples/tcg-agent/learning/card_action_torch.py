"""Versioned actor-visible card/action candidate scoring."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn

CONTRACT = {"schema": "native-card-action-features/1", "stateDimensions": 224,
            "candidateDimensions": 191, "signature": "sha256-byte-centered/1",
            "pooling": "mean-card-signature/1", "selection": "uniform-chosen-card-target/1"}
CONTRACT_SHA256 = hashlib.sha256(json.dumps(CONTRACT, separators=(",", ":")).encode()).hexdigest()
SCHEMA = CONTRACT["schema"]
STATE_DIM, CANDIDATE_DIM = 224, 191
RUNS = Path(__file__).resolve().parents[1] / "runs"


def printing_aliases(value):
    if not isinstance(value, dict) or len(value) > 4096 or any(
            not isinstance(k, str) or not k or len(k) > 512
            or not isinstance(v, str) or not v or len(v) > 512 for k, v in value.items()):
        raise ValueError("bounded printing aliases required")
    for start in value:
        seen, token = set(), start
        while token in value:
            if token in seen:
                raise ValueError("printing aliases contain a cycle")
            seen.add(token)
            token = value[token]
    return value


def vector(value, size):
    if not isinstance(value, list) or len(value) != size or any(
            isinstance(x, bool) or not isinstance(x, (int, float))
            or not math.isfinite(x) or abs(x) > 1e6 for x in value):
        raise ValueError(f"expected {size} finite bounded numbers")
    return value


def record(value, labelled=False):
    if not isinstance(value, dict) or value.get("schema", SCHEMA) != SCHEMA:
        raise ValueError("card/action record schema mismatch")
    state = vector(value.get("state"), STATE_DIM)
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 512:
        raise ValueError("expected 1 through 512 legal candidates; no truncation")
    for candidate in candidates:
        vector(candidate, CANDIDATE_DIM)
    label = value.get("label")
    if labelled and (isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(candidates)):
        raise ValueError("label must index a legal candidate")
    labels = value.get("labels")
    if value.get("kind") == "selection":
        if labelled and (not isinstance(labels, list) or not labels or len(set(labels)) != len(labels)
                or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < len(candidates) for x in labels)
                or label != labels[0]):
            raise ValueError("selection labels must contain unique legal chosen indices")
    elif labels is not None:
        raise ValueError("multiple labels require an attributed selection record")
    return state, candidates, labels if labelled and value.get("kind") == "selection" else [label] if labelled else None


class CandidateScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(STATE_DIM + CANDIDATE_DIM, 128), nn.Tanh(),
                                     nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, states, candidates, mask):
        context = states[:, None, :].expand(-1, candidates.shape[1], -1)
        logits = self.network(torch.cat((context, candidates), -1)).squeeze(-1)
        return logits.masked_fill(~mask, float("-inf"))


def padded_batch(values, device, labelled=False):
    if not isinstance(values, list) or not 1 <= len(values) <= 64:
        raise ValueError("batch must contain 1 through 64 decisions")
    parsed = [record(value, labelled) for value in values]
    width = max(len(row[1]) for row in parsed)
    states = torch.tensor([row[0] for row in parsed], dtype=torch.float32)
    candidates = torch.zeros((len(parsed), width, CANDIDATE_DIM))
    mask = torch.zeros((len(parsed), width), dtype=torch.bool)
    for index, (_, choices, _) in enumerate(parsed):
        candidates[index, :len(choices)] = torch.tensor(choices, dtype=torch.float32)
        mask[index, :len(choices)] = True
    targets = torch.zeros((len(parsed), width)) if labelled else None
    if labelled:
        for index, row in enumerate(parsed):
            targets[index, row[2]] = 1 / len(row[2])
    return tuple(value.to(device) if value is not None else None for value in (states, candidates, mask, targets))


def supervised_loss(logits, mask, targets, reduction="mean"):
    log_probs = torch.nn.functional.log_softmax(logits, 1).masked_fill(~mask, 0)
    losses = -(log_probs * targets).sum(1)
    return losses.sum() if reduction == "sum" else losses.mean()


def model_root_for(value):
    """Return the approved root for a strict descendant, resolving path escapes."""
    roots = [RUNS.resolve()]
    declared = os.environ.get("PTCG_AGENT_MODEL_ROOT")
    if declared is not None:
        raw = Path(declared)
        root = raw.resolve()
        if not declared or not raw.is_absolute() or root.parent == root or root == RUNS.parent.resolve():
            raise ValueError("PTCG_AGENT_MODEL_ROOT must be an absolute dedicated model directory")
        roots.append(root)
    path = Path(value).resolve()
    for root in roots:
        if path != root and path.is_relative_to(root):
            return root
    raise ValueError("model path must be a strict descendant of repository runs or PTCG_AGENT_MODEL_ROOT")


def private_checkpoint(value):
    path = Path(value).resolve()
    model_root_for(path)
    if not Path(value).is_absolute() or path.suffix != ".pt":
        raise ValueError("checkpoint must be an absolute .pt path under an approved model root")
    return path


def configure(device, seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xffffffff:
        raise ValueError("bounded integer seed required")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CPU/CUDA runtime unavailable")
    return torch.device(device)


def load_checkpoint(path, device):
    path = private_checkpoint(path)
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get("featureContract") != CONTRACT or saved.get("contractSha256") != CONTRACT_SHA256:
        raise ValueError("checkpoint feature contract mismatch")
    printing_aliases(saved.get("printingAliases", {}))
    model = CandidateScorer().to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    if any(not torch.isfinite(p).all() or (p.abs() > 1e6).any() for p in model.parameters()):
        raise ValueError("invalid checkpoint weights")
    return model.eval(), saved


def infer(model, values, device):
    batch = padded_batch(values, device)
    with torch.inference_mode():
        logits = model(*batch[:3])
        if not torch.isfinite(logits[batch[2]]).all():
            raise ValueError("nonfinite legal candidate score")
        return logits.argmax(1).tolist()


def rank(model, values, device):
    batch = padded_batch(values, device)
    with torch.inference_mode():
        logits = model(*batch[:3])
        if not torch.isfinite(logits[batch[2]]).all():
            raise ValueError("nonfinite legal candidate score")
        scores = [row[:len(item["candidates"])].cpu().tolist() for row, item in zip(logits, values)]
    rankings = [sorted(range(len(row)), key=lambda index: (-row[index], index)) for row in scores]
    return {"indices": [row[0] for row in rankings], "rankings": rankings, "scores": scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    device = configure(args.device, args.seed)
    model, saved = load_checkpoint(args.checkpoint, device)
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or not isinstance(request.get("id"), int):
                raise ValueError("correlated request required")
            operation = request.get("op")
            if operation == "ping":
                result = {"schema": SCHEMA, "stateDim": STATE_DIM, "candidateDim": CANDIDATE_DIM,
                          "contractSha256": CONTRACT_SHA256, "printingAliases": saved.get("printingAliases", {}),
                          "checkpoint": str(private_checkpoint(args.checkpoint)), "device": str(device),
                          "parameters": sum(p.numel() for p in model.parameters()), "torch": torch.__version__}
            elif operation == "infer":
                ranked = rank(model, [request], device)
                result = {"index": ranked["indices"][0], "ranking": ranked["rankings"][0], "scores": ranked["scores"][0]}
            elif operation == "infer_batch":
                result = rank(model, request.get("items"), device)
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
