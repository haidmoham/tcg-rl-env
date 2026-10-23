"""Bounded imitation learning over actor-visible native legal candidates."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn


SCHEMA = "native-ordinary-features/1"
STATE_DIM = 32
CANDIDATE_DIM = 27
MAX_RECORDS = 512
MAX_PADDED_CANDIDATES = 1_000_000
RUNS = Path(__file__).resolve().parents[1] / "runs"


def vector(value, size, name):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must have {size} numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) or abs(x) > 1e6 for x in value):
        raise ValueError(f"{name} must contain finite bounded numbers")
    return value


def record(value, require_label=False):
    if not isinstance(value, dict):
        raise ValueError("record must be an object")
    if value.get("schema", SCHEMA) != SCHEMA:
        raise ValueError("unsupported feature schema")
    state = vector(value.get("state"), STATE_DIM, "state")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("a nonempty legal candidate list is required")
    for candidate in candidates:
        vector(candidate, CANDIDATE_DIM, "candidate")
    label = value.get("label")
    if require_label and (isinstance(label, bool) or not isinstance(label, int)
                          or not 0 <= label < len(candidates)):
        raise ValueError("label must index an existing legal candidate")
    return state, candidates, label


class CandidateScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(STATE_DIM + CANDIDATE_DIM, 32),
                                     nn.Tanh(), nn.Linear(32, 1))

    def forward(self, states, candidates, mask):
        expanded = states[:, None, :].expand(-1, candidates.shape[1], -1)
        logits = self.network(torch.cat((expanded, candidates), dim=-1)).squeeze(-1)
        return logits.masked_fill(~mask, float("-inf"))


def padded_batch(values, device, require_label=False):
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_RECORDS:
        raise ValueError(f"batch must contain 1 through {MAX_RECORDS} records")
    parsed = [record(value, require_label) for value in values]
    width = max(len(item[1]) for item in parsed)
    if width * len(parsed) > MAX_PADDED_CANDIDATES:
        raise ValueError("padded candidate memory budget exceeded; no list was truncated")
    states = torch.tensor([item[0] for item in parsed], dtype=torch.float32, device=device)
    candidates = torch.zeros((len(parsed), width, CANDIDATE_DIM), device=device)
    mask = torch.zeros((len(parsed), width), dtype=torch.bool, device=device)
    for index, (_, choices, _) in enumerate(parsed):
        candidates[index, :len(choices)] = torch.tensor(choices, dtype=torch.float32,
                                                       device=device)
        mask[index, :len(choices)] = True
    labels = torch.tensor([item[2] for item in parsed], device=device) if require_label else None
    return states, candidates, mask, labels


def private_path(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("an absolute checkpoint path is required")
    path = Path(value).resolve()
    if not path.is_relative_to(RUNS.resolve()) or path.suffix != ".pt":
        raise ValueError("checkpoint must be a .pt file inside this repository's ignored runs directory")
    return path


class Learner:
    def __init__(self, device, seed, checkpoint=None):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but this PyTorch runtime has no available CUDA device")
        selected = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.device = torch.device(selected)
        self.seed = seed
        self.model = CandidateScorer().to(self.device)
        self.checkpoint = None
        self.trained = False
        if checkpoint:
            self.load(checkpoint)

    def load(self, path):
        path = private_path(path)
        saved = torch.load(path, map_location=self.device, weights_only=True)
        if saved.get("schema") != SCHEMA:
            raise ValueError("checkpoint feature schema mismatch")
        self.model.load_state_dict(saved["state_dict"], strict=True)
        if any(not torch.isfinite(parameter).all() for parameter in self.model.parameters()):
            raise ValueError("checkpoint contains nonfinite model parameters")
        self.checkpoint = str(path)

    def info(self):
        return {"schema": SCHEMA, "stateDim": STATE_DIM, "candidateDim": CANDIDATE_DIM,
                "device": str(self.device), "torch": torch.__version__,
                "cudaRuntime": torch.version.cuda,
                "deviceName": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "checkpoint": self.checkpoint, "objective": "legal-candidate supervised imitation"}

    def infer(self, items):
        batch = padded_batch(items, self.device)
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(*batch[:3])
            if not torch.isfinite(logits[batch[2]]).all():
                raise RuntimeError("nonfinite legal candidate score")
            indices = logits.argmax(dim=1).tolist()
        return indices

    def train(self, request):
        if self.trained:
            raise ValueError("this smoke process allows exactly one bounded training call")
        updates = request.get("updates", 10)
        if isinstance(updates, bool) or not isinstance(updates, int) or not 1 <= updates <= 10:
            raise ValueError("updates must be an integer from 1 through 10")
        checkpoint = private_path(request.get("checkpoint"))
        if checkpoint.exists():
            raise ValueError("checkpoint already exists; choose a fresh path")
        values = request.get("records")
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        batch = padded_batch(values, self.device, require_label=True)
        states, candidates, mask, labels = batch
        self.trained = True
        initial = [parameter.detach().clone() for parameter in self.model.parameters()]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)
        losses, gradients = [], []
        gpu_start = gpu_end = None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(self.device)
            gpu_start.record()
        started = time.perf_counter()
        self.model.train()
        for _ in range(updates):
            if time.perf_counter() - started >= 600:
                break
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(states, candidates, mask)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                   for parameter in self.model.parameters()):
                raise RuntimeError("missing or nonfinite gradient")
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in self.model.parameters()):
                raise RuntimeError("nonfinite model parameter")
            losses.append(float(loss.detach().cpu()))
            gradients.append(float(norm.detach().cpu()))
        if gpu_end is not None:
            gpu_end.record()
            torch.cuda.synchronize(self.device)
        wall = time.perf_counter() - started
        delta = sum(float((parameter.detach() - before).abs().sum().cpu())
                    for parameter, before in zip(self.model.parameters(), initial))
        if not losses or delta <= 0:
            raise RuntimeError("training did not change model weights")
        self.model.eval()
        with torch.inference_mode():
            before_reload = self.model(states, candidates, mask)
            predictions = before_reload.argmax(dim=1)
            agreement = float((predictions == labels).float().mean().cpu())
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        saved = {"schema": SCHEMA, "state_dict": self.model.state_dict(), "seed": self.seed,
                 "objective": "legal-candidate supervised imitation", "updates": len(losses),
                 "recordsSha256": hashlib.sha256(encoded).hexdigest()}
        with tempfile.NamedTemporaryFile(dir=checkpoint.parent, delete=False, suffix=".tmp") as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save(saved, temporary_path)
            os.replace(temporary_path, checkpoint)
        finally:
            temporary_path.unlink(missing_ok=True)
        reloaded = CandidateScorer().to(self.device)
        reloaded.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True)["state_dict"])
        reloaded.eval()
        with torch.inference_mode():
            after_reload = reloaded(states, candidates, mask)
        if not torch.equal(before_reload, after_reload):
            raise RuntimeError("checkpoint reload logits differ")
        self.checkpoint = str(checkpoint)
        return {"objective": saved["objective"], "records": len(values), "updates": len(losses),
                "losses": losses, "gradientNorms": gradients, "weightL1Change": delta,
                "trainAgreement": agreement, "trainWallSeconds": wall,
                "cudaEventMilliseconds": gpu_start.elapsed_time(gpu_end) if gpu_start else None,
                "peakCudaAllocatedBytes": torch.cuda.max_memory_allocated(self.device) if gpu_start else None,
                "paddedCandidateWidth": candidates.shape[1], "realCandidates": int(mask.sum().cpu()),
                "reloadLogitsExact": True, "checkpoint": str(checkpoint),
                "checkpointSha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "recordsSha256": saved["recordsSha256"], "device": str(self.device)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    learner = Learner(args.device, args.seed, args.checkpoint)
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or "id" not in request:
                raise ValueError("request must contain an id")
            operation = request.get("op", "infer")
            if operation == "ping":
                result = learner.info()
            elif operation == "infer":
                result = {"index": learner.infer([request])[0]}
            elif operation == "infer_batch":
                result = {"indices": learner.infer(request.get("items"))}
            elif operation == "train":
                result = learner.train(request)
            elif operation == "close":
                print(json.dumps({"id": request["id"], "ok": True}), flush=True)
                break
            else:
                raise ValueError("unknown operation")
            response = {"id": request["id"], "ok": True, **result}
        except Exception as error:
            response = {"id": request.get("id") if isinstance(request, dict) else None,
                        "ok": False, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
