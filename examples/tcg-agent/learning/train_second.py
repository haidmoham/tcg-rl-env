"""Bounded second-model imitation with whole-game validation."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from learning.native_torch import CANDIDATE_DIM, CandidateScorer, RUNS, SCHEMA, STATE_DIM, record
import torch

TRAIN_SEEDS = tuple(range(91001, 91019))
VALIDATION_SEEDS = tuple(range(91019, 91025))
MAX_RECORDS = 10000
MAX_PADDED_ROWS = 1000000


def validate_plan():
    path = REPO / "learning/model-two-plan.json"
    plan = json.loads(path.read_text())
    collection, training = plan.get("collection", {}), plan.get("training", {})
    if (plan.get("schema") != "native-second-model-plan/1" or plan.get("featureSchema") != SCHEMA
            or plan.get("modelParameters") != 1953
            or collection.get("trainingSeeds") != list(TRAIN_SEEDS)
            or collection.get("validationSeeds") != list(VALIDATION_SEEDS)
            or collection.get("minimumCompleteTrainingGames") != 8
            or collection.get("minimumCompleteValidationGames") != 2
            or collection.get("minimumTrainingRecordsExclusive") != 125
            or training.get("defaultMaximumUpdates") != 200
            or training.get("hardMaximumUpdates") != 1000
            or training.get("maximumUpdateWallSeconds") != 120
            or training.get("minibatchSize") != 64
            or training.get("validationEveryUpdates") != 10
            or training.get("patienceChecks") != 3
            or training.get("minimumValidationLossImprovement") != 0.0001
            or training.get("selectionUsesFinalEvaluation") is not False):
        raise ValueError("second-model trainer disagrees with the preregistered plan")
    return sha256(path)


@dataclass(frozen=True)
class Collection:
    training: list
    validation: list
    provenance: dict


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("collection artifact must use a relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("collection artifact escapes its directory or is missing")
    return path


def digest_map(value, name):
    if not isinstance(value, dict) or not value or any(
            not isinstance(key, str) or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None for key, digest in value.items()):
        raise ValueError(f"invalid {name}")
    return value


def load_collection(directory):
    plan_hash = validate_plan()
    root = Path(directory).resolve()
    manifest_path = root / "receipt.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "native-supervised-collection/1"
            or manifest.get("complete") is not True or manifest.get("fatal") is not False
            or manifest.get("fatalFailures") or manifest.get("cancelled")):
        raise ValueError("collection is incomplete, fatal, or cancelled")
    pins = digest_map(manifest.get("pins"), "native pins")
    if not {"workerSha256", "executableSha256", "manifestSha256", "policySha256"}.issubset(pins):
        raise ValueError("required native pins are missing")
    sources = digest_map(manifest.get("sourceHashes"), "source hashes")
    if sources.get("learning/model-two-plan.json") != plan_hash:
        raise ValueError("collection source plan hash differs from the admitted training plan")
    expected = {seed: {"id": f"seed-{seed}", "seed": seed,
                       "seatOrder": [0, 1] if (seed - 91001) % 2 == 0 else [1, 0],
                       "split": "train" if seed in TRAIN_SEEDS else "validation"}
                for seed in (*TRAIN_SEEDS, *VALIDATION_SEEDS)}
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 24:
        raise ValueError("the fixed 24-game collection plan is required")
    seen = set()
    for task in tasks:
        if not isinstance(task, dict) or task.get("seed") not in expected:
            raise ValueError("collection task is outside the fixed seed plan")
        seed = task["seed"]
        if seed in seen or any(task.get(key) != value for key, value in expected[seed].items()):
            raise ValueError("duplicate or changed collection task")
        seen.add(seed)
    completed = manifest.get("completed")
    capacity = manifest.get("capacityFailures", [])
    unattempted = manifest.get("unattempted", [])
    if any(not isinstance(items, list) for items in (completed, capacity, unattempted)):
        raise ValueError("explicit collection outcomes are required")
    seen = set()
    for items, status in ((completed, "completed"), (capacity, "capacity-timeout"),
                          (unattempted, "unattempted")):
        for item in items:
            if not isinstance(item, dict) or item.get("seed") not in expected:
                raise ValueError("collection outcome is outside the fixed seed plan")
            seed = item["seed"]
            if seed in seen or any(item.get(key) != value for key, value in expected[seed].items()):
                raise ValueError("duplicate, overlapping, or changed collection outcome")
            if item.get("status") != status:
                raise ValueError("collection outcome status mismatch")
            if status == "capacity-timeout" and (item.get("reason"), item.get("code")) not in {
                    ("episode-wall-budget", "NATIVE_EPISODE_TIME_BUDGET"),
                    ("global-wall-budget", "GLOBAL_COLLECTION_TIME_BUDGET")}:
                raise ValueError("only explicit wall-budget capacity failures may be excluded")
            if status == "unattempted" and item.get("reason") != "global-wall-budget":
                raise ValueError("unattempted games require the global wall budget")
            seen.add(seed)
    if seen != set(expected):
        raise ValueError("collection outcomes do not account for all planned games")
    training, validation, games = [], [], []
    for item in completed:
        seed = item["seed"]
        record_path = artifact(root, item.get("recordsPath"))
        receipt_path = artifact(root, item.get("receiptPath"))
        receipt = json.loads(receipt_path.read_text())
        if sha256(receipt_path) != item.get("receiptSha256"):
            raise ValueError("game receipt hash mismatch")
        if sha256(record_path) != item.get("recordsFileSha256"):
            raise ValueError("game record file hash mismatch")
        if (receipt.get("seed") != seed or receipt.get("seatOrder") != item["seatOrder"]
                or receipt.get("completedNaturally") is not True or receipt.get("rejected") != 0
                or receipt.get("realHandExchanges", 0) < 1 or receipt.get("privacyChecks", 0) < 1
                or receipt.get("pins") != pins or receipt.get("sourceHashes") != sources
                or receipt.get("featureContract") != {"schema": SCHEMA,
                    "stateDimensions": STATE_DIM, "candidateDimensions": CANDIDATE_DIM}):
            raise ValueError("game lacks matching native, source, legality, or privacy gates")
        if any(receipt.get(key) != item.get(key) for key in
               ("ordinaryDecisions", "commands", "terminalStateHash", "trajectorySha256")):
            raise ValueError("game receipt and collection item disagree")
        lines = [line for line in record_path.read_text().splitlines() if line.strip()]
        canonical = hashlib.sha256(("[" + ",".join(lines) + "]").encode()).hexdigest()
        if canonical != item.get("recordsSha256") or canonical != receipt.get("recordsSha256"):
            raise ValueError("canonical game record hash mismatch")
        values = [json.loads(line) for line in lines]
        if not values or len(values) != receipt.get("ordinaryDecisions"):
            raise ValueError("game record count mismatch")
        for value in values:
            record(value, require_label=True)
        (training if item["split"] == "train" else validation).extend(values)
        games.append({**expected[seed], "records": len(values), "recordsSha256": canonical,
                      "receiptSha256": item["receiptSha256"]})
    train_games = sum(game["split"] == "train" for game in games)
    val_games = len(games) - train_games
    if train_games < 8 or val_games < 2 or len(training) <= 125:
        raise ValueError("second model requires at least 8 train games, 2 validation games, and more than 125 train labels")
    if len(training) + len(validation) > MAX_RECORDS:
        raise ValueError("dataset exceeds 10000 records; no rows were dropped")
    return Collection(training, validation, {"collectionSha256": sha256(manifest_path), "planSha256": plan_hash,
        "pins": pins, "sourceHashes": sources, "games": sorted(games, key=lambda value: value["seed"]),
        "trainingGames": train_games, "validationGames": val_games,
        "trainingRecords": len(training), "validationRecords": len(validation),
        "capacityFailures": capacity, "unattempted": unattempted,
        "reservedFinalEvaluationSeeds": [92001, 92002]})


def resident_dataset(values, device):
    width = max(len(value["candidates"]) for value in values)
    if len(values) > MAX_RECORDS or len(values) * width > MAX_PADDED_ROWS:
        raise ValueError("resident padding budget exceeded; no legal candidates were truncated")
    states = torch.tensor([value["state"] for value in values], dtype=torch.float32)
    candidates = torch.zeros((len(values), width, CANDIDATE_DIM))
    mask = torch.zeros((len(values), width), dtype=torch.bool)
    for index, value in enumerate(values):
        count = len(value["candidates"])
        candidates[index, :count] = torch.tensor(value["candidates"], dtype=torch.float32)
        mask[index, :count] = True
    labels = torch.tensor([value["label"] for value in values], dtype=torch.long)
    return tuple(value.to(device) for value in (states, candidates, mask, labels))


def evaluate(model, dataset):
    total_loss, correct = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset[0]), 64):
            state, candidates, mask, labels = (value[start:start + 64] for value in dataset)
            logits = model(state, candidates, mask)
            if not torch.isfinite(logits[mask]).all():
                raise RuntimeError("nonfinite validation candidate score")
            total_loss += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum").cpu())
            correct += int((logits.argmax(dim=1) == labels).sum().cpu())
    count = len(dataset[0])
    return {"loss": total_loss / count, "teacherAgreement": correct / count, "records": count}


def atomic_checkpoint(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        torch.save(value, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def fit(collection, output, *, device="cuda", updates=200, max_seconds=120, seed=1729):
    if seed != 1729:
        raise ValueError("the preregistered fresh initialization seed is 1729")
    if isinstance(updates, bool) or not isinstance(updates, int) or not 1 <= updates <= 1000:
        raise ValueError("updates must be an integer from 1 through 1000")
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 120:
        raise ValueError("GPU update budget must be positive and at most 120 seconds")
    output = Path(output).resolve()
    if not output.is_relative_to(RUNS.resolve()):
        raise ValueError("model2 output must remain in ignored runs")
    output.mkdir(parents=True, exist_ok=False)
    if device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested training device is unavailable")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    training = resident_dataset(collection.training, device)
    validation = resident_dataset(collection.validation, device)
    model = CandidateScorer().to(device)
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    baseline = evaluate(model, validation)
    startup_seconds = time.perf_counter() - started
    generator = torch.Generator().manual_seed(seed)
    order, offset, epochs, examples = None, 0, 0, 0
    history, losses, gradient_norms = [], [], []
    best_loss, significant_best, stale = math.inf, math.inf, 0
    best_update = None
    checkpoint = output / "best-model.pt"
    gpu_start = gpu_end = None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        gpu_start.record()
    loop_started = time.perf_counter()
    stop_reason = "update-limit"
    for update in range(1, updates + 1):
        if time.perf_counter() - loop_started >= max_seconds:
            stop_reason = "wall-budget"
            break
        if order is None or offset == len(order):
            order = torch.randperm(len(training[0]), generator=generator).to(device)
            offset = 0
            epochs += 1
        indices = order[offset:offset + 64]
        offset += len(indices)
        examples += len(indices)
        state, candidates, mask, labels = (value[indices] for value in training)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(state, candidates, mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        if any(value.grad is None or not torch.isfinite(value.grad).all() for value in model.parameters()):
            raise RuntimeError("missing or nonfinite gradient")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if any(not torch.isfinite(value).all() for value in model.parameters()):
            raise RuntimeError("nonfinite model parameter")
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(norm.detach().cpu()))
        if update % 10 == 0 or update == updates or time.perf_counter() - loop_started >= max_seconds:
            metrics = evaluate(model, validation)
            history.append({"update": update, **metrics})
            if metrics["loss"] < best_loss:
                best_loss, best_update = metrics["loss"], update
                atomic_checkpoint(checkpoint, {"schema": SCHEMA, "state_dict": model.state_dict(),
                    "seed": seed, "objective": "legal-candidate supervised imitation",
                    "modelGeneration": 2, "initialization": "fresh", "updates": update,
                    "validationLoss": best_loss, "collectionSha256": collection.provenance["collectionSha256"]})
            if metrics["loss"] < significant_best - 0.0001:
                significant_best, stale = metrics["loss"], 0
            else:
                stale += 1
            if stale >= 3:
                stop_reason = "validation-patience"
                break
    if not losses:
        raise RuntimeError("wall budget expired before a training update")
    if best_update is None or history[-1]["update"] != len(losses):
        metrics = evaluate(model, validation)
        history.append({"update": len(losses), **metrics})
        if metrics["loss"] < best_loss:
            best_loss, best_update = metrics["loss"], len(losses)
            atomic_checkpoint(checkpoint, {"schema": SCHEMA, "state_dict": model.state_dict(),
                "seed": seed, "objective": "legal-candidate supervised imitation",
                "modelGeneration": 2, "initialization": "fresh", "updates": best_update,
                "validationLoss": best_loss, "collectionSha256": collection.provenance["collectionSha256"]})
    if gpu_end is not None:
        gpu_end.record()
        torch.cuda.synchronize()
    update_wall = time.perf_counter() - loop_started
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    weight_delta = sum(float((value.detach().cpu() - initial[key]).abs().sum())
                       for key, value in model.state_dict().items())
    if weight_delta <= 0:
        raise RuntimeError("best trained checkpoint did not change weights")
    reloaded = CandidateScorer().to(device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    reloaded.eval()
    with torch.inference_mode():
        for start in range(0, len(validation[0]), 64):
            batch = tuple(value[start:start + 64] for value in validation[:3])
            if not torch.equal(model(*batch), reloaded(*batch)):
                raise RuntimeError("best checkpoint reload logits differ")
    result = {"schema": "native-second-imitation/1", "initialization": "fresh", "seed": seed,
        "parameters": sum(value.numel() for value in model.parameters()), "device": device,
        "torch": torch.__version__, "cudaRuntime": torch.version.cuda,
        "deviceName": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "objective": "legal-candidate supervised imitation", "provenance": collection.provenance,
        "limits": {"requestedUpdates": updates, "hardUpdateLimit": 1000, "maxSeconds": max_seconds,
                   "minibatch": 64, "learningRate": 0.003, "validationEveryUpdates": 10,
                   "patience": 3, "minDelta": 0.0001},
        "startupAndUploadSeconds": startup_seconds, "updateWallSeconds": update_wall,
        "completedUpdates": len(losses), "examplesProcessed": examples, "epochsStarted": epochs,
        "shuffle": "seeded without replacement within each epoch; last minibatch can be partial",
        "validationWeighting": "decision-weighted across held-out whole games",
        "stopReason": stop_reason, "initialValidation": baseline, "validationHistory": history,
        "bestUpdate": best_update, "bestValidation": evaluate(model, validation),
        "bestTraining": evaluate(model, training), "losses": losses, "gradientNorms": gradient_norms,
        "weightL1ChangeFromInitialization": weight_delta, "reloadLogitsExact": True,
        "cudaEventMilliseconds": gpu_start.elapsed_time(gpu_end) if gpu_start else None,
        "peakCudaAllocatedBytes": torch.cuda.max_memory_allocated() if gpu_start else None,
        "paddedWidths": {"train": training[1].shape[1], "validation": validation[1].shape[1]},
        "checkpoint": str(checkpoint), "checkpointSha256": sha256(checkpoint),
        "claim": "bounded second imitation model; teacher agreement is not playing strength"}
    (output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "model-config.json").write_text(json.dumps({"executable": sys.executable,
        "checkpoint": str(checkpoint), "seed": seed, "device": device}, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("collection", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    collection = load_collection(args.collection)
    print(json.dumps(fit(collection, args.output, device=args.device, updates=args.updates,
                         max_seconds=args.max_seconds), allow_nan=False))


if __name__ == "__main__":
    main()
