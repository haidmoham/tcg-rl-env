"""Train one bounded imitation model from three gated games and hold out a fourth."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import_started = time.perf_counter()
from learning.native_torch import CANDIDATE_DIM, Learner, SCHEMA, STATE_DIM, padded_batch, record
import torch
module_import_seconds = time.perf_counter() - import_started


def read_game(root, seed):
    paths = list(root.rglob(f"seed-{seed}-records.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected exactly one copy of seed {seed}, got {len(paths)}")
    path = paths[0]
    receipt_path = path.with_name(f"seed-{seed}-receipt.json")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("seed") != seed or receipt.get("completedNaturally") is not True or receipt.get("rejected") != 0
            or receipt.get("realHandExchanges", 0) < 1 or receipt.get("privacyChecks", 0) < 1
            or receipt.get("featureContract", {}).get("schema") != SCHEMA
            or receipt.get("featureContract", {}).get("stateDimensions") != STATE_DIM
            or receipt.get("featureContract", {}).get("candidateDimensions") != CANDIDATE_DIM):
        raise ValueError(f"seed {seed} lacks complete native legality/privacy gate")
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    canonical_hash = hashlib.sha256(("[" + ",".join(lines) + "]").encode()).hexdigest()
    if canonical_hash != receipt.get("recordsSha256"):
        raise ValueError(f"seed {seed} canonical record hash mismatch")
    values = [json.loads(line) for line in lines]
    for value in values:
        record(value, require_label=True)
    if len(values) != receipt.get("ordinaryDecisions") or not values:
        raise ValueError(f"seed {seed} decision count mismatch")
    provenance = {"seed": seed, "records": len(values), "nativeCommands": receipt.get("commands"),
                  "recordsFileSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "canonicalRecordsSha256": canonical_hash, "numericRecordsValidated": True,
                  "receiptSha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                  "realHandExchanges": receipt["realHandExchanges"],
                  "privacyChecks": receipt["privacyChecks"]}
    return values, provenance


def sample_gpu():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                                    "--format=csv,noheader"], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-seeds", nargs=3, type=int, default=[90001, 90002, 90003])
    parser.add_argument("--heldout-seed", type=int, default=90004)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if not output.is_relative_to((REPO / "runs").resolve()):
        raise ValueError("output must remain in ignored runs")
    output.mkdir(parents=True, exist_ok=False)
    seeds = [*args.train_seeds, args.heldout_seed]
    if len(set(seeds)) != 4 or any(not 0 <= seed <= 0xffffffff for seed in seeds):
        raise ValueError("four distinct bounded seeds required for whole-game split")
    games = [read_game(dataset, seed) for seed in seeds]
    training = [value for values, _ in games[:3] for value in values]
    heldout = games[3][0]
    if len(training) > 512:
        raise ValueError("whole-game training set exceeds 512-record budget; no records dropped")
    before_gpu = sample_gpu()
    started = time.perf_counter()
    learner = Learner("cuda", 1729)
    startup_seconds = time.perf_counter() - started
    checkpoint = output / "model.pt"
    result = learner.train({"records": training, "updates": 10, "checkpoint": str(checkpoint)})
    states, candidates, mask, labels = padded_batch(heldout, learner.device, require_label=True)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = learner.model(states, candidates, mask)
        predictions = logits.argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("held-out loss is not finite")
        if not mask.gather(1, predictions[:, None]).all():
            raise ValueError("held-out prediction selected padding")
        agreement = float((predictions == labels).float().mean().cpu())
    torch.cuda.synchronize()
    heldout_seconds = time.perf_counter() - started
    receipt = {"schema": "native-imitation-smoke/1", "featureSchema": SCHEMA,
               "python": sys.executable, "runtime": learner.info(),
               "moduleImportSeconds": module_import_seconds, "modelStartupSeconds": startup_seconds,
               "trainingGames": [provenance for _, provenance in games[:3]],
               "heldoutGame": games[3][1], "training": result,
               "heldout": {"records": len(heldout), "loss": float(loss.cpu()),
                           "teacherAgreement": agreement, "inferenceSeconds": heldout_seconds,
                           "allPredictionsLegal": True, "wholeGameSplit": True},
               "wholeDeviceNvidiaSmiBefore": before_gpu, "wholeDeviceNvidiaSmiAfter": sample_gpu(),
               "utilizationCaveat": "whole-device samples include unrelated work; CUDA events measure loop timeline, not kernel utilization",
               "claim": "bounded supervised imitation pipeline smoke; no policy improvement or reinforcement-learning claim"}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    config = {"executable": sys.executable, "checkpoint": str(checkpoint), "seed": 1729, "device": "cuda"}
    (output / "model-config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
