"""Bounded card/action behavioral cloning with whole-game validation."""

import argparse
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
from learning.card_action_torch import (CONTRACT, CONTRACT_SHA256, RUNS, CandidateScorer,
    configure, load_checkpoint, model_root_for, padded_batch, printing_aliases, record, supervised_loss)
import torch


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(root, value):
    if not isinstance(value, str) or Path(value).is_absolute():
        raise ValueError("relative artifact path required")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("artifact escaped collection or is missing")
    return path


def digest_map(value):
    if not isinstance(value, dict) or not value or any(
            not isinstance(k, str) or not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{64}", v)
            for k, v in value.items()):
        raise ValueError("nonempty SHA256 provenance map required")
    return value


def load_dataset(value):
    path = Path(value).resolve()
    if path.is_dir():
        path /= "receipt.json"
    root = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if (manifest.get("schema") != "native-card-action-dataset/1" or manifest.get("complete") is not True
            or manifest.get("fatal") is not False or manifest.get("featureContract") != CONTRACT):
        raise ValueError("complete nonfatal card/action dataset with matching features required")
    sources, pins = digest_map(manifest.get("sourceHashes")), digest_map(manifest.get("pins"))
    for name, expected in sources.items():
        if sha256(artifact(REPO, name)) != expected:
            raise ValueError("collection source changed after admission")
    aliases = printing_aliases(manifest.get("printingAliases", {}))
    games = manifest.get("games")
    if not isinstance(games, list) or not 2 <= len(games) <= 256:
        raise ValueError("bounded whole-game dataset required")
    training, validation, seeds, ids, file_paths = [], [], set(), set(), set()
    dataset_bytes, feature_numbers = 0, 0
    split_games = {"train": 0, "validation": 0}
    for game in games:
        if not isinstance(game, dict) or game.get("split") not in split_games:
            raise ValueError("whole game must belong to train or validation")
        seed, identity = game.get("seed"), game.get("id")
        if (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xffffffff
                or not isinstance(identity, str) or not identity or seed in seeds or identity in ids):
            raise ValueError("duplicate, overlapping, or invalid game identity")
        seeds.add(seed)
        ids.add(identity)
        commands = game.get("commands")
        if (game.get("completedNaturally") is not True or game.get("rejected") != 0
                or isinstance(commands, bool) or not isinstance(commands, int) or commands <= 0
                or game.get("privacyChecks") != commands * 2 or game.get("realHandExchanges", 0) <= 0):
            raise ValueError("game did not pass natural completion, legal action and privacy gates")
        records_path = artifact(root, game.get("recordsPath"))
        dataset_bytes += records_path.stat().st_size
        if dataset_bytes > 512 * 1024 * 1024:
            raise ValueError("dataset raw bytes exceed the bounded CPU memory budget")
        if records_path in file_paths or sha256(records_path) != game.get("recordsSha256"):
            raise ValueError("record file duplicate or raw-byte SHA256 mismatch")
        file_paths.add(records_path)
        values = [json.loads(line) for line in records_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not values:
            raise ValueError("game contains no ordinary decisions")
        for item in values:
            record(item, labelled=True)
            feature_numbers += 224 + len(item["candidates"]) * 191
            if feature_numbers > 30000000:
                raise ValueError("dataset feature numbers exceed the bounded CPU memory budget")
        (training if game["split"] == "train" else validation).extend(values)
        split_games[game["split"]] += 1
    if not training or not validation or len(training) + len(validation) > 25000:
        raise ValueError("both disjoint splits and at most 25000 decisions required")
    return training, validation, aliases, {"datasetSha256": sha256(path), "sourceHashes": sources, "pins": pins,
        "trainingRecords": len(training), "validationRecords": len(validation), "games": split_games,
        "gameIds": sorted(ids), "seeds": sorted(seeds)}


def evaluate(model, values, device):
    total_loss, correct = 0.0, 0
    ordinary_count, ordinary_correct, selection_count, selection_correct, selection_exact = 0, 0, 0, 0, 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), 64):
            batch = padded_batch(values[start:start + 64], device, labelled=True)
            logits = model(*batch[:3])
            loss = supervised_loss(logits, batch[2], batch[3], reduction="sum")
            if not torch.isfinite(loss) or not torch.isfinite(logits[batch[2]]).all():
                raise ValueError("nonfinite evaluation loss or score")
            total_loss += float(loss.cpu())
            predictions = logits.argmax(1).tolist()
            for index, (item, prediction) in enumerate(zip(values[start:start + 64], predictions)):
                if item.get("kind") == "selection":
                    selection_count += 1
                    chosen = set(item["labels"])
                    hit = prediction in chosen
                    selection_correct += int(hit)
                    scores = logits[index, :len(item["candidates"])].cpu().tolist()
                    ranking = sorted(range(len(scores)), key=lambda candidate: (-scores[candidate], candidate))
                    selection_exact += int(set(ranking[:len(chosen)]) == chosen)
                else:
                    ordinary_count += 1
                    hit = prediction == item["label"]
                    ordinary_correct += int(hit)
                correct += int(hit)
    return {"records": len(values), "loss": total_loss / len(values), "teacherAgreement": correct / len(values),
        "ordinary": {"records": ordinary_count, "top1Agreement": ordinary_correct / ordinary_count if ordinary_count else None},
        "selection": {"records": selection_count,
            "top1InChosenSet": selection_correct / selection_count if selection_count else None,
            "exactChosenSetAgreement": selection_exact / selection_count if selection_count else None,
            "exactSetMetricCardinality": "teacher chosen count; not a learned or deployed count prediction"}}


def atomic_checkpoint(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_verified_checkpoint(path, value, model, probe, device):
    model.eval()
    with torch.inference_mode():
        before = model(*probe[:3]).detach().clone()
    atomic_checkpoint(path, value)
    reloaded, _ = load_checkpoint(str(path), device)
    with torch.inference_mode():
        after = reloaded(*probe[:3])
    if not torch.equal(before, after):
        raise ValueError("saved checkpoint differs from the selected in-memory logits")


def train(dataset, output, *, device="cuda", seed=1729, updates=2000, max_seconds=180,
          initial_checkpoint=None, patience=8, energy_contexts_only=False, learning_rate=0.001):
    if isinstance(updates, bool) or not isinstance(updates, int) or not 1 <= updates <= 5000:
        raise ValueError("updates must be 1 through 5000")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 0.001:
        raise ValueError("bounded positive learning rate required")
    if not isinstance(max_seconds, (int, float)) or isinstance(max_seconds, bool) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 300:
        raise ValueError("update wall budget must be positive and at most 300 seconds")
    if isinstance(patience, bool) or not isinstance(patience, int) or not 1 <= patience <= 30:
        raise ValueError("patience must be 1 through 30 validation checks")
    training, validation, aliases, provenance = load_dataset(dataset)
    if energy_contexts_only:
        def energy_context(item):
            if item.get("kind") != "ordinary":
                return False
            view = item["actorView"]
            energy_ids = {card["instanceId"] for card in view["hand"] if card.get("category") == "energy"}
            return any(action.get("kind") == "play" and action.get("sourceCardId") in energy_ids
                       for action in view["actions"])
        original_counts = {"train": len(training), "validation": len(validation)}
        training = [item for item in training if energy_context(item)]
        validation = [item for item in validation if energy_context(item)]
        if not training or not validation:
            raise ValueError("energy contexts required in both whole-game splits")
        provenance["recordFilter"] = {"kind": "ordinary-with-legal-energy-placement/1",
            "before": original_counts, "after": {"train": len(training), "validation": len(validation)},
            "usesEvaluationTracesAsLabels": False}
    selected = configure(device, seed)
    output = Path(output).resolve()
    model_root = model_root_for(output)
    if output.exists():
        raise ValueError("fresh output directory under an approved model root required")
    model = CandidateScorer().to(selected)
    initial_sha = None
    if initial_checkpoint:
        model, initial_saved = load_checkpoint(initial_checkpoint, selected)
        if initial_saved.get("printingAliases", {}) != aliases:
            raise ValueError("initial checkpoint aliases differ from dataset")
        initial_sha = sha256(Path(initial_checkpoint))
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    initial_validation = evaluate(model, validation, selected)
    output.mkdir(parents=True)
    checkpoint = output / "best-model.pt"
    reload_probe = padded_batch(validation[:64], selected, labelled=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    permutation, cursor, epochs, examples = [], 0, 0, 0
    best_loss, best_update, stale = math.inf, None, 0
    history, losses, gradients = [], [], []
    if initial_checkpoint:
        best_loss, best_update = initial_validation["loss"], 0
        history.append({"update": 0, **initial_validation, "improved": True, "candidate": "initial-checkpoint"})
        save_verified_checkpoint(checkpoint, {"featureContract": CONTRACT, "contractSha256": CONTRACT_SHA256,
            "schema": CONTRACT["schema"], "state_dict": model.state_dict(), "seed": seed,
            "printingAliases": aliases, "update": 0, "datasetSha256": provenance["datasetSha256"],
            "initialCheckpointSha256": initial_sha, "objective": "legal-candidate supervised imitation"},
            model, reload_probe, selected)
    stop_reason = "update-limit"
    gpu_start = gpu_end = None
    if selected.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected)
        gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(selected)
        gpu_start.record()
    started = time.perf_counter()
    for update in range(1, updates + 1):
        if time.perf_counter() - started >= max_seconds:
            stop_reason = "wall-budget"
            break
        if cursor == len(permutation):
            permutation = torch.randperm(len(training), generator=generator).tolist()
            cursor = 0
            epochs += 1
        indices = permutation[cursor:cursor + 64]
        cursor += len(indices)
        examples += len(indices)
        batch = padded_batch([training[i] for i in indices], selected, labelled=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(*batch[:3])
        loss = supervised_loss(logits, batch[2], batch[3])
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("missing or nonfinite gradients")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("nonfinite model parameters")
        losses.append(float(loss.detach().cpu()))
        gradients.append(float(norm.cpu()))
        if update % 25 == 0 or update == updates:
            measured = evaluate(model, validation, selected)
            improved = measured["loss"] < best_loss - 0.0001
            history.append({"update": update, **measured, "improved": improved})
            if improved:
                best_loss, best_update, stale = measured["loss"], update, 0
                save_verified_checkpoint(checkpoint, {"featureContract": CONTRACT, "contractSha256": CONTRACT_SHA256,
                    "schema": CONTRACT["schema"], "state_dict": model.state_dict(), "seed": seed,
                    "printingAliases": aliases, "update": update, "datasetSha256": provenance["datasetSha256"],
                    "objective": "legal-candidate supervised imitation"}, model, reload_probe, selected)
            else:
                stale += 1
            if stale >= patience:
                stop_reason = "validation-patience"
                break
    if gpu_end:
        gpu_end.record()
        torch.cuda.synchronize(selected)
    elapsed = time.perf_counter() - started
    if not losses:
        raise ValueError("no training updates completed")
    if best_update is None or history[-1]["update"] != len(losses):
        measured = evaluate(model, validation, selected)
        improved = measured["loss"] < best_loss - 0.0001
        history.append({"update": len(losses), **measured, "improved": improved})
        if improved:
            best_loss, best_update = measured["loss"], len(losses)
            save_verified_checkpoint(checkpoint, {"featureContract": CONTRACT, "contractSha256": CONTRACT_SHA256,
                "schema": CONTRACT["schema"], "state_dict": model.state_dict(), "seed": seed,
                "printingAliases": aliases, "update": best_update, "datasetSha256": provenance["datasetSha256"],
                "objective": "legal-candidate supervised imitation"}, model, reload_probe, selected)
    best, _ = load_checkpoint(str(checkpoint), selected)
    delta = sum(float((p.detach() - before).abs().sum().cpu()) for p, before in zip(best.parameters(), initial_parameters))
    retained_initial = initial_checkpoint is not None and best_update == 0
    if delta <= 0 and not retained_initial:
        raise ValueError("selected checkpoint did not change from initialization")
    if retained_initial and delta != 0:
        raise ValueError("retained initial checkpoint weights changed")
    receipt = {"schema": "native-card-action-training/1", "featureContract": CONTRACT, "contractSha256": CONTRACT_SHA256,
        "provenance": provenance, "initialCheckpointSha256": initial_sha, "completedUpdates": len(losses),
        "bestUpdate": best_update, "stopReason": stop_reason, "examplesProcessed": examples, "epochsStarted": epochs,
        "retainedInitialCheckpoint": retained_initial,
        "initialValidation": initial_validation, "bestValidation": evaluate(best, validation, selected),
        "bestTraining": evaluate(best, training, selected), "history": history, "losses": losses,
        "gradientNorms": gradients, "weightL1ChangeFromInitialization": delta, "reloadLogitsExact": True,
        "checkpoint": str(checkpoint), "checkpointSha256": sha256(checkpoint), "device": str(selected),
        "torch": torch.__version__, "parameters": sum(p.numel() for p in best.parameters()),
        "updateWallSeconds": elapsed, "cudaEventMilliseconds": gpu_start.elapsed_time(gpu_end) if gpu_start else None,
        "updateTimingScope": "update loop including periodic validation and verified checkpoint writes; excludes imports, dataset load, initial validation and final checks",
        "peakCudaAllocatedBytes": torch.cuda.max_memory_allocated(selected) if gpu_start else None,
        "limits": {"updates": updates, "maximumUpdateSeconds": max_seconds, "batchSize": 64,
                   "validationEvery": 25, "patience": patience, "learningRate": learning_rate},
        "selectionUsesFinalEvaluation": False, "fullStandardClaim": False}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    config = {"executable": sys.executable, "checkpoint": str(checkpoint), "seed": seed,
              "device": str(selected), "printingAliases": aliases}
    if model_root != RUNS.resolve():
        config["modelRoot"] = str(model_root)
    (output / "model-config.json").write_text(json.dumps(config, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--energy-contexts-only", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    result = train(args.dataset, args.output, device=args.device, seed=args.seed, updates=args.updates,
                   max_seconds=args.max_seconds, initial_checkpoint=args.initial_checkpoint, patience=args.patience,
                   energy_contexts_only=args.energy_contexts_only, learning_rate=args.learning_rate)
    print(json.dumps({key: result[key] for key in ("completedUpdates", "bestUpdate", "stopReason", "bestValidation", "checkpointSha256")}))


if __name__ == "__main__":
    main()
