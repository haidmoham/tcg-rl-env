"""Brief game-disjoint Model6 energy BC warm start from trusted Model5 weights."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from learning import model6_torch as bc
from learning.card_action_torch import RUNS
import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("relative artifact path required")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("artifact escaped root or is missing")
    return path


def load_dataset(value, catalog):
    path = Path(value).resolve()
    if path.is_dir():
        path /= "receipt.json"
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if (manifest.get("schema") not in ("model6-energy-dataset/1", "entity-policy-dataset/1")
            or manifest.get("complete") is not True or manifest.get("fatal") is not False
            or manifest.get("featureContract") != bc.CONTRACT or manifest.get("catalogSha256") != catalog["catalogSha256"]):
        raise ValueError("complete fresh Model6 energy collection with matching catalog required")
    sources = manifest.get("sourceHashes")
    required = {"learning/model6_torch.py", "learning/train_model6.py", "learning/model6-contract.json",
                "learning/model6-features.mjs", "learning/entity_torch.py", "learning/entity-contract.json"}
    if not isinstance(sources, dict) or not required.issubset(sources):
        raise ValueError("Model6 and frozen Model5 source hashes required")
    for name, digest in sources.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest) or sha256(artifact(REPO, name)) != digest:
            raise ValueError("source freeze mismatch")
    games = manifest.get("games")
    if not isinstance(games, list) or not 2 <= len(games) <= 256:
        raise ValueError("bounded whole-game collection required")
    splits, identities, seeds, paths = {"train": [], "validation": []}, set(), set(), set()
    split_games = {"train": 0, "validation": 0}
    total_bytes, total_numbers = 0, 0
    for game in games:
        if (not isinstance(game, dict) or game.get("split") not in splits or not isinstance(game.get("id"), str)
                or not game["id"] or game["id"] in identities or type(game.get("seed")) is not int
                or not 0 <= game["seed"] <= 0xffffffff or game["seed"] in seeds):
            raise ValueError("unique game identities and disjoint seeds required")
        identities.add(game["id"]); seeds.add(game["seed"])
        if (game.get("completedNaturally") is not True or game.get("rejected") != 0
                or type(game.get("commands")) is not int or game["commands"] <= 0
                or type(game.get("privacyChecks")) is not int or game["privacyChecks"] < 2
                or game.get("auditMode", manifest.get("auditMode")) != "sampled-native-training/1"):
            raise ValueError("natural completion, legal actions and sampled privacy gates required")
        records_path = artifact(path.parent, game.get("recordsPath"))
        total_bytes += records_path.stat().st_size
        if total_bytes > 512 * 1024 * 1024 or records_path in paths or sha256(records_path) != game.get("recordsSha256"):
            raise ValueError("record bytes, identity or raw digest mismatch")
        paths.add(records_path)
        count = 0
        with records_path.open("rb") as stream:
            for line in stream:
                if len(line) > 4 * 1024 * 1024 or not line.strip():
                    raise ValueError("bounded nonempty record lines required")
                row = json.loads(line); bc.record(row, catalog, labelled=True)
                total_numbers += (len(row["globals"]) + len(row["entities"]) * (bc.m5.ENTITY_DIM + bc.ENERGY_ENTITY_DIM)
                                  + len(row["candidates"]) * (bc.m5.CANDIDATE_DIM + bc.ENERGY_CANDIDATE_DIM)
                                  + (len(row["energyAttacks"]) + len(row["energyCandidateAttacks"])) * bc.ENERGY_ATTACK_DIM)
                if total_numbers > 40000000:
                    raise ValueError("feature memory budget exceeded")
                splits[game["split"]].append(row); count += 1
        if not count:
            raise ValueError("game has no decisions")
        split_games[game["split"]] += 1
    if any(not rows for rows in splits.values()) or sum(map(len, splits.values())) > 25000:
        raise ValueError("bounded nonempty game-disjoint splits required")
    return splits["train"], splits["validation"], {"datasetSha256": sha256(path), "sourceHashes": sources,
        "games": split_games, "gameIds": sorted(identities), "seeds": sorted(seeds),
        "trainingRecords": len(splits["train"]), "validationRecords": len(splits["validation"])}


def evaluate(model, records, device, catalog, details=False):
    total_loss, ordinary, ordinary_correct, selections, selection_hits, exact = 0., 0, 0, 0, 0, 0
    choices = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(records), 32):
            rows = records[start:start + 32]
            batch, targets = bc.padded_batch(rows, device, catalog, labelled=True)
            logits = model(**batch)
            loss = bc.supervised_loss(logits, batch["candidate_mask"], targets, reduction="sum")
            if not torch.isfinite(loss) or not torch.isfinite(logits[batch["candidate_mask"]]).all():
                raise ValueError("nonfinite validation score or loss")
            total_loss += float(loss.cpu())
            for index, row in enumerate(rows):
                scores = logits[index, :len(row["candidates"])].cpu().tolist()
                ranking = sorted(range(len(scores)), key=lambda choice: (-scores[choice], choice))
                if details:
                    label = row["label"]
                    probabilities = torch.softmax(logits[index, :len(scores)], 0).cpu().tolist()
                    choices.append({"decisionId": row["humanLabel"]["decisionId"],
                        "actionKind": row["humanLabel"]["actionKind"], "label": label,
                        "prediction": ranking[0], "humanChoiceRank": ranking.index(label) + 1,
                        "humanChoiceProbability": probabilities[label], "candidates": len(scores)})
                if row["kind"] == "selection":
                    chosen = set(row["labels"]); selections += 1
                    selection_hits += int(ranking[0] in chosen); exact += int(set(ranking[:len(chosen)]) == chosen)
                else:
                    ordinary += 1; ordinary_correct += int(ranking[0] == row["label"])
    return {**({"choices": choices, "meanHumanChoiceProbability": sum(x["humanChoiceProbability"] for x in choices) / len(choices),
                "meanHumanChoiceRank": sum(x["humanChoiceRank"] for x in choices) / len(choices)} if details else {}),
            "records": len(records), "loss": total_loss / len(records),
            "ordinary": {"records": ordinary, "top1Agreement": ordinary_correct / ordinary if ordinary else None},
            "selection": {"records": selections, "top1InChosenSet": selection_hits / selections if selections else None,
                          "exactChosenSetAgreement": exact / selections if selections else None,
                          "exactSetMetricCardinality": "teacher chosen count; not a deployed count prediction"}}


def train(dataset, output, *, catalog, initial_checkpoint, initial_sha256, device="cuda", seed=1729,
          updates=300, max_seconds=180, learning_rate=.0003, freeze_base=False, human_localized=False):
    if type(freeze_base) is not bool:
        raise ValueError("freeze_base must be boolean")
    if type(updates) is not int or not 1 <= updates <= 500 or type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 300:
        raise ValueError("brief warm start needs 1 through 500 updates and at most 300 update seconds")
    if learning_rate not in (.0001, .0003):
        raise ValueError("brief BC learning rate must be .0001 or .0003")
    if human_localized and (updates > 100 or max_seconds > 120):
        raise ValueError("human localized fit is capped at 100 updates and 120 seconds")
    if not isinstance(initial_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", initial_sha256):
        raise ValueError("explicit trusted Model5 checkpoint SHA256 required")
    initial_checkpoint = bc.private_checkpoint(initial_checkpoint)
    if sha256(initial_checkpoint) != initial_sha256:
        raise ValueError("trusted Model5 checkpoint digest mismatch")
    catalog_path = Path(catalog).resolve(); metadata = bc.catalog_metadata(catalog_path)
    if human_localized:
        from learning import human_bc
        training, validation, provenance = human_bc.load_dataset(dataset, metadata)
        if provenance["baselineSha256"] != initial_sha256:
            raise ValueError("human dataset and PPO parent differ")
    else:
        training, validation, provenance = load_dataset(dataset, metadata)
    device = bc.configure(device, seed)
    output = Path(output).resolve(); model_root = bc.model_root_for(output)
    if output.exists():
        raise ValueError("fresh approved Model6 output directory required")
    if human_localized:
        model, warm = human_bc.warm_start(initial_checkpoint, metadata, training, device, initial_sha256)
    else:
        model = bc.EnergyEntityScorer(metadata).to(device)
        warm = model.warm_start_model5(str(initial_checkpoint), metadata, validation[:32], device)
    base_sha256 = warm["baseModel5CheckpointSha256"] if human_localized else initial_sha256
    model.base.requires_grad_(not freeze_base)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    initial_validation = evaluate(model, validation, device, metadata, details=human_localized)
    best_state, best_loss, best_update = initial_state, initial_validation["loss"], 0
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    order, cursor, history, losses, gradients, stale = [], 0, [], [], [], 0
    started = time.perf_counter(); stop = "update-limit"
    for update in range(1, updates + 1):
        if time.perf_counter() - started >= max_seconds:
            stop = "wall-budget"; break
        if cursor == len(order):
            order = torch.randperm(len(training), generator=generator).tolist(); cursor = 0
        indices = order[cursor:cursor + 32]; cursor += len(indices)
        batch, targets = bc.padded_batch([training[index] for index in indices], device, metadata, labelled=True)
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model(**batch); loss = bc.supervised_loss(logits, batch["candidate_mask"], targets)
        if not torch.isfinite(loss) or not torch.isfinite(logits[batch["candidate_mask"]]).all():
            raise ValueError("nonfinite BC training loss or score")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
            raise ValueError("missing or nonfinite BC gradient")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("nonfinite BC parameter")
        losses.append(float(loss.detach().cpu())); gradients.append(float(norm.cpu()))
        if update % 25 == 0 or update == updates:
            measured = evaluate(model, validation, device, metadata); improved = measured["loss"] < best_loss - .0001
            history.append({"update": update, **measured, "improved": improved})
            if improved:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_loss, best_update, stale = measured["loss"], update, 0
            else:
                stale += 1
            if stale >= 4:
                stop = "validation-patience"; break
    if not losses:
        raise ValueError("no BC updates completed")
    if not history or history[-1]["update"] != len(losses):
        measured = evaluate(model, validation, device, metadata)
        if measured["loss"] < best_loss - .0001:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_loss, best_update = measured["loss"], len(losses)
        history.append({"update": len(losses), **measured, "improved": best_update == len(losses)})
    base_exact_after_bc = all(torch.equal(best_state[key], initial_state[key]) for key in best_state if key.startswith("base."))
    if freeze_base and not base_exact_after_bc:
        raise ValueError("frozen Model5 base state changed during Model6 BC")
    model.load_state_dict(best_state, strict=True); model.eval()
    elapsed = time.perf_counter() - started
    for name, expected in provenance["sourceHashes"].items():
        if sha256(artifact(REPO, name)) != expected:
            raise ValueError("collection sources changed during BC")
    if sha256(initial_checkpoint) != initial_sha256:
        raise ValueError("base Model5 checkpoint changed during BC")
    output.mkdir(parents=True); checkpoint = output / "best-model.pt"
    payload = {"schema": bc.CHECKPOINT_SCHEMA, "featureContract": bc.CONTRACT, "contractSha256": bc.CONTRACT_SHA256,
               "architecture": model.architecture, "catalog": metadata, "state_dict": best_state,
               "baseModel5CheckpointSha256": base_sha256, "datasetSha256": provenance["datasetSha256"],
               "initialization": warm.get("initialization", "trusted-model5-zero-energy-residual/1"),
               **({"parentPpoCheckpointSha256": initial_sha256} if human_localized else {}), "seed": seed, "update": best_update,
               "baseFrozenDuringBc": freeze_base}
    probe, _ = bc.padded_batch(validation[:32], device, metadata)
    with torch.inference_mode():
        expected = model(**probe).detach().clone()
    with checkpoint.open("xb") as stream:
        torch.save(payload, stream)
    reloaded, _ = bc.load_checkpoint(str(checkpoint), device, metadata)
    with torch.inference_mode():
        if not torch.equal(expected, reloaded(**probe)):
            raise ValueError("selected Model6 BC checkpoint reload changed actual-record logits")
    delta = sum(float((best_state[key] - initial_state[key]).abs().sum()) for key in best_state)
    receipt = {"schema": "model6-energy-bc-training/1", "featureContract": bc.CONTRACT,
        "contractSha256": bc.CONTRACT_SHA256, "catalog": metadata, "architecture": model.architecture,
        "provenance": provenance, "baseModel5CheckpointSha256": base_sha256, **warm,
        "completedUpdates": len(losses), "bestUpdate": best_update, "retainedInitialCheckpoint": best_update == 0,
        "stopReason": stop, "initialValidation": initial_validation, "bestValidation": evaluate(reloaded, validation, device, metadata),
        "bestTraining": evaluate(reloaded, training, device, metadata), "losses": losses, "gradientNorms": gradients,
        "history": history, "weightL1ChangeFromInitialization": delta, "reloadLogitsExact": True,
        "checkpoint": str(checkpoint), "checkpointSha256": sha256(checkpoint), "parameters": sum(p.numel() for p in model.parameters()),
        "baseParameters": sum(p.numel() for p in model.base.parameters()),
        "trainableParameters": sum(p.numel() for p in trainable), "baseFrozenDuringBc": freeze_base,
        "baseModel5StateExactAfterBc": base_exact_after_bc,
        "energyExtraParameters": sum(p.numel() for name, p in model.named_parameters() if not name.startswith("base.")),
        "device": str(device), "torch": torch.__version__, "updateWallSeconds": elapsed,
        "limits": {"updates": updates, "maximumUpdateSeconds": max_seconds, "batchSize": 32, "validationEvery": 25,
                   "patience": 4, "learningRate": learning_rate}, "fullStandardClaim": False}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    config = {"executable": sys.executable, "checkpoint": str(checkpoint), "catalogPath": str(catalog_path),
              "catalogSha256": metadata["catalogSha256"], "seed": seed, "device": str(device)}
    if model_root != RUNS.resolve():
        config["modelRoot"] = str(model_root)
    if human_localized:
        receipt["schema"] = "model6-human-localized-bc/1"
        receipt["initialTargetFit"] = receipt.pop("initialValidation")
        receipt.pop("bestValidation")
        receipt["postTargetFit"] = evaluate(reloaded, training, device, metadata, details=True)
        receipt["heldOutValidation"] = None
        receipt["transferClaim"] = False
        receipt["selectionLabelsTrained"] = False
        receipt["baseActorStateExactAfterBc"] = receipt.pop("baseModel5StateExactAfterBc")
        for entry in receipt["history"]:
            entry["measurementScope"] = "fitted-examples-only"
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "model-config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("dataset"); parser.add_argument("output")
    parser.add_argument("--catalog", required=True); parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--initial-sha256", required=True); parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1729); parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--human-localized", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=180); parser.add_argument("--learning-rate", type=float, default=.0003)
    args = parser.parse_args(); result = train(args.dataset, args.output, catalog=args.catalog,
        initial_checkpoint=args.initial_checkpoint, initial_sha256=args.initial_sha256, device=args.device, seed=args.seed,
        updates=args.updates, max_seconds=args.max_seconds, learning_rate=args.learning_rate, freeze_base=args.freeze_base,
        human_localized=args.human_localized)
    summary = {key: result[key] for key in ("checkpoint", "checkpointSha256", "bestUpdate", "completedUpdates")}
    if not args.human_localized:
        summary["bestValidation"] = result["bestValidation"]
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
