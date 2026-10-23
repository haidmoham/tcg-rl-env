"""Bounded two-head human BC; admission/evaluation gate runs before torch import."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def prepare(path, expected_sha256):
    path = Path(path).resolve()
    if digest(path) != expected_sha256:
        raise ValueError("admission receipt changed")
    receipt = read(path)
    if receipt.get("schema") != "supervised-bootstrap-admission/1":
        raise ValueError("explicit bootstrap admission required")
    split = {}
    for name, key in (("train", "trainSha256"), ("evaluation", "evaluationSha256")):
        file = path.parent / (name + ".jsonl")
        if digest(file) != receipt[key]:
            raise ValueError("admitted dataset changed")
        split[name] = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
    train_games, eval_games = set(receipt["trainingGames"]), set(receipt["evaluationGames"])
    if train_games & eval_games:
        raise ValueError("whole-game evaluation overlap")
    counts = {name: {task: sum(r["bootstrap"].get("head",r["bootstrap"]["task"]) == task for r in rows)
                     for task in ("ordinary-choice", "selection-ranking")} for name, rows in split.items()}
    simulator = receipt.get("evaluationProtocol")
    if simulator and digest(simulator["path"]) != simulator["sha256"]:
        raise ValueError("predeclared simulator protocol changed")
    ready = receipt["fitGate"]["ready"] and (bool(simulator) or (bool(eval_games) and all(counts["evaluation"].values())))
    return path, receipt, split, {"ready": bool(ready), "counts": counts,
        "reason": receipt["fitGate"]["reason"] if ready else "fresh whole-game evaluation reservation required before fitting",
        "trainingStarted": False, "optimizerSteps": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--ordinary-checkpoint")
    parser.add_argument("--selection-checkpoint")
    parser.add_argument("--model-root")
    parser.add_argument("--output")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--max-updates", type=int, default=100)
    parser.add_argument("--max-seconds", type=float, default=120)
    args = parser.parse_args()
    path, receipt, split, gate = prepare(args.admission, args.admission_sha256)
    print(json.dumps(gate), flush=True)
    if args.inspect or not gate["ready"]:
        return
    if not 2 <= args.max_updates <= 100 or not 0 < args.max_seconds <= 120:
        raise ValueError("bounded initial two-head fit only")
    if not all((args.ordinary_checkpoint, args.selection_checkpoint, args.output, args.model_root)):
        raise ValueError("explicit copied-parent checkpoints and fresh candidate output required")
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("candidate output must be fresh")
    parent_paths = {"ordinary-choice": Path(args.ordinary_checkpoint).resolve(),
                    "selection-ranking": Path(args.selection_checkpoint).resolve()}
    expected = {"ordinary-choice": receipt["baselineSha256"], "selection-ranking": receipt["selectionSha256"]}
    if any(digest(parent_paths[t]) != expected[t] for t in parent_paths):
        raise ValueError("parent checkpoint changed")
    os.environ["PTCG_AGENT_MODEL_ROOT"] = str(Path(args.model_root).resolve())
    # No torch, model, inference, or checkpoint load occurs before the gate above.
    import torch
    from learning import model6_torch as bc
    from learning import human_bc
    device = bc.configure("cpu", 1729)
    bc.model_root_for(output/"ordinary-candidate.pt")
    catalog = path.parent / "catalog.json"
    metadata = bc.catalog_metadata(catalog)
    if metadata["catalogSha256"] != receipt["catalogSha256"] or bc.CONTRACT_SHA256 != receipt["contractSha256"]:
        raise ValueError("frozen catalog/feature contract changed")
    tasks = tuple(parent_paths)
    groups = {s: {t: [r for r in rows if r["bootstrap"].get("head",r["bootstrap"]["task"]) == t] for t in tasks} for s, rows in split.items()}
    for rows in split.values():
        for row in rows:
            bc.record(row, metadata, labelled=True)
            annotation = row["bootstrap"]
            sets = annotation["acceptableIndexSets"]
            if (annotation.get("eligibleForPpo") is not False or annotation.get("behaviorLogProbability") is not None
                    or not sets or any(not ids or len(ids) != len(set(ids))
                        or any(type(i) is not int or not 0 <= i < len(row["candidates"]) for i in ids) for ids in sets)):
                raise ValueError("invalid human supervision")
            if len(sets) > 1 and any(len(ids) != 1 for ids in sets):
                raise ValueError("joint acceptable combinations need a separate objective")
    ordinary, initialization = human_bc.warm_start(str(parent_paths[tasks[0]]), metadata, groups["train"][tasks[0]], device, expected[tasks[0]])
    selection, selection_saved = bc.load_checkpoint(str(parent_paths[tasks[1]]), device, metadata)
    models = {tasks[0]: ordinary, tasks[1]: selection}
    optimizers = {t: torch.optim.Adam(models[t].parameters(), lr=1e-4) for t in tasks}

    def losses(logits, rows):
        result = []
        for index, row in enumerate(rows):
            logp = logits[index, :len(row["candidates"])].log_softmax(0)
            sets = row["bootstrap"]["acceptableIndexSets"]
            if len(sets) > 1:
                ids = sorted({ids[0] for ids in sets})
                result.append(-torch.logsumexp(logp[ids], 0))
            else:
                # Ranking chosen members of an observed set; this is not a
                # normalized distribution over legal prompt combinations.
                result.append(-logp[sets[0]].mean())
        return torch.stack(result)

    def measure(task, split_name="train"):
        rows = groups[split_name][task]
        total, correct, count = 0.0, 0, 0
        models[task].eval()
        with torch.inference_mode():
            for start in range(0, len(rows), 8):
                part = rows[start:start+8]
                batch, _ = bc.padded_batch(part, device, metadata)
                logits = models[task](**batch)
                total += float(losses(logits, part).sum())
                for i, row in enumerate(part):
                    acceptable = {j for ids in row["bootstrap"]["acceptableIndexSets"] for j in ids}
                    correct += int(int(logits[i].argmax()) in acceptable)
                    count += 1
        return {"records": count, "meanLoss": total/count, "topChoiceInAnnotatedSet": correct,
                "interpretation": "development/training material only; no held-out or gameplay-strength claim"}

    before = {t: measure(t) for t in tasks}
    started, updates = time.monotonic(), {t: 0 for t in tasks}
    print(json.dumps({"trainingStarted":True,"budgetUpdates":args.max_updates,"budgetTrainingSeconds":args.max_seconds,
                      "headRecords":gate["counts"]["train"]}),flush=True)
    for step in range(args.max_updates):
        if time.monotonic() - started >= args.max_seconds:
            break
        task = tasks[step % 2]
        rows = groups["train"][task]
        offset = (updates[task] * 8) % len(rows)
        part = [rows[(offset+i) % len(rows)] for i in range(min(8,len(rows)))]
        batch, _ = bc.padded_batch(part, device, metadata)
        models[task].train()
        loss = losses(models[task](**batch), part).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite supervised loss")
        optimizers[task].zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(models[task].parameters(), 1)
        optimizers[task].step()
        updates[task] += 1
        if (step+1)%10 == 0:
            print(json.dumps({"updates":updates,"trainingSeconds":time.monotonic()-started}),flush=True)
    after = {t: measure(t) for t in tasks}
    if any(digest(parent_paths[t]) != expected[t] for t in tasks):
        raise ValueError("original checkpoint changed")
    output.mkdir(parents=True)
    # Preserve the ordinary PPO wrapper and every critic/optimizer tensor;
    # only replace actor weights, with BC updates attributed separately.
    ordinary_saved = torch.load(parent_paths[tasks[0]], map_location="cpu", weights_only=True)
    for key, tensor in ordinary.state_dict().items():
        ordinary_saved["state_dict"]["actor."+key] = tensor.detach().cpu()
    selection_saved["state_dict"] = {k:v.detach().cpu() for k,v in selection.state_dict().items()}
    proof = {"schema":"two-head-human-bootstrap-fit/1", "admissionSha256":args.admission_sha256,
        "parentSha256":expected, "updates":updates,"trainingSeconds":time.monotonic()-started,
        "developmentBefore":before,"developmentAfter":after,"evaluationUsedForSelectionOrTuning":False,
        "generalizationMeasured":False,"evaluationProtocol":receipt.get("evaluationProtocol"),
        "initialization":initialization,"promotion":False,"ppoRun":False,"criticOptimizerUsed":False,
        "selectionSupport":"offered-card ranking, with unchanged native preview/combination driver"}
    for task, payload, filename in ((tasks[0],ordinary_saved,"ordinary-candidate.pt"),
                                     (tasks[1],selection_saved,"selection-candidate.pt")):
        payload["humanBootstrap"] = {"admissionSha256":args.admission_sha256,"task":task,"updates":updates[task]}
        torch.save(payload, output/filename)
    proof["candidateSha256"]={"ordinary":digest(output/"ordinary-candidate.pt"),"selection":digest(output/"selection-candidate.pt")}
    (output/"receipt.json").write_text(json.dumps(proof,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(proof),flush=True)


if __name__ == "__main__":
    main()
