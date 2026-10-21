"""Supervise one staged trainer and scheduled frozen-checkpoint evaluations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def launch(command, log_path):
    log = log_path.open("xb")
    try:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    except Exception:
        log.close()
        raise
    return {"process": process, "log": log, "pid": process.pid, "exit_code": None}


def poll(job):
    code = job["process"].poll()
    if code is not None and job["exit_code"] is None:
        job["exit_code"] = code
        job["log"].close()
    return code


def terminate(job):
    if job is not None:
        try:
            os.killpg(job["pid"], signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            job["process"].wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(job["pid"], signal.SIGKILL)
            job["process"].wait()
        finally:
            try:
                os.killpg(job["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        poll(job)


def run(args):
    invoked = time.monotonic()
    if os.name != "posix":
        raise RuntimeError("Campaign supervision requires the pinned Linux/WSL runtime")
    protocol, bindings = read(args.protocol), read(args.bindings)
    artifact_map = json.dumps(bindings["artifacts"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(artifact_map).hexdigest() != bindings["artifact_manifest_sha256"]:
        raise ValueError("campaign source closure digest differs")
    if protocol["schema"] != "86-tcg-agent/native-sustained/1" or protocol["reporting"]["interval_seconds"] != 1800:
        raise ValueError("campaign must use the sustained protocol and 30-minute time-only reports")
    if protocol["evaluation"]["games_per_checkpoint"] != 576 or protocol["reporting"]["update_trigger"]:
        raise ValueError("campaign requires 576-game evaluations with no update-count report trigger")
    for path, expected in bindings["artifacts"].items():
        if sha(Path(path)) != expected:
            raise ValueError("a bound campaign dependency changed")
    if str(Path(__file__).resolve()) not in {str(Path(path).resolve()) for path in bindings["artifacts"]}:
        raise ValueError("actual campaign supervisor is not in the source closure")
    if not args.validate_only and protocol["execution_enabled"] is not True:
        raise ValueError("campaign protocol does not authorize execution")
    args.output.mkdir(parents=True, exist_ok=False)
    atomic(args.output / "campaign-manifest.json", {
        "schema": "native-sustained-campaign/1", "protocol": protocol,
        "protocol_sha256": sha(args.protocol), "bindings_sha256": sha(args.bindings),
        "source_artifact_manifest_sha256": bindings["artifact_manifest_sha256"],
        "supervisor_pid": os.getpid(), "python": sys.executable,
        "training_directory": str(args.output / "training"), "mode": "validation" if args.validate_only else "run",
    })
    if args.validate_only:
        atomic(args.output / "validation.json", {"passed": True, "training_started": False})
        return
    os.nice(max(0, 19 - os.nice(0)))
    def interrupted(signum, frame):
        raise InterruptedError(f"campaign received signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    trainer = launch([sys.executable, str(ROOT / "scripts/train_native_sustained.py"), "--mode", "train",
                      "--protocol", str(args.protocol), "--bindings", str(args.bindings),
                      "--campaign-start-monotonic", str(invoked),
                      "--output", str(args.output / "training")], args.output / "training.log")
    pending, evaluations, known, reports, snapshots = [], [], {}, [], set()
    active = None
    final_requested = False
    training = args.output / "training"
    deadline = protocol["limits"]["wall_seconds"]
    schedule = protocol["reporting"]["scheduled_campaign_seconds"]
    lead = protocol["reporting"]["evaluation_lead_seconds"]
    maximum = protocol["evaluation"]["max_evaluation_checkpoints"]
    training_deadline = protocol["limits"]["training_stop_campaign_seconds"]

    def final_complete():
        return any("final" in item["labels"] and item.get("exit_code") == 0
                   and item.get("result", {}).get("status") == "complete"
                   and item["result"].get("frozen_player_unchanged") is True
                   and item["result"].get("checkpoint_files_unchanged") is True
                   for item in evaluations if item.get("result") is not None)

    def request(pointer, label):
        if not pointer.is_file():
            return False
        descriptor = read(pointer)
        identity = descriptor["weights_sha256"]
        if identity in known:
            known[identity]["labels"].append(label)
            return True
        if len(known) >= maximum:
            raise ValueError("predeclared maximum evaluation checkpoints exceeded")
        number = len(known)
        receipt = args.output / f"evaluation-request-{number:02d}.json"
        atomic(receipt, descriptor)
        item = {"number": number, "labels": [label], "weights_sha256": identity,
                "checkpoint_receipt": str(receipt), "requested_campaign_seconds": time.monotonic() - invoked,
                "output": str(args.output / f"evaluation-{number:02d}")}
        known[identity] = item
        pending.append(item)
        return True

    def completed_reference(label):
        for item in evaluations:
            if label in item["labels"] and Path(item["output"], "evaluation-result.json").is_file():
                return item
        return None

    try:
        while True:
            elapsed = time.monotonic() - invoked
            training_code = poll(trainer)
            request(training / "initial-checkpoint.json", "initial") if "initial" not in snapshots else None
            if (training / "initial-checkpoint.json").is_file():
                snapshots.add("initial")
            if "pre_rl" not in snapshots and request(training / "pre-rl-checkpoint.json", "pre_rl"):
                snapshots.add("pre_rl")
            for target in schedule:
                if target not in snapshots and elapsed >= max(0, target - lead):
                    if request(training / "latest-checkpoint.json", f"report_{target}"):
                        snapshots.add(target)
            if training_code is not None and not final_requested:
                final_requested = request(training / "latest-checkpoint.json", "final")
                if not final_requested:
                    raise RuntimeError("trainer exited without a completed checkpoint")
            if final_requested:
                for item in pending[:]:
                    if "final" not in item["labels"]:
                        item["status"] = "skipped_to_reserve_final_evaluation"
                        item["finished_campaign_seconds"] = elapsed
                        evaluations.append(item)
                        pending.remove(item)
                pending.sort(key=lambda item: "final" not in item["labels"])
            if active is not None and poll(active["job"]) is not None:
                item = active["item"]
                item["exit_code"] = active["job"]["exit_code"]
                item["finished_campaign_seconds"] = elapsed
                result_path = Path(item["output"]) / "evaluation-result.json"
                item["result"] = read(result_path) if result_path.is_file() else None
                evaluations.append(item)
                active = None
            if active is None and pending and elapsed < deadline and (elapsed < training_deadline or final_requested):
                item = pending.pop(0)
                command = [sys.executable, str(ROOT / "scripts/evaluate_native_checkpoint.py"),
                           "--protocol", str(args.protocol), "--bindings", str(args.bindings),
                           "--campaign-evaluation-deadline-monotonic", str(invoked + (deadline if "final" in item["labels"] else training_deadline)),
                           "--checkpoint-receipt", item["checkpoint_receipt"], "--output", item["output"]]
                for label, flag in (("initial", "initial"), ("pre_rl", "pre-rl")):
                    reference = completed_reference(label)
                    if reference is not None:
                        command += [f"--{flag}-checkpoint-receipt", reference["checkpoint_receipt"],
                                    f"--{flag}-evaluation-receipt", str(Path(reference["output"]) / "evaluation-result.json")]
                active = {"item": item, "job": launch(command, args.output / f"evaluation-{item['number']:02d}.log")}
                item["pid"] = active["job"]["pid"]
            progress = read(training / "progress.json") if (training / "progress.json").is_file() else None
            status = {"schema": "native-sustained-campaign-status/1", "campaign_elapsed_seconds": elapsed,
                      "training_pid": trainer["pid"], "training_exit_code": training_code,
                      "training_progress": progress, "active_evaluation": None if active is None else active["item"],
                      "completed_evaluations": evaluations, "pending_evaluations": pending,
                      "training_updates_trigger_reports": False}
            atomic(args.output / "campaign-status.json", status)
            for target in schedule:
                if elapsed >= target and target not in reports:
                    report = {**status, "scheduled_campaign_seconds": target,
                              "report_scope": "Time-only milestone. Include checkpoint age and pending evaluations; never fabricate a completed 576-game result."}
                    atomic(args.output / f"report-{target:05d}.json", report)
                    reports.append(target)
            if elapsed >= deadline:
                terminate(trainer)
                if active is not None:
                    terminate(active["job"])
                atomic(args.output / "campaign-result.json", {**status, "status": "campaign_wall_cap",
                                                             "final_evaluation_complete": final_complete()})
                break
            if training_code is not None and final_requested and active is None and not pending:
                complete = final_complete()
                atomic(args.output / "campaign-result.json", {**status,
                    "status": "training_fault" if training_code != 0 else "bounded_campaign_finished" if complete else "final_evaluation_incomplete",
                    "final_evaluation_complete": complete})
                break
            time.sleep(5)
    finally:
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
        terminate(trainer)
        if active is not None:
            terminate(active["job"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    options = parser.parse_args()
    options.protocol = options.protocol.resolve()
    options.bindings = options.bindings.resolve()
    options.output = options.output.resolve()
    run(options)
