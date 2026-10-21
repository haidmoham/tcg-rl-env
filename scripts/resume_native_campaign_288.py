"""Adopt a live historical trainer without resetting its campaign clock."""
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
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def atomic(path, value):
    pending = Path(str(path) + ".pending")
    pending.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(pending, path)


def identity(pid):
    directory = Path("/proc") / str(pid)
    try:
        stat = (directory / "stat").read_text().split(") ", 1)[1].split()
        command = (directory / "cmdline").read_bytes().split(b"\0")
        if stat[0] == "Z" or not command[0]:
            return None
        return {"pid": pid, "start_ticks": int(stat[19]), "pgid": int(stat[2]),
                "sid": int(stat[3]), "executable": str((directory / "exe").resolve(strict=True)),
                "command": [part.decode() for part in command if part]}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def require_identity(expected):
    if identity(expected["pid"]) != expected:
        raise RuntimeError("process identity changed before an ownership operation")


def source_artifacts(args):
    return {str(path.resolve()): sha(path) for path in (
        Path(__file__), ROOT / "scripts/evaluate_native_checkpoint_288.py", args.amendment)}


def validate(args):
    if os.name != "posix" or args.output.is_relative_to(ROOT) or args.output.is_relative_to(args.campaign):
        raise ValueError("use Linux and a private successor directory outside the campaign and checkout")
    manifest = read(args.campaign / "campaign-manifest.json")
    state = read(args.campaign / "campaign-status.json")
    if (args.campaign / "campaign-result.json").exists():
        raise RuntimeError("the original campaign is already terminal")
    bindings = read(args.bindings)
    if sha(args.bindings) != manifest["bindings_sha256"]:
        raise ValueError("bindings differ from the original campaign")
    for path, expected in bindings["artifacts"].items():
        if sha(path) != expected:
            raise ValueError("a frozen campaign source changed")
    supervisor, trainer = identity(manifest["supervisor_pid"]), identity(state["training_pid"])
    if not supervisor or not trainer:
        raise RuntimeError("takeover requires the original supervisor and trainer alive")
    for process, script in ((supervisor, "run_native_campaign.py"), (trainer, "train_native_sustained.py")):
        command = process["command"]
        if not any(Path(part).name == script for part in command) or str(args.campaign) not in " ".join(command):
            raise ValueError("campaign process command does not match its receipt")
    command = trainer["command"]
    started = float(command[command.index("--campaign-start-monotonic") + 1])
    if not 0 < time.monotonic() - started < manifest["protocol"]["limits"]["training_stop_campaign_seconds"]:
        raise RuntimeError("takeover must precede the original training stop")
    active = state["active_evaluation"]
    evaluator = identity(active["pid"]) if active else None
    if evaluator:
        if (evaluator["pid"] != evaluator["pgid"] or evaluator["pid"] != evaluator["sid"]
                or "evaluate_native_checkpoint.py" not in " ".join(evaluator["command"])
                or active["output"] not in evaluator["command"]):
            raise ValueError("old evaluation is not an isolated declared session")
    training_manifest = read(args.campaign / "training/manifest.json")
    protocol_path = Path(command[command.index("--protocol") + 1]).resolve(strict=True)
    if read(protocol_path) != training_manifest["protocol"] or sha(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError("parent protocol identity differs")
    sys.path.insert(0, str(ROOT / "scripts"))
    import evaluate_native_checkpoint_288 as evaluation_api
    evaluation_api.validate_amendment(read(args.amendment), protocol_path.read_bytes())
    evaluation_api.validate_sources(bindings)
    evaluation_api.checkpoint_descriptor(args.campaign / "training/latest-checkpoint.json")
    return {"schema": "native-sustained-campaign288-takeover/1", "campaign": str(args.campaign),
            "parent_manifest_sha256": sha(args.campaign / "campaign-manifest.json"),
            "original_started_monotonic": started, "supervisor_identity": supervisor,
            "trainer_identity": trainer, "active_evaluator_identity": evaluator,
            "bindings_path": str(args.bindings), "bindings_sha256": sha(args.bindings),
            "protocol_path": str(protocol_path), "amendment_path": str(args.amendment),
            "amendment_sha256": sha(args.amendment), "source_artifacts": source_artifacts(args),
            "preflight_status": state, "supervisor_pid": os.getpid()}


def group_members(pgid):
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / "stat").read_text().split(") ", 1)[1].split()
            if int(stat[2]) == pgid and stat[0] != "Z":
                item = identity(int(path.name))
                if item is None:
                    current = (path / "stat").read_text().split(") ", 1)[1].split()
                    if current[0] == "Z":
                        continue
                    raise RuntimeError("cannot inspect a live evaluation group member")
                members.append(item)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def campaign_evaluators(campaign):
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            command = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            if str(campaign) in command and "evaluate_native_checkpoint" in command:
                item = identity(int(path.name))
                if item:
                    members.append(item)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def stop_evaluation(process, output):
    require_identity(process)
    members = group_members(process["pgid"])
    if any(item["sid"] != process["sid"] for item in members):
        raise RuntimeError("evaluation session ownership differs")
    write(output / "interrupted-evaluation-processes.json", members)
    os.killpg(process["pgid"], signal.SIGTERM)
    until = time.monotonic() + 10
    while group_members(process["pgid"]) and time.monotonic() < until:
        time.sleep(0.1)
    survivors = group_members(process["pgid"])
    if survivors:
        if any(item not in members for item in survivors):
            raise RuntimeError("unexpected process in evaluation group")
        os.killpg(process["pgid"], signal.SIGKILL)
    if group_members(process["pgid"]):
        raise RuntimeError("old evaluation workers did not exit")


def drain_successor_evaluation(active):
    members = group_members(active["process"].pid)
    if not members:
        return
    for member in members:
        if (member["sid"] != active["process"].pid
                or not any(Path(part).name == "evaluate_native_checkpoint_288.py" for part in member["command"])
                or active["item"]["output"] not in member["command"]):
            raise RuntimeError("unexpected process in successor evaluation group")
        require_identity(member)
    os.killpg(active["process"].pid, signal.SIGTERM)
    until = time.monotonic() + 10
    while group_members(active["process"].pid) and time.monotonic() < until:
        time.sleep(0.1)
    survivors = group_members(active["process"].pid)
    if survivors:
        if any(member not in members for member in survivors):
            raise RuntimeError("evaluation group changed during termination")
        os.killpg(active["process"].pid, signal.SIGKILL)
        until = time.monotonic() + 2
        while group_members(active["process"].pid) and time.monotonic() < until:
            time.sleep(0.05)
    if group_members(active["process"].pid):
        raise RuntimeError("successor evaluation group did not exit")


def stop_trainer_at_campaign_wall(trainer):
    if identity(trainer["pid"]) != trainer:
        return
    require_identity(trainer)
    os.kill(trainer["pid"], signal.SIGTERM)
    until = time.monotonic() + 10
    while identity(trainer["pid"]) == trainer and time.monotonic() < until:
        time.sleep(0.1)
    if identity(trainer["pid"]) == trainer:
        require_identity(trainer)
        os.kill(trainer["pid"], signal.SIGKILL)


def interrupted_cost(item):
    directory = Path(item["output"])
    files = {str(path): sha(path) for path in sorted(directory.rglob("*")) if path.is_file()}
    rows = [read(path) for path in sorted(directory.glob("[0-9][0-9][0-9][0-9].json"))]
    choices = sum(sum(choice["learner"] for choice in row.get("choices", [])) for row in rows)
    result = directory / "evaluation-result.json"
    return {"evaluation": item, "preserved_files": files, "completed_game_receipts": len(rows),
            "recorded_learner_choices_lower_bound": choices,
            "charged_learner_choice_bounds": [choices, 100000],
            "complete_original_result": read(result) if result.exists() else None,
            "frozen_player_unchanged": None,
            "scope": "Interrupted original576 evaluation. All costs remain charged. Active-call and startup costs may be unknown. No partial outcomes select the amended assignments or checkpoint."}


def run(args):
    plan = validate(args)
    if not args.takeover:
        print(json.dumps({key: value for key, value in plan.items() if key != "preflight_status"}, sort_keys=True))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "takeover-manifest.json", plan)
    write(args.output / "original-status-snapshot.json", plan["preflight_status"])
    parent = read(args.campaign / "campaign-manifest.json")
    protocol = parent["protocol"]
    supervisor, trainer = plan["supervisor_identity"], plan["trainer_identity"]
    stopped = retired = False
    try:
        require_identity(supervisor)
        require_identity(trainer)
        os.kill(supervisor["pid"], signal.SIGSTOP)
        stopped = True
        until = time.monotonic() + 2
        while time.monotonic() < until:
            if Path(f"/proc/{supervisor['pid']}/stat").read_text().split(") ", 1)[1][0] in "Tt":
                break
            time.sleep(0.02)
        else:
            raise RuntimeError("old supervisor did not stop")
        require_identity(supervisor)
        require_identity(trainer)
        state = read(args.campaign / "campaign-status.json")
        write(args.output / "stopped-status-snapshot.json", state)
        item = state["active_evaluation"]
        if item:
            process = identity(item["pid"])
            if process:
                if process != plan["active_evaluator_identity"]:
                    raise RuntimeError("active evaluator changed since preflight")
                require_identity(supervisor)
                require_identity(trainer)
                stop_evaluation(process, args.output)
            elif group_members(item["pid"]):
                raise RuntimeError("old evaluation parent exited while workers remain; retry preflight")
        interrupted_items = [row for row in state["completed_evaluations"]
                             if row.get("result") is None and Path(row.get("output", ""), "manifest.json").exists()]
        if item:
            interrupted_items.append(item)
        write(args.output / "interrupted-evaluation-cost.json", {
              "evaluations": [interrupted_cost(row) for row in interrupted_items],
              "scope": "Preserved original evaluator costs, separate from amended evaluations."})
        if campaign_evaluators(args.campaign):
            raise RuntimeError("campaign evaluation workers remain before adoption")
        for path, expected in {**read(args.bindings)["artifacts"], **plan["source_artifacts"]}.items():
            if sha(path) != expected:
                raise ValueError("amendment source changed during handoff")
        require_identity(supervisor)
        require_identity(trainer)
        os.kill(supervisor["pid"], signal.SIGKILL)
        retired = True
        require_identity(trainer)
        write(args.output / "takeover-verified.json", {"trainer_identity_unchanged": True,
              "old_supervisor_retired": True, "trainer_identity": identity(trainer["pid"]),
              "campaign_elapsed_seconds": time.monotonic() - plan["original_started_monotonic"]})
        supervise(args, plan, state, protocol)
    except Exception as error:
        write(args.output / "supervisor-failure.json", {"error": f"{type(error).__name__}: {error}",
              "old_supervisor_retired": retired, "trainer_identity": identity(trainer["pid"]),
              "campaign_elapsed_seconds": time.monotonic() - plan["original_started_monotonic"]})
        if retired:
            while identity(trainer["pid"]) == trainer and time.monotonic() < plan["original_started_monotonic"] + protocol["limits"]["wall_seconds"]:
                time.sleep(5)
            if identity(trainer["pid"]) == trainer:
                stop_trainer_at_campaign_wall(trainer)
        raise
    finally:
        if stopped and not retired and identity(supervisor["pid"]) == supervisor:
            os.kill(supervisor["pid"], signal.SIGCONT)


def supervise(args, plan, state, protocol):
    os.nice(max(0, 19 - os.nice(0)))
    trainer = plan["trainer_identity"]
    started = plan["original_started_monotonic"]
    maximum = protocol["evaluation"]["max_evaluation_checkpoints"]
    evaluations = [row for row in state["completed_evaluations"] if row.get("result") is not None]
    pending = [row for row in state["completed_evaluations"] if row.get("result") is None
               and Path(row.get("output", ""), "manifest.json").exists()] + list(state["pending_evaluations"])
    if state["active_evaluation"]:
        pending.insert(0, state["active_evaluation"])
    known = {item["weights_sha256"]: item for item in evaluations + pending}
    snapshots = {label for item in known.values() for label in item["labels"]}
    reports, active, final_requested = set(), None, False
    def request(pointer, label):
        if not pointer.exists():
            return False
        descriptor = read(pointer)
        key = descriptor["weights_sha256"]
        if key in known:
            if label not in known[key]["labels"]:
                known[key]["labels"].append(label)
            return True
        if len(known) >= maximum:
            raise ValueError("original seven-checkpoint family exhausted")
        number = len(known)
        receipt = args.output / f"evaluation-request-{number:02d}.json"
        write(receipt, descriptor)
        item = {"number": number, "labels": [label], "weights_sha256": key,
                "checkpoint_receipt": str(receipt), "requested_campaign_seconds": time.monotonic() - started}
        known[key] = item
        pending.append(item)
        return True
    def interrupted(signum, frame):
        raise InterruptedError(f"successor supervisor received signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        while True:
            elapsed = time.monotonic() - started
            current_trainer = identity(trainer["pid"])
            if current_trainer is not None and current_trainer != trainer:
                raise RuntimeError("adopted trainer process identity changed")
            training_result = args.campaign / "training/training-result.json"
            if not training_result.exists():
                training_result = args.campaign / "training/result.json"
            terminal_training = read(training_result) if training_result.exists() else None
            training_code = None if current_trainer else "not_observed_adopted_nonchild_exit"
            training_status = terminal_training.get("status") if terminal_training else None
            training_success = training_status in ("completed_bounded_run", "bounded_stop")
            for target in protocol["reporting"]["scheduled_campaign_seconds"]:
                label = f"report_{target}"
                if label not in snapshots and elapsed >= target - protocol["reporting"]["evaluation_lead_seconds"]:
                    if request(args.campaign / "training/latest-checkpoint.json", label):
                        snapshots.add(label)
            if current_trainer is None and not final_requested:
                final_requested = request(args.campaign / "training/latest-checkpoint.json", "final")
            if final_requested:
                for item in pending[:]:
                    if "final" not in item["labels"]:
                        pending.remove(item)
                        evaluations.append(dict(item, status="skipped_to_reserve_final_evaluation"))
            if active and active["process"].poll() is not None:
                drain_successor_evaluation(active)
                item = active["item"]
                active["log"].close()
                result_path = Path(item["output"]) / "evaluation-result.json"
                item.update(exit_code=active["process"].returncode, finished_campaign_seconds=elapsed,
                            result=read(result_path) if result_path.exists() else None)
                evaluations.append(item)
                active = None
            if active is None and pending and elapsed < protocol["limits"]["wall_seconds"] and (elapsed < protocol["limits"]["training_stop_campaign_seconds"] or final_requested):
                if campaign_evaluators(args.campaign):
                    raise RuntimeError("another campaign evaluation worker exists before launch")
                item = pending.pop(0)
                item["output"] = str(args.output / f"evaluation-{item['number']:02d}")
                deadline = started + protocol["limits"]["wall_seconds" if "final" in item["labels"] else "training_stop_campaign_seconds"]
                command = [sys.executable, str(ROOT / "scripts/evaluate_native_checkpoint_288.py"),
                           "--protocol", plan["protocol_path"], "--amendment", str(args.amendment),
                           "--bindings", str(args.bindings), "--checkpoint-receipt", item["checkpoint_receipt"],
                           "--campaign-evaluation-deadline-monotonic", str(deadline), "--output", item["output"]]
                for label, flag in (("initial", "initial"), ("pre_rl", "pre-rl")):
                    reference = next((row for row in evaluations if label in row["labels"] and row.get("result")), None)
                    if reference:
                        command += [f"--{flag}-checkpoint-receipt", reference["checkpoint_receipt"],
                                    f"--{flag}-evaluation-receipt", str(Path(reference["output"]) / "evaluation-result.json")]
                for path, expected in plan["source_artifacts"].items():
                    if sha(path) != expected:
                        raise ValueError("successor source changed before evaluation")
                log = (args.output / f"evaluation-{item['number']:02d}.log").open("xb")
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                item["pid"] = process.pid
                active = {"process": process, "log": log, "item": item}
            progress = read(args.campaign / "training/progress.json")
            status = {"schema": "native-sustained-campaign288-status/1", "campaign": str(args.campaign),
                      "original_campaign_manifest_sha256": plan["parent_manifest_sha256"],
                      "amendment_sha256": plan["amendment_sha256"], "supervisor_pid": os.getpid(),
                      "training_pid": trainer["pid"], "training_exit_code": training_code,
                      "training_terminal_status": training_status,
                      "campaign_elapsed_seconds": elapsed, "training_progress": progress,
                      "active_evaluation": active["item"] if active else None,
                      "pending_evaluations": pending, "completed_evaluations": evaluations,
                      "historical_interrupted_cost_receipt": str(args.output / "interrupted-evaluation-cost.json"),
                      "training_updates_trigger_reports": False}
            atomic(args.output / "campaign-status.json", status)
            for target in protocol["reporting"]["scheduled_campaign_seconds"]:
                if elapsed >= target and target not in reports:
                    write(args.output / f"report-{target:05d}.json", dict(status, scheduled_campaign_seconds=target))
                    reports.add(target)
            if elapsed >= protocol["limits"]["wall_seconds"] or (current_trainer is None and final_requested and active is None and not pending):
                complete = any("final" in row["labels"] and row.get("exit_code") == 0
                               and (row.get("result") or {}).get("status") == "complete"
                               and row["result"].get("frozen_player_unchanged") is True
                               and row["result"].get("checkpoint_files_unchanged") is True for row in evaluations)
                write(args.output / "campaign-result.json", dict(status, schema="native-sustained-campaign288-result/1",
                      status="training_fault" if training_status == "fault" else "bounded_campaign_finished"
                             if complete and training_success else "final_evaluation_incomplete",
                      final_evaluation_complete=complete))
                break
            time.sleep(5)
    finally:
        if active:
            drain_successor_evaluation(active)
            active["process"].wait(timeout=10)
            active["log"].close()
        current = identity(trainer["pid"])
        if current == trainer and time.monotonic() - started >= protocol["limits"]["wall_seconds"]:
            stop_trainer_at_campaign_wall(trainer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--takeover", action="store_true")
    args = parser.parse_args()
    for key in ("campaign", "bindings", "amendment", "output"):
        setattr(args, key, getattr(args, key).resolve())
    run(args)


if __name__ == "__main__":
    main()
