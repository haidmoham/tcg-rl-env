"""Independently inspect completed sustained-pilot tensors without games or updates."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WALL_SECONDS = 1800
TEACHER_QUERY_CAP = 2000


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


@dataclass(frozen=True)
class Finding:
    requirement: str
    status: str
    evidence: object


class Audit:
    def __init__(self):
        self.started = time.monotonic()
        self.findings = []
        self.teacher_queries = 0
        self.teacher_seconds = 0.0
        self.model_loads = 0
        self.frames_checked = 0

    def check(self, name, passed, evidence):
        self.findings.append(Finding(name, "pass" if passed else "fail", evidence))

    def missing(self, name, reason):
        self.findings.append(Finding(name, "not_yet_evidenced", reason))

    def tick(self):
        if time.monotonic() - self.started >= WALL_SECONDS:
            raise TimeoutError("independent tensor audit wall cap exhausted")


@contextmanager
def wall_cap(seconds):
    def expired(signum, frame):
        raise TimeoutError("independent tensor audit hard wall cap exhausted")
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def live_campaign_processes(campaign, transition=None):
    targets = [str(campaign).encode()]
    if transition is not None:
        targets.append(str(transition).encode())
    names = (b"run_native_campaign.py", b"resume_native_campaign_288.py", b"train_native_sustained.py",
             b"evaluate_native_checkpoint.py", b"evaluate_native_checkpoint_288.py")
    active = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
            state = (entry / "stat").read_text().rsplit(")", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            continue
        if state != "Z" and any(target in command for target in targets) and any(name in command for name in names):
            active.append(int(entry.name))
    return active


def terminal_receipt(campaign, transition):
    parent_sha = sha(campaign / "campaign-manifest.json")
    if transition is not None:
        takeover = read(transition / "takeover-manifest.json")
        if Path(takeover["campaign"]).resolve() != campaign or takeover["parent_manifest_sha256"] != parent_sha:
            raise ValueError("takeover manifest does not identify the original campaign")
        path = transition / "campaign-result.json"
        result = read(path)
        if (result.get("schema") != "native-sustained-campaign288-result/1"
                or Path(result["campaign"]).resolve() != campaign
                or result["original_campaign_manifest_sha256"] != parent_sha
                or result["training_pid"] != takeover["trainer_identity"]["pid"]
                or result["amendment_sha256"] != takeover["amendment_sha256"]):
            raise ValueError("takeover terminal receipt lineage differs")
        if not takeover["source_artifacts"] or any(sha(source) != expected for source, expected in takeover["source_artifacts"].items()):
            raise ValueError("takeover source identities differ")
    else:
        path = campaign / "campaign-result.json"
        result = read(path)
    if result["status"] not in {"campaign_wall_cap", "training_fault", "bounded_campaign_finished", "final_evaluation_incomplete"}:
        raise ValueError("campaign receipt does not describe a terminal outcome")
    return path


def validate_sources(bindings):
    artifacts = bindings["artifacts"]
    if hashlib.sha256(canonical(artifacts)).hexdigest() != bindings["artifact_manifest_sha256"]:
        raise ValueError("bound source closure digest differs")
    for path, expected in artifacts.items():
        if sha(path) != expected:
            raise ValueError(f"bound artifact changed: {Path(path).name}")
    for key, field in (("engine_library", "engine_sha256"), ("teacher_checkpoint", "teacher_sha256"),
                       ("deck_manifest", "deck_manifest_sha256")):
        if sha(bindings[key]) != bindings[field] or artifacts.get(bindings[key]) != bindings[field]:
            raise ValueError(f"bound {key} differs")


def inputs(args):
    if os.name != "posix" or not Path("/proc").is_dir():
        raise RuntimeError("use the pinned Linux/WSL CPU runtime")
    if (args.output.is_relative_to(ROOT) or args.output.is_relative_to(args.campaign)
            or args.transition is not None and args.output.is_relative_to(args.transition)):
        raise ValueError("audit output must be private and outside the immutable campaign")
    bindings = read(args.bindings)
    validate_sources(bindings)
    training = args.campaign / "training"
    manifest = read(training / "manifest.json")
    campaign = read(args.campaign / "campaign-manifest.json")
    protocol, lineage = manifest["protocol"], manifest["lineage"]
    if (protocol["schema"] != "86-tcg-agent/native-sustained/1" or protocol["sequence"] != [1, 3, 4]
            or protocol["skipped_stage"] != 2 or manifest["source_artifact_manifest"] != bindings["artifacts"]
            or not sha(args.bindings) == lineage["bindings_sha256"] == campaign["bindings_sha256"]):
        raise ValueError("campaign, bindings and protocol disagree")
    if (hashlib.sha256(canonical(protocol)).hexdigest() != lineage["protocol_sha256"]
            or campaign["protocol"] != protocol or campaign["protocol_sha256"] != lineage["protocol_file_sha256"]
            or bindings["artifact_manifest_sha256"] != lineage["source_artifact_manifest_sha256"]):
        raise ValueError("campaign protocol/source identities disagree")
    latest = read(training / "latest-checkpoint.json")
    end = int(latest["checkpoint_id"].removeprefix("checkpoint-"))
    checkpoints = []
    fixed_lineage = ("protocol_sha256", "protocol_file_sha256", "bindings_sha256", "source_artifact_manifest_sha256",
                     "teacher_sha256", "engine_sha256", "deck_manifest_sha256")
    for index in range(end + 1):
        path = training / f"checkpoint-{index:06d}.json"
        row = read(path)
        if (row["status"] != "completed" or row["checkpoint_id"] != path.stem
                or row["lineage"]["protocol_sha256"] != lineage["protocol_sha256"]
                or any(row["lineage"][key] != lineage[key] for key in fixed_lineage)
                or row["lineage"]["source_artifact_manifest_sha256"] != bindings["artifact_manifest_sha256"]):
            raise ValueError("immutable checkpoint lineage differs")
        for kind in ("weights", "state"):
            if sha(row[kind + "_path"]) != row[kind + "_sha256"]:
                raise ValueError(f"checkpoint {kind} changed")
        checkpoints.append(row)
    if ({key: value for key, value in latest.items() if key != "receipt_sha256"} != checkpoints[-1]
            or sha(latest["receipt_path"]) != latest["receipt_sha256"]):
        raise ValueError("latest pointer differs from completed receipt")
    if args.mode == "audit":
        active = live_campaign_processes(args.campaign, args.transition)
        if active:
            raise RuntimeError(f"campaign runtime processes are still live: {active}")
        terminal_receipt(args.campaign, args.transition)
    return bindings, protocol, checkpoints


def model_load(audit, native, row):
    audit.tick()
    model = native.load_student(row["weights_path"], expected_sha256=row["weights_sha256"])
    audit.model_loads += 1
    if native.state_digest(model) != row["model_identity"]["state_digest"]:
        raise ValueError("reloaded checkpoint tensor identity differs from receipt")
    return model


def tensor_changes(torch, before, after):
    left, right = before.state_dict(), after.state_dict()
    if left.keys() != right.keys():
        raise ValueError("model tensor names differ")
    return [name for name in left if not torch.equal(left[name], right[name])]


def frame_check(torch, audit, frame, schema):
    audit.tick()
    if frame.schema_digest != schema or frame.real_rows != 1 or len(frame.records) != 1:
        raise ValueError("stored frame schema/row count differs")
    digest = hashlib.sha256()
    digest.update(json.dumps([frame.schema_digest, frame._seq_len,
                              [asdict(record) for record in frame.records]], sort_keys=True).encode())
    for array in frame._arrays:
        if not isinstance(array.data, bytes):
            raise ValueError("stored frame does not own immutable bytes")
        digest.update(json.dumps([array.name, array.dtype, array.shape]).encode())
        digest.update(array.data)
    if digest.hexdigest() != frame.digest:
        raise ValueError("stored copied-frame digest differs")
    record = frame.records[0]
    legal = record.eligible_columns
    if (not legal or tuple(sorted(set(legal))) != legal or min(legal) < 0
            or max(legal) > record.option_count or (not record.can_submit and max(legal) == record.option_count)):
        raise ValueError("stored physical action columns are invalid")
    audit.frames_checked += 1


def forward(torch, model, frame):
    with torch.no_grad():
        logits, raw = model(frame._payload(), validate=True)
        values = raw.tanh()
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[0, list(frame.records[0].eligible_columns)] = True
    if (logits.shape[0] != 1 or values.shape != (1,) or not torch.equal(logits.isfinite(), mask)
            or not torch.isneginf(logits[~mask]).all() or not values.isfinite().all()):
        raise ValueError("independent model forward violates legal/value contract")
    return logits[0], float(values[0]), mask[0]


def close(left, right, tolerance=1e-6):
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def label_metrics(torch, audit, model, labels, schema):
    rows = []
    for frame, target in labels:
        frame_check(torch, audit, frame, schema)
        logits, _, legal = forward(torch, model, frame)
        if (target.shape != logits.shape or not target.isfinite().all() or (target < 0).any()
                or not torch.equal(target[~legal], torch.zeros_like(target[~legal]))
                or not close(float(target.sum()), 1.0) or not (target[legal] > 0).all()):
            raise ValueError("teacher label is not a full legal probability distribution")
        q = target[legal]
        ce = float(-(q * logits[legal].log_softmax(-1)).sum())
        entropy = float(-(q * q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum())
        rows.append({"frame_digest": frame.digest, "cross_entropy": ce, "forward_kl": ce - entropy,
                     "top1_agreement": int(logits.argmax() == target.argmax())})
    return {"count": len(rows), "rows": rows,
            "mean_cross_entropy": sum(row["cross_entropy"] for row in rows) / len(rows),
            "mean_forward_kl": sum(row["forward_kl"] for row in rows) / len(rows),
            "top1_agreement": sum(row["top1_agreement"] for row in rows) / len(rows)}


def compare_metrics(audit, name, actual, stored):
    audit.check(name, actual["count"] == stored["count"]
                and len(actual["rows"]) == len(stored["rows"])
                and all(close(actual[key], stored[key]) for key in ("mean_cross_entropy", "mean_forward_kl", "top1_agreement"))
                and all(left["frame_digest"] == right["frame_digest"]
                        and all(close(left[key], right[key]) for key in ("cross_entropy", "forward_kl", "top1_agreement"))
                        for left, right in zip(actual["rows"], stored["rows"])),
                {key: value for key, value in actual.items() if key != "rows"})


def optimizer_audit(torch, audit, checkpoints, load):
    local_steps, previous_total = {1: 0, 3: 0, 4: 0}, 0
    for row in checkpoints:
        audit.tick()
        state = load(row["state_path"])
        valid = state["lineage"] == row["lineage"] and state["progress"]["optimizer_steps_total"] == row["optimizer_steps_total"]
        rng = state["torch_rng"]
        generator = torch.Generator(device="cpu")
        generator.set_state(rng)
        valid = valid and torch.equal(generator.get_state(), rng)
        if row["stage"] == 0:
            valid = valid and state["optimizer"] is None
        else:
            stage = row["stage"]
            local_steps[stage] += row["optimizer_steps_total"] - previous_total
            optimizer = state["optimizer"]
            steps = [float(value["step"]) for value in optimizer["state"].values()]
            valid = valid and bool(steps) and all(step == local_steps[stage] for step in steps)
            valid = valid and all(group["lr"] == 1e-4 and group["weight_decay"] == 0 for group in optimizer["param_groups"])
            valid = valid and bool(state["rng_states"])
            for name, saved in state["rng_states"].items():
                if name == "query_rng":
                    random.Random().setstate(saved)
                elif name == "last_episode_rng":
                    for rng_state in saved.values():
                        restored = torch.Generator(device="cpu")
                        restored.set_state(rng_state)
                        valid = valid and torch.equal(restored.get_state(), rng_state)
        previous_total = row["optimizer_steps_total"]
        audit.check(f"{row['checkpoint_id']}: optimizer steps and saved RNG state", valid,
                    {"optimizer_steps_total": row["optimizer_steps_total"], "stage": row["stage"],
                     "stage_optimizer_steps": local_steps.get(row["stage"], 0),
                     "scope": "States deserialized and generator state restored without optimizer or active-game resume."})


def stage1(torch, native, audit, training, protocol, checkpoints, load, schema, args, bindings):
    rows = [row for row in checkpoints if row["stage"] == 1]
    if not rows:
        audit.missing("stage 1 tensor audit", "No completed policy-transfer checkpoint")
        return None
    initial = model_load(audit, native, checkpoints[0])
    fresh = native.build_student(native.SmallNativeConfig(**protocol["student"]), seed=protocol["seeds"]["student_init"])
    audit.check("fresh independent initial state matches declared seed", not tensor_changes(torch, fresh, initial),
                {"seed": protocol["seeds"]["student_init"], "checkpoint": checkpoints[0]["weights_sha256"]})
    del fresh
    probe_file = training / "stage1-heldout-probes.pt"
    if sha(probe_file) != rows[-1]["lineage"]["probe_data_sha256"]:
        raise ValueError("heldout probe bytes changed")
    labels, probes = [], load(probe_file)
    probe_ids = {frame.digest for frame, _ in probes}
    actor_names = {name for name, _ in initial.named_parameters() if not name.startswith("value_head.")}
    previous = initial
    for row in rows:
        path = training / f"stage1-update-{row['stage_updates']:04d}-labels.pt"
        if sha(path) != row["lineage"]["data_sha256"]:
            raise ValueError("stage 1 labels changed")
        batch = load(path)
        current = model_load(audit, native, row)
        changed = tensor_changes(torch, previous, current)
        audit.check(f"{row['checkpoint_id']}: policy-only parameter changes",
                    bool(changed) and set(changed) <= actor_names,
                    {"changed_actor_tensors": len(changed), "critic_and_buffers_unchanged": set(changed) <= actor_names})
        for frame, target in batch:
            frame_check(torch, audit, frame, schema)
            _, _, legal = forward(torch, initial, frame)
            if (target.ndim != 1 or target.numel() != legal.numel() or not target.isfinite().all()
                    or (target < 0).any() or not (target[legal] > 0).all()
                    or not torch.equal(target[~legal], torch.zeros_like(target[~legal]))
                    or not close(float(target.sum()), 1.0)):
                raise ValueError("stored training label violates legal probability support")
        audit.check(f"{row['checkpoint_id']}: fixed label batch and probe identity exclusion",
                    len(batch) == protocol["stage1"]["labels_per_update"]
                    and not any(frame.digest in probe_ids for frame, _ in batch), {"labels": len(batch)})
        labels.extend(batch)
        previous = current
    audit.check("heldout policy probe size", len(probes) == protocol["stage1"]["probe_frames"], {"frames": len(probes)})
    before = label_metrics(torch, audit, initial, probes, schema)
    after = label_metrics(torch, audit, previous, probes, schema)
    compare_metrics(audit, "independently reforwarded initial heldout policy metrics", before, read(training / "stage1-probe-before.json"))
    compare_metrics(audit, "independently reforwarded transferred heldout policy metrics", after, read(training / "stage1-probe-after.json"))
    reloaded = model_load(audit, native, rows[-1])
    exact_reload = all(torch.equal(forward(torch, previous, frame)[0], forward(torch, reloaded, frame)[0]) for frame, _ in probes)
    audit.check("transfer checkpoint reload actor logits exact", exact_reload, {"frames": len(probes)})
    if args.recompute_teacher_targets:
        if len(labels) + len(probes) > TEACHER_QUERY_CAP:
            raise ValueError("teacher target audit exceeds declared query cap")
        audit.tick()
        teacher = native.load_teacher(bindings["teacher_checkpoint"], expected_sha256=bindings["teacher_sha256"])
        audit.model_loads += 1
        digest_before = native.state_digest(teacher)
        maximum_error = 0.0
        for frame, target in labels + list(probes):
            audit.tick()
            start = time.monotonic()
            logits, _, _ = forward(torch, teacher, frame)
            q = logits.softmax(-1)
            audit.teacher_seconds += time.monotonic() - start
            audit.teacher_queries += 1
            maximum_error = max(maximum_error, float((q - target).abs().max()))
        audit.check("independent frozen teacher-target replay", maximum_error <= 1e-6
                    and native.state_digest(teacher) == digest_before,
                    {"queries": audit.teacher_queries, "maximum_probability_error": maximum_error,
                     "scope": "Read-only post-run target replay. Does not observe the historical teacher process's memory."})
        del teacher
    else:
        audit.missing("independent teacher-target recomputation", "Use --recompute-teacher-targets after campaign termination; at most 2,000 read-only calls.")
    return rows[-1]


def critic_metrics(torch, model, examples, constant):
    rows = []
    for frame, target in examples:
        _, prediction, _ = forward(torch, model, frame)
        rows.append({"frame_digest": frame.digest, "prediction": prediction, "target": target})
    return {"count": len(rows), "mse": sum((row["prediction"] - row["target"]) ** 2 for row in rows) / len(rows),
            "mean_bias": sum(row["prediction"] - row["target"] for row in rows) / len(rows),
            "constant_mse": sum((constant - row["target"]) ** 2 for row in rows) / len(rows), "rows": rows}


def stage3(torch, native, audit, training, protocol, checkpoints, load, schema, transferred):
    rows = [row for row in checkpoints if row["stage"] == 3]
    if transferred is None or not rows:
        audit.missing("stage 3 tensor audit", "No complete transfer/calibration boundary")
        return
    before = model_load(audit, native, transferred)
    examples_path = training / "stage3-terminal-examples.pt"
    if sha(examples_path) != rows[-1]["lineage"]["data_sha256"]:
        raise ValueError("calibration example bytes changed")
    examples = load(examples_path)
    for split, domain in (("fit", "calibration"), ("heldout", "calibration_heldout")):
        receipts = [json.loads(line) for line in (training / f"stage3-{domain}-episodes.jsonl").read_bytes().splitlines()]
        expected = []
        offset = 0
        for row in receipts:
            if row["status"] != "terminal" or row["winner"] not in (0, 1, 2):
                raise ValueError("calibration includes a non-natural outcome")
            outcome = 0.0 if row["winner"] == 2 else (1.0 if row["winner"] == row["learner_seat"] else -1.0)
            if outcome != row["outcome"]:
                raise ValueError("calibration acting-seat perspective differs")
            expected.extend([outcome] * row["retained_learner_steps"])
            segment = examples[split][offset:offset + row["retained_learner_steps"]]
            if any(frame.records[0].select_player != row["learner_seat"] for frame, _ in segment):
                raise ValueError("calibration frame acting seat differs from episode receipt")
            offset += row["retained_learner_steps"]
        audit.check(f"calibration {split}: outcome MC targets and natural games",
                    [target for _, target in examples[split]] == expected
                    and len(receipts) == protocol["calibration"]["training_games" if split == "fit" else "heldout_games"],
                    {"games": len(receipts), "retained_frames": len(expected)})
        for frame, _ in examples[split]:
            frame_check(torch, audit, frame, schema)
    fixed_logits = [forward(torch, before, frame)[0] for frame, _ in examples["heldout"]]
    critic_names = {name for name, _ in before.named_parameters() if name.startswith("value_head.")}
    previous = before
    for row in rows:
        if sha(training / "stage3-terminal-examples.pt") != row["lineage"]["data_sha256"]:
            raise ValueError("calibration example bytes changed")
        current = model_load(audit, native, row)
        changed = tensor_changes(torch, previous, current)
        audit.check(f"{row['checkpoint_id']}: critic-only tensor changes",
                    bool(changed) and set(changed) <= critic_names,
                    {"changed_critic_tensors": len(changed), "actor_encoder_buffers_unchanged": set(changed) <= critic_names})
        previous = current
    exact = all(torch.equal(logits, forward(torch, previous, frame)[0])
                for logits, (frame, _) in zip(fixed_logits, examples["heldout"]))
    audit.check("all heldout calibration actor logits unchanged", exact, {"frames": len(fixed_logits)})
    constant = sum(target for _, target in examples["fit"]) / len(examples["fit"])
    stored = read(training / "stage3-result.json")
    for label, model in (("heldout_before", before), ("heldout_after", previous)):
        actual = critic_metrics(torch, model, examples["heldout"], constant)
        valid = actual["count"] == stored[label]["count"] and close(constant, stored[label]["constant_training_mean"])
        valid = valid and all(close(actual[key], stored[label][key]) for key in ("mse", "mean_bias", "constant_mse"))
        valid = valid and len(actual["rows"]) == len(stored[label]["rows"]) and all(
            a["frame_digest"] == b["frame_digest"] and a["target"] == b["target"] and close(a["prediction"], b["prediction"])
            for a, b in zip(actual["rows"], stored[label]["rows"]))
        audit.check(f"independent critic forward {label}", valid, {key: value for key, value in actual.items() if key != "rows"})


def stage4(torch, native, audit, training, protocol, checkpoints, load, schema):
    from tcg.native_actorcritic import terminal_targets
    rows = [row for row in checkpoints if row["stage"] == 4]
    if not rows:
        audit.missing("completed PPO tensor audit", "No completed PPO update checkpoint")
        return
    indexed = {row["weights_sha256"]: row for row in checkpoints}
    receipts = {str(row["engine_seed"]): row for row in map(json.loads, (training / "stage4-episodes.jsonl").read_bytes().splitlines())}
    for row in rows:
        parent = indexed[row["lineage"]["parent_checkpoint_sha256"]]
        before = model_load(audit, native, parent)
        after = model_load(audit, native, row)
        data = training / f"stage4-update-{row['stage_updates']:04d}-episodes.pt"
        if sha(data) != row["lineage"]["data_sha256"]:
            raise ValueError("PPO episode bytes changed")
        episodes = load(data)
        steps, advantages, returns = [], [], []
        logp_error = value_error = recurrence_error = 0.0
        target_digest = hashlib.sha256()
        parent_digest = native.state_digest(before)
        for episode in episodes:
            audit.tick()
            receipt = receipts[episode.episode_id]
            outcome = 0.0 if receipt["winner"] == 2 else (1.0 if receipt["winner"] == episode.learner_seat else -1.0)
            if (receipt["status"] != "terminal" or receipt["winner"] not in (0, 1, 2)
                    or episode.outcome != outcome or receipt["outcome"] != outcome
                    or episode.policy_digest != parent_digest or receipt["policy_digest"] != parent_digest
                    or len(episode.steps) != receipt["retained_learner_steps"]):
                raise ValueError("PPO batch contains mixed policies or non-natural/partial episodes")
            deltas = []
            for index, step in enumerate(episode.steps):
                frame_check(torch, audit, step.frame, schema)
                record = step.frame.records[0]
                if (record.select_player != episode.learner_seat or step.action not in record.eligible_columns
                        or step.genuine != (len(record.eligible_columns) > 1)):
                    raise ValueError("PPO step violates stored actor/legal-column contract")
                logits, value, _ = forward(torch, before, step.frame)
                logp_error = max(logp_error, abs(float(logits.log_softmax(-1)[step.action]) - step.old_logp))
                value_error = max(value_error, abs(value - step.old_value))
                next_value = episode.steps[index + 1].old_value if index + 1 < len(episode.steps) else 0.0
                reward = episode.outcome if index + 1 == len(episode.steps) else 0.0
                deltas.append(reward + protocol["ppo"]["gamma"] * next_value - step.old_value)
            local = [sum((protocol["ppo"]["gamma"] * protocol["ppo"]["gae_lambda"]) ** offset * delta
                         for offset, delta in enumerate(deltas[index:])) for index in range(len(deltas))]
            recursive, carry = [], 0.0
            for delta in reversed(deltas):
                carry = delta + protocol["ppo"]["gamma"] * protocol["ppo"]["gae_lambda"] * carry
                recursive.append(carry)
            recurrence_error = max(recurrence_error, max(abs(a - b) for a, b in zip(local, reversed(recursive))))
            steps.extend(episode.steps)
            advantages.extend(local)
            returns.extend([episode.outcome] * len(episode.steps))
        genuine = [advantage for step, advantage in zip(steps, advantages) if step.genuine]
        mean = sum(genuine) / len(genuine) if genuine else 0.0
        variance = sum((value - mean) ** 2 for value in genuine) / len(genuine) if genuine else 0.0
        normalized = [(value - mean) / (variance ** 0.5 + 1e-8) if step.genuine else 0.0
                      for step, value in zip(steps, advantages)]
        actual_steps, actual_advantages, actual_returns = terminal_targets(episodes)
        targets_error = max(abs(a - b) for a, b in zip(normalized, actual_advantages))
        target_rows_match = len(actual_steps) == len(steps) and len(actual_advantages) == len(normalized) and all(
            a.frame.digest == b.frame.digest and a.action == b.action for a, b in zip(steps, actual_steps))
        for step, advantage, target in zip(steps, normalized, returns):
            target_digest.update(canonical([step.frame.digest, step.action, advantage, target]))
        changed = tensor_changes(torch, before, after)
        parameters = dict(before.named_parameters())
        actor_changed = [name for name in changed if name in parameters and not name.startswith("value_head.")]
        critic_changed = [name for name in changed if name.startswith("value_head.")]
        buffers_changed = [name for name in changed if name not in parameters]
        audit.check(f"{row['checkpoint_id']}: frozen-rollout old logits/values and complete batch",
                    logp_error <= 1e-6 and value_error <= 1e-6
                    and len(steps) >= protocol["ppo"]["target_learner_microsteps"]
                    and len(steps) == row["lineage"]["terminal_batch_steps"]
                    and len(episodes) == row["lineage"]["terminal_episodes"]
                    and row["lineage"]["input_policy_digest"] == parent_digest,
                    {"steps": len(steps), "natural_episodes": len(episodes), "max_old_logp_error": logp_error,
                     "max_old_value_error": value_error})
        audit.check(f"{row['checkpoint_id']}: actual actor/critic changes and fixed buffers",
                    bool(actor_changed) and bool(critic_changed) and not buffers_changed,
                    {"changed_actor_tensors": len(actor_changed), "changed_critic_tensors": len(critic_changed),
                     "changed_buffers": buffers_changed})
        audit.check(f"{row['checkpoint_id']}: independent expected GAE and MC reconstruction",
                    recurrence_error < 1e-10 and all(math.isfinite(value) for value in normalized)
                    and target_rows_match and targets_error < 1e-10 and tuple(returns) == actual_returns
                    and all(value == 0 for step, value in zip(steps, normalized) if not step.genuine),
                    {"explicit_td_sum_vs_independent_recurrence_error": recurrence_error,
                     "genuine_steps": len(genuine), "normalization_mean": mean, "normalization_variance": variance,
                     "frozen_production_target_replay_max_advantage_error": targets_error,
                     "production_row_order_and_MC_returns_exact": target_rows_match and tuple(returns) == actual_returns,
                     "expected_targets_sha256": target_digest.hexdigest(),
                     "scope": "Independent explicit TD-sum derivation crosschecked against post-run execution of the bound production target function on the same actual frozen trajectories. Original optimizer inputs were not serialized; this is an executable function replay, not direct observation of those original arrays."})


def reload_boundaries(torch, native, audit, training, checkpoints, load):
    probes = load(training / "stage1-heldout-probes.pt")
    boundaries = [checkpoints[0]]
    for stage in (1, 3, 4):
        rows = [row for row in checkpoints if row["stage"] == stage]
        if rows:
            boundaries.append(rows[-1])
    for row in boundaries:
        first, second = model_load(audit, native, row), model_load(audit, native, row)
        exact = not tensor_changes(torch, first, second)
        for frame, _ in probes:
            audit.tick()
            a_logits, a_value, a_mask = forward(torch, first, frame)
            b_logits, b_value, b_mask = forward(torch, second, frame)
            exact = exact and torch.equal(a_logits, b_logits) and a_value == b_value and torch.equal(a_mask, b_mask)
        audit.check(f"{row['checkpoint_id']}: independent reload tensor/logit/value equivalence", exact,
                    {"frames": len(probes), "stage": row["stage"]})


def run(args):
    audit = Audit()
    bindings, protocol, checkpoints = inputs(args)
    training = args.campaign / "training"
    pinned_paths = {path.resolve() for path in training.iterdir()
                    if path.is_file() and (path.suffix in {".pt", ".jsonl"}
                    or path.name in {"manifest.json", "stage1-result.json", "stage3-result.json",
                                     "stage1-probe-before.json", "stage1-probe-after.json"}
                    or path.name.startswith("checkpoint-") and path.suffix == ".json")}
    pinned_paths.update({args.bindings, args.campaign / "campaign-manifest.json"})
    if args.mode == "audit":
        pinned_paths.add(terminal_receipt(args.campaign, args.transition))
        if args.transition is not None:
            pinned_paths.add(args.transition / "takeover-manifest.json")
    input_pins = {str(path): sha(path) for path in sorted(pinned_paths)}
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "86-tcg-agent/native-sustained-tensor-audit/1", "audit_source_sha256": sha(__file__),
                "mode": args.mode, "bindings_sha256": sha(args.bindings), "protocol_sha256": hashlib.sha256(canonical(protocol)).hexdigest(),
                "source_artifact_manifest_sha256": bindings["artifact_manifest_sha256"],
                "input_artifacts": input_pins, "transition": str(args.transition) if args.transition is not None else None,
                "limits": {"wall_seconds": WALL_SECONDS, "teacher_queries": TEACHER_QUERY_CAP,
                           "cpu_threads": 1, "nice": 19, "gpu": False, "optimizer_updates": 0, "native_games": 0},
                "teacher_target_recomputation": args.recompute_teacher_targets}
    write(args.output / "manifest.json", manifest)
    if args.mode == "validate":
        write(args.output / "validation.json", {"status": "input_validation_passed", "models_loaded": 0,
              "games_started": 0, "campaign_terminal": ((args.transition or args.campaign) / "campaign-result.json").is_file(),
              "live_campaign_pids": live_campaign_processes(args.campaign, args.transition),
              "source_artifacts_verified": len(bindings["artifacts"])})
        print(json.dumps({"status": "input_validation_passed", "models_loaded": 0, "games_started": 0}))
        return 0
    os.nice(max(0, 19 - os.nice(0)))
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", FASTENC_LIB=bindings["engine_library"])
    try:
        with wall_cap(WALL_SECONDS - (time.monotonic() - audit.started)):
            active = live_campaign_processes(args.campaign, args.transition)
            if active:
                raise RuntimeError(f"campaign runtime processes became live before imports: {active}")
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.backends.mkldnn.enabled = False
            sys.path.insert(0, str(ROOT))
            from tcg import native_policy as native
            schema = native.configure_mew(bindings["mew_root"])

            def load(path):
                audit.tick()
                path = Path(path).resolve()
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != input_pins[str(path)]:
                    raise ValueError(f"pinned tensor audit input changed: {path.name}")
                return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)

            training = args.campaign / "training"
            optimizer_audit(torch, audit, checkpoints, load)
            transferred = stage1(torch, native, audit, training, protocol, checkpoints, load, schema, args, bindings)
            stage3(torch, native, audit, training, protocol, checkpoints, load, schema, transferred)
            stage4(torch, native, audit, training, protocol, checkpoints, load, schema)
            reload_boundaries(torch, native, audit, training, checkpoints, load)
            validate_sources(bindings)
            for path, expected in input_pins.items():
                if sha(path) != expected:
                    raise ValueError("pinned tensor audit input changed during execution")
        audit.missing("exact calibration learner-choice count", "Requires separately bounded native replay of the 32 calibration games. Retained-frame counts and both-seat explicit microsteps cannot reconstruct charges.")
        audit.missing("serialized original PPO advantage/return tensors", "The frozen production target function was replayed and crosschecked against an independent TD-sum derivation. Original optimization inputs were not serialized; no optimizer-step replay was performed.")
        audit.missing("native actor visibility and non-consuming live state/RNG/log behavior", "Stored immutable frame bytes and model forwards do not independently reconstruct native hidden worlds or historical live-game state.")
        counts = {status: sum(row.status == status for row in audit.findings) for status in ("pass", "fail", "not_yet_evidenced")}
        result = {"schema": manifest["schema"], "status": "issues" if counts["fail"] else "tensor_checks_pass_with_open_gates",
                  "audit_source_sha256": manifest["audit_source_sha256"], "findings": [asdict(row) for row in audit.findings],
                  "counts": counts, "models_loaded": audit.model_loads, "stored_frames_checked": audit.frames_checked,
                  "teacher_queries": audit.teacher_queries, "teacher_scoring_seconds": audit.teacher_seconds,
                  "audit_wall_seconds": time.monotonic() - audit.started, "native_games_started": 0, "optimizer_updates": 0,
                  "attestation_boundary": "Trainer booleans and bound source describe historical in-memory teacher immutability and target usage. Independent measurements here inspect artifact tensors and run read-only forwards; they do not observe the historical processes."}
        write(args.output / "audit-result.json", result)
        print(json.dumps({"status": result["status"], "counts": counts, "result_sha256": sha(args.output / "audit-result.json")}))
        return int(bool(counts["fail"]))
    except Exception as error:
        write(args.output / "failure.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
              "findings_completed": [asdict(row) for row in audit.findings], "models_loaded": audit.model_loads,
              "teacher_queries": audit.teacher_queries, "audit_wall_seconds": time.monotonic() - audit.started,
              "native_games_started": 0, "optimizer_updates": 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transition", type=Path)
    parser.add_argument("--mode", choices=("validate", "audit"), default="validate")
    parser.add_argument("--recompute-teacher-targets", action="store_true")
    args = parser.parse_args()
    for key in ("campaign", "bindings", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.transition = args.transition.resolve() if args.transition is not None else None
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
