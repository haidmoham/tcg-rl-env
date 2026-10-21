"""Historical staged native learning with committed checkpoint publication."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
import time

from . import native_actorcritic as actorcritic
from .native_distillation import (append_json, canonical, file_hash, fit_labels, label_metrics,
                                  run_episode, seed32, wall_guard)


@dataclass
class TrainingProgress:
    stage: int = 0
    stage_updates: int = 0
    training_updates_total: int = 0
    ppo_updates: int = 0
    optimizer_steps_total: int = 0
    learner_steps_total: int = 0
    episodes: int = 0
    natural_terminal_episodes: int = 0


class SustainedBudget:
    """Adapt both frozen collectors to one clock and one charged choice count."""

    def __init__(self, protocol, progress, *, started, output):
        self.protocol, self.progress, self.started = protocol, progress, started
        declared = protocol["limits"]
        self.training_cap = declared["training_learner_microsteps"]
        self.limits = {**declared, "admitted_learner_microsteps": self.training_cap}
        self.teacher_queries, self.teacher_seconds = 0, 0.0
        self.stage_started, self.stage_seconds = started, declared["stage1_wall_seconds"]
        self.work_deadline = None
        self.readonly_evaluation_steps = 0
        self.stage_receipts = []
        self.output, self.last_progress_at = output, 0.0
        self.inflight_optimizer_steps = 0
        self.update_in_progress = False

    @property
    def admitted(self):
        return self.progress.learner_steps_total

    @admitted.setter
    def admitted(self, value):
        self.progress.learner_steps_total = value
        self.publish_progress()

    @property
    def charged(self):
        return self.admitted

    @charged.setter
    def charged(self, value):
        self.admitted = value

    def enter(self, stage, seconds):
        if self.progress.stage:
            self.stage_receipts.append({"stage": self.progress.stage,
                                        "wall_seconds": time.monotonic() - self.stage_started,
                                        "cumulative_training_learner_steps": self.admitted,
                                        "completed_stage_updates": self.progress.stage_updates})
        self.progress.stage, self.progress.stage_updates = stage, 0
        self.stage_started = self.started if stage == 1 else time.monotonic()
        self.stage_seconds = seconds
        self.work_deadline = min(self.started + self.limits["training_stop_campaign_seconds"], self.stage_started + seconds)
        self.work_deadline -= self.limits["finalization_reserve_seconds"]
        self.publish_progress(force=True)

    def remaining(self, *, reserve=True):
        arm = self.started + min(self.limits["wall_seconds"], self.limits["training_stop_campaign_seconds"])
        phase = self.stage_started + self.stage_seconds
        return min(arm, phase) - time.monotonic() - (self.limits["finalization_reserve_seconds"] if reserve else 0)

    def stop_reason(self):
        if time.monotonic() >= self.started + self.limits["training_stop_campaign_seconds"]:
            return "arm_wall_cap"
        if self.admitted >= self.training_cap:
            return "shared_arm_training_choice_allocation"
        if self.remaining() <= 0:
            return "stage_finalization_reserve"
        return None

    def check(self, *, reserve=True):
        self.publish_progress()
        if self.remaining(reserve=reserve) <= 0 or self.admitted >= self.training_cap:
            raise TimeoutError("shared arm or declared stage allocation exhausted")

    def teacher_available(self):
        return (self.stop_reason() is None
                and self.teacher_queries < self.protocol["stage1"]["teacher_queries"]
                and self.teacher_seconds < self.protocol["stage1"]["teacher_query_seconds"])

    def receipt(self):
        elapsed = time.monotonic() - self.started
        return {"wall_seconds": elapsed, "cumulative_wall_seconds": elapsed,
                "cumulative_training_learner_steps": self.admitted,
                "admitted_learner_microsteps": self.admitted,
                "readonly_evaluation_learner_steps": self.readonly_evaluation_steps,
                "readonly_evaluation_accounting": "External evaluation receipts are separate. Zero here means no evaluation was executed by the trainer, not zero supervisor evaluation cost.",
                "evaluation_learner_microsteps_per_checkpoint": self.limits["evaluation_learner_microsteps_per_checkpoint"],
                "training_choice_allocation": self.training_cap,
                "teacher_queries": self.teacher_queries, "teacher_seconds": self.teacher_seconds,
                "stage_wall_seconds": time.monotonic() - self.stage_started,
                "accounting": "All explicit training learner choices count, including probes, calibration, heldout calibration and censored games. Read-only external evaluation has a separately declared per-checkpoint cap and its own actual receipt. Campaign wall time never resets or pauses for evaluation."}

    def publish_progress(self, *, force=False, status="running"):
        now = time.monotonic()
        if not force and now - self.last_progress_at < 1.0:
            return
        self.last_progress_at = now
        latest_path = self.output / "latest-checkpoint.json"
        latest = json.loads(latest_path.read_bytes()) if latest_path.is_file() else None
        row = {"schema": "86-tcg-agent/native-sustained-progress/1", "status": status,
               **vars(self.progress), "update_in_progress": self.update_in_progress,
               "inflight_optimizer_steps": self.inflight_optimizer_steps,
               "inflight_optimizer_steps_checkpoint_usable": False,
               "latest_completed_checkpoint": None if latest is None else {
                   key: latest[key] for key in ("checkpoint_id", "weights_sha256", "receipt_path",
                                                "receipt_sha256", "stage", "learner_steps_total",
                                                "training_updates_total", "optimizer_steps_total")},
               "resources": self.receipt()}
        atomic_json(self.output / "progress.json", row)

    def observe_optimizer(self, optimizer):
        def completed_step(optimizer, args, kwargs):
            self.inflight_optimizer_steps += 1
            self.publish_progress(force=True)
        optimizer.register_step_post_hook(completed_step)
        return optimizer


def validate_inputs(protocol_path, bindings_path):
    protocol = json.loads(Path(protocol_path).read_bytes())
    bindings = json.loads(Path(bindings_path).read_bytes())
    if (protocol.get("schema") != "86-tcg-agent/native-sustained/1"
            or protocol.get("sequence") != [1, 3, 4] or protocol.get("skipped_stage") != 2
            or protocol.get("opponent") != "fixed_uniform_native_legal"):
        raise ValueError("sustained pilot requires historical stages 1,3,4 and fixed uniform legal opponents")
    limits = protocol["limits"]
    for key, upper in {"wall_seconds": 7200, "evaluation_learner_microsteps_per_checkpoint": 100000,
                       "training_learner_microsteps": 100000, "stage1_wall_seconds": 1800,
                       "stage3_wall_seconds": 600, "per_game_wall_seconds": 120,
                       "max_game_microsteps": 4096}.items():
        if type(limits[key]) not in (int, float) or not 0 < limits[key] <= upper:
            raise ValueError(f"invalid sustained {key}")
    if not 0 < limits["training_stop_campaign_seconds"] <= limits["wall_seconds"]:
        raise ValueError("absolute training stop must fit the unchanged campaign clock")
    if not 0 < limits["finalization_reserve_seconds"] < min(limits["stage1_wall_seconds"], limits["stage3_wall_seconds"]):
        raise ValueError("finalization reserve exceeds a stage allocation")
    stage1 = protocol["stage1"]
    exact1 = {"max_updates": 3, "labels_per_update": 512, "query_probability": 0.1,
              "probe_frames": 32, "lr": 1e-4, "temperature": 1.0}
    if any(stage1.get(key) != value for key, value in exact1.items()):
        raise ValueError("policy transfer recipe differs from the declared sustained pilot")
    if not 0 < stage1["teacher_queries"] <= 25000 or not 0 < stage1["teacher_query_seconds"] <= 1800:
        raise ValueError("teacher cap exceeds the historical authorization")
    calibration = protocol["calibration"]
    exact3 = {"training_games": 16, "heldout_games": 16, "fit_examples": 1024,
              "heldout_examples": 512, "retained_training_per_game": 64,
              "retained_heldout_per_game": 32, "epochs": 8, "lr": 1e-4}
    if any(calibration.get(key) != value for key, value in exact3.items()):
        raise ValueError("critic calibration recipe differs")
    ppo = protocol["ppo"]
    exact4 = {"target_learner_microsteps": 4096, "max_episodes_per_update": 96, "max_updates": 64,
              "gamma": 1.0, "gae_lambda": 0.95, "clip": 0.2, "entropy_coef": 0.01,
              "value_coef": 0.5, "lr": 1e-4, "weight_decay": 0.0, "grad_clip": 0.5,
              "max_epochs": 4, "kl_stop": 0.02}
    if any(ppo.get(key) != value for key, value in exact4.items()):
        raise ValueError("PPO recipe differs from the declared sustained pilot")
    if (protocol["reporting"]["interval_seconds"] != 1800 or protocol["reporting"]["update_trigger"] is not False
            or protocol["evaluation"]["games_per_checkpoint"] != 576):
        raise ValueError("report/evaluation cadence must be every 30 minutes with 576 games")
    if protocol["student"] != {"d_model": 128, "n_layers": 2, "n_heads": 4, "ffn_mult": 2, "dropout": 0.0}:
        raise ValueError("student shape differs from the qualified historical pilot")
    if any(type(seed) is not int or not 0 < seed < 2**32 for seed in protocol["seeds"].values()):
        raise ValueError("seed roots must be nonzero uint32 values")
    artifacts = bindings["artifacts"]
    if not artifacts or hashlib.sha256(canonical(artifacts)).hexdigest() != bindings["artifact_manifest_sha256"]:
        raise ValueError("source artifact closure digest differs")
    closure = {}
    for source, expected in artifacts.items():
        path = Path(source).resolve(strict=True)
        if file_hash(path) != expected:
            raise ValueError(f"source closure mismatch for {path.name}")
        closure[path] = expected
    root = Path(__file__).resolve().parents[1]
    for path in (Path(__file__), root / "scripts/train_native_sustained.py", root / "tcg/native_policy.py",
                 root / "tcg/native_distillation.py", root / "tcg/native_actorcritic.py"):
        if path.resolve(strict=True) not in closure:
            raise ValueError("actual executing source is absent from the pinned closure")
    for key, hash_key in (("engine_library", "engine_sha256"), ("deck_manifest", "deck_manifest_sha256"),
                          ("teacher_checkpoint", "teacher_sha256")):
        path = Path(bindings[key]).resolve(strict=True)
        if closure.get(path) != bindings[hash_key] or file_hash(path) != bindings[hash_key]:
            raise ValueError(f"pinned {key} differs or is absent from closure")
    manifest = json.loads(Path(bindings["deck_manifest"]).read_bytes())
    decks = tuple(tuple(row["card_ids"]) for row in manifest["decks"])
    if len(decks) != 6 or any(len(deck) != 60 or any(type(card) is not int or card <= 0 for card in deck) for deck in decks):
        raise ValueError("historical deck manifest must contain six exact 60-card decks")
    return protocol, bindings, decks


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def atomic_json(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def commit_checkpoint(torch, native, model, optimizer, output, progress, budget, lineage, rng_states,
                      *, completed_update=False, ppo=False):
    sequence = len(tuple(output.glob("checkpoint-*.json")))
    name = f"checkpoint-{sequence:06d}"
    weights, state, receipt_path = (output / f"{name}{suffix}" for suffix in (".pt", ".state.pt", ".json"))
    candidate = replace(progress)
    if completed_update:
        candidate.stage_updates += 1
        candidate.training_updates_total += 1
        candidate.ppo_updates += int(ppo)
        candidate.optimizer_steps_total += budget.inflight_optimizer_steps
    weight_sha = native.save_policy(model, weights)
    resources = budget.receipt()
    previous = output / "latest-checkpoint.json"
    current_lineage = dict(lineage, parent_checkpoint_sha256=json.loads(previous.read_bytes())["weights_sha256"]
                           if previous.is_file() else None)
    torch.save({"optimizer": optimizer.state_dict() if optimizer is not None else None,
                "torch_rng": torch.get_rng_state(), "rng_states": rng_states,
                "progress": dict(vars(candidate)), "resources": resources, "lineage": current_lineage}, state)
    receipt = {"schema": "86-tcg-agent/native-sustained-checkpoint/1", "status": "completed",
               "checkpoint_id": name, "weights_path": str(weights.resolve()), "state_path": str(state.resolve()),
               "receipt_path": str(receipt_path.resolve()), "weights_sha256": weight_sha,
               "state_sha256": file_hash(state), "stage": candidate.stage, "stage_updates": candidate.stage_updates,
               "training_updates_total": candidate.training_updates_total, "ppo_updates": candidate.ppo_updates,
               "optimizer_steps_total": candidate.optimizer_steps_total,
               "learner_steps_total": candidate.learner_steps_total, "training_wall_seconds": resources["wall_seconds"],
               "resources": resources, "budget": resources, "model_identity": native.model_identity(model),
               "actor_sha256": actorcritic.actor_digest(model), "lineage": current_lineage}
    if budget.remaining(reserve=False) <= 0 or budget.admitted > budget.training_cap:
        raise TimeoutError("completed checkpoint cannot be published beyond its allocation")
    write_json(receipt_path, receipt)
    latest = {**receipt, "receipt_sha256": file_hash(receipt_path)}
    atomic_json(output / "latest-checkpoint.json", latest)
    for key, value in vars(candidate).items():
        setattr(progress, key, value)
    if completed_update:
        budget.inflight_optimizer_steps = 0
        budget.update_in_progress = False
    budget.publish_progress(force=True)
    return latest


def helper_protocol(protocol):
    return {"limits": protocol["limits"], "learning": {"query_probability": protocol["stage1"]["query_probability"]}}


def episode_assignment(seed, domain, index):
    item = actorcritic.assignment(seed, domain, index)
    deck_seed = seed32(seed, domain, "decks", index)
    deck_rng = random.Random(deck_seed)
    item.update(learner_deck=deck_rng.randrange(6), opponent_deck=deck_rng.randrange(6),
                deck_assignment_seed=deck_seed, deck_assignment="independent_uniform_six_deck_draws")
    return item


def record_episode(torch, output, domain, index, row, rng, progress):
    append_json(output / f"{domain}-episodes.jsonl", row)
    torch.save(rng, output / f"{domain}-{index:06d}.rng.pt")
    progress.episodes += 1
    progress.natural_terminal_episodes += row["status"] == "terminal"
    if row["status"] == "fault":
        raise RuntimeError(f"{domain} native episode fault; inspect immutable receipts")


def distill(torch, native, model, decks, protocol, bindings, output, progress, budget, lineage):
    budget.enter(1, protocol["limits"]["stage1_wall_seconds"])
    teacher = native.load_teacher(bindings["teacher_checkpoint"], expected_sha256=bindings["teacher_sha256"])
    teacher_before = native.state_digest(teacher)
    critic_before = {name: p.detach().clone() for name, p in model.named_parameters() if name.startswith("value_head.")}
    optimizer = budget.observe_optimizer(torch.optim.AdamW([p for _, p in native.freeze_policy_only(model)],
                                                          lr=protocol["stage1"]["lr"], weight_decay=0.0))
    query_rng = random.Random(seed32(protocol["seeds"]["training"], "sustained-query-selection"))
    probes, index, excluded, labels_used = [], 0, 0, 0
    derived = helper_protocol(protocol)
    while len(probes) < protocol["stage1"]["probe_frames"] and budget.teacher_available():
        item = episode_assignment(protocol["seeds"]["probe"], "sustained-probe", index)
        probe_protocol = {**derived, "limits": dict(derived["limits"],
                          max_game_microsteps=protocol["stage1"]["probe_max_microsteps"])}
        row, labels, rng = run_episode(torch, native, model, decks, item, probe_protocol, budget, teacher=teacher,
                                        probe=True, label_capacity=protocol["stage1"]["probe_frames"] - len(probes))
        record_episode(torch, output, "stage1-probe", index, row, rng, progress)
        probes.extend(labels)
        index += 1
    if len(probes) != protocol["stage1"]["probe_frames"]:
        raise RuntimeError("fixed heldout probe could not be completed within the declared allocation")
    forbidden = {frame.digest for frame, _ in probes}
    torch.save(probes, output / "stage1-heldout-probes.pt")
    write_json(output / "stage1-probe-before.json", label_metrics(torch, native, model, probes))
    index, pending = 0, []
    while progress.stage_updates < protocol["stage1"]["max_updates"] and budget.teacher_available():
        item = episode_assignment(protocol["seeds"]["training"], "sustained-distillation", index)
        row, labels, rng = run_episode(torch, native, model, decks, item, derived, budget, teacher=teacher,
                                      query_rng=query_rng, label_capacity=protocol["stage1"]["labels_per_update"] - len(pending))
        excluded_here = sum(frame.digest in forbidden for frame, _ in labels)
        excluded += excluded_here
        pending.extend((frame, target) for frame, target in labels if frame.digest not in forbidden)
        row.update(excluded_probe_overlap=excluded_here)
        record_episode(torch, output, "stage1-training", index, row, rng, progress)
        index += 1
        if len(pending) < protocol["stage1"]["labels_per_update"]:
            continue
        if any(frame.digest in forbidden for frame, _ in pending):
            raise ValueError("heldout probe digest leaked into a stage-1 update")
        data = output / f"stage1-update-{progress.stage_updates + 1:04d}-labels.pt"
        torch.save(pending, data)
        budget.update_in_progress = True
        report = fit_labels(torch, native, model, optimizer, pending, budget)
        if report is None:
            budget.update_in_progress = False
            break
        labels_used += len(pending)
        if any(not torch.equal(value, dict(model.named_parameters())[name]) for name, value in critic_before.items()):
            raise ValueError("policy-only distillation changed critic head parameters")
        if native.state_digest(teacher) != teacher_before:
            raise ValueError("frozen teacher changed")
        current = dict(lineage, objective="teacher_policy_CE_only", data_sha256=file_hash(data),
                       probe_data_sha256=file_hash(output / "stage1-heldout-probes.pt"),
                       excluded_probe_overlap=excluded, labels_used=labels_used)
        commit_checkpoint(torch, native, model, optimizer, output, progress, budget, current,
                          {"query_rng": query_rng.getstate(), "last_episode_rng": rng}, completed_update=True)
        append_json(output / "stage1-updates.jsonl", dict(report, **vars(progress), resources=budget.receipt()))
        pending.clear()
    write_json(output / "stage1-probe-after.json", label_metrics(torch, native, model, probes))
    result = {"completed_updates": progress.stage_updates, "queried_training_labels_used": labels_used,
              "excluded_exact_probe_digests": excluded, "unoptimized_labels": len(pending),
              "probe_overlap_policy": "Exclude exact frame digests before fit; collect only under the original fixed query/budget rules. No outcome-based resampling.",
              "teacher_unchanged": native.state_digest(teacher) == teacher_before,
              "critic_head_unchanged": True, "resources": budget.receipt()}
    write_json(output / "stage1-result.json", result)
    return result


def calibrate(torch, native, model, decks, protocol, output, progress, budget, lineage):
    budget.enter(3, protocol["limits"]["stage3_wall_seconds"])
    actor_before = actorcritic.actor_digest(model)
    optimizer = budget.observe_optimizer(torch.optim.AdamW(actorcritic.set_trainable(model, critic_only=True),
                                                          lr=protocol["calibration"]["lr"], weight_decay=0.0))
    examples, heldout, outcomes = [], [], {"calibration": [], "calibration_heldout": []}
    for domain, count, destination, cap, per_game in (
            ("calibration", protocol["calibration"]["training_games"], examples,
             protocol["calibration"]["fit_examples"], protocol["calibration"]["retained_training_per_game"]),
            ("calibration_heldout", protocol["calibration"]["heldout_games"], heldout,
             protocol["calibration"]["heldout_examples"], protocol["calibration"]["retained_heldout_per_game"])):
        for index in range(count):
            budget.check()
            item = episode_assignment(protocol["seeds"][domain], "sustained-" + domain, index)
            capacity = min(per_game, max(0, cap - len(destination)))
            row, episode, rng = actorcritic.collect_episode(torch, native, model, decks, item, protocol, budget,
                                                             retain_limit=capacity)
            record_episode(torch, output, "stage3-" + domain, index, row, rng, progress)
            outcomes[domain].append(row["status"])
            if episode is not None:
                destination.extend((step.frame, episode.outcome) for step in episode.steps)
    if (any(status != "terminal" for statuses in outcomes.values() for status in statuses)
            or not examples or not heldout):
        raise RuntimeError("calibration requires the declared natural terminal training and heldout games")
    fixed = tuple(frame for frame, _ in examples[:8])
    logits_before = tuple(native.score_policy(model, frame).logits.clone() for frame in fixed)
    constant = sum(target for _, target in examples) / len(examples)
    before = actorcritic.calibration_metrics(torch, native, model, heldout, constant)
    data = output / "stage3-terminal-examples.pt"
    torch.save({"fit": examples, "heldout": heldout}, data)
    reports = []
    for epoch in range(protocol["calibration"]["epochs"]):
        budget.check()
        budget.update_in_progress = True
        report = actorcritic.fit_critic(torch, native, model, optimizer, examples, budget)
        if actorcritic.actor_digest(model) != actor_before:
            raise ValueError("stage 3 changed the frozen actor/encoder state")
        if any(not torch.equal(logits, native.score_policy(model, frame).logits) for logits, frame in zip(logits_before, fixed)):
            raise ValueError("stage 3 changed exact fixed-frame actor logits")
        commit_checkpoint(torch, native, model, optimizer, output, progress, budget,
                          dict(lineage, objective="outcome_MC_value_head_MSE_only", data_sha256=file_hash(data),
                               actor_freeze_verified=True, exact_actor_logits_verified=True),
                          {"last_episode_rng": rng}, completed_update=True)
        reports.append(dict(report, epoch=epoch + 1))
    after = actorcritic.calibration_metrics(torch, native, model, heldout, constant)
    result = {"fit_epochs": reports, "heldout_before": before, "heldout_after": after,
              "natural_games": {key: len(value) for key, value in outcomes.items()},
              "actor_freeze_verified": True, "exact_actor_logits_verified": True,
              "actual_fit_examples": len(examples), "actual_heldout_examples": len(heldout),
              "teacher_queries_added": 0, "resources": budget.receipt()}
    write_json(output / "stage3-result.json", result)
    atomic_json(output / "pre-rl-checkpoint.json", json.loads((output / "latest-checkpoint.json").read_bytes()))
    return result


def improve(torch, native, model, decks, protocol, output, progress, budget, lineage):
    remaining = max(0.0, budget.started + protocol["limits"]["training_stop_campaign_seconds"] - time.monotonic())
    budget.enter(4, remaining)
    optimizer = budget.observe_optimizer(torch.optim.AdamW(actorcritic.set_trainable(model, critic_only=False),
                                                          lr=protocol["ppo"]["lr"], weight_decay=0.0))
    index = 0
    while progress.stage_updates < protocol["ppo"]["max_updates"]:
        budget.check()
        episodes, steps, attempts = [], 0, 0
        policy_digest = native.state_digest(model)
        while steps < protocol["ppo"]["target_learner_microsteps"] and attempts < protocol["ppo"]["max_episodes_per_update"]:
            budget.check()
            item = episode_assignment(protocol["seeds"]["ppo"], "sustained-ppo", index)
            row, episode, rng = actorcritic.collect_episode(torch, native, model, decks, item, protocol, budget)
            record_episode(torch, output, "stage4", index, row, rng, progress)
            index += 1
            attempts += 1
            if episode is not None:
                episodes.append(episode)
                steps += len(episode.steps)
        if not episodes:
            raise RuntimeError("PPO batch has no natural complete learner episodes")
        if steps < protocol["ppo"]["target_learner_microsteps"]:
            return {"completed_updates": progress.stage_updates, "resources": budget.receipt(),
                    "stop_reason": "episode_collection_cap_before_declared_rollout_target",
                    "discarded_terminal_batch_steps": steps, "discarded_terminal_episodes": len(episodes)}
        if native.state_digest(model) != policy_digest:
            raise ValueError("rollout collection mutated the frozen learner")
        data = output / f"stage4-update-{progress.stage_updates + 1:04d}-episodes.pt"
        torch.save(episodes, data)
        budget.update_in_progress = True
        report = actorcritic.ppo_update(torch, native, model, optimizer, episodes, protocol["ppo"], budget)
        current = dict(lineage, objective="PPO_plus_terminal_MC_critic", input_policy_digest=policy_digest,
                       data_sha256=file_hash(data), terminal_batch_steps=steps, terminal_episodes=len(episodes),
                       rollout_target_reached=steps >= protocol["ppo"]["target_learner_microsteps"])
        commit_checkpoint(torch, native, model, optimizer, output, progress, budget, current,
                          {"last_episode_rng": rng}, completed_update=True, ppo=True)
        append_json(output / "stage4-updates.jsonl", {"epochs": report, "attempted_games": attempts,
                    "terminal_games": len(episodes), "terminal_batch_steps": steps,
                    **vars(progress), "resources": budget.receipt()})
    return {"completed_updates": progress.stage_updates, "resources": budget.receipt()}
