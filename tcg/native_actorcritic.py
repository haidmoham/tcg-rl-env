"""Teacher-free terminal critic calibration and bounded native PPO."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from .native_distillation import append_json, canonical, file_hash, seed32, wall_guard


@dataclass(frozen=True)
class LearnerStep:
    frame: Any
    action: int
    old_logp: float
    old_value: float
    genuine: bool


@dataclass(frozen=True)
class TerminalEpisode:
    episode_id: str
    learner_seat: int
    policy_digest: str
    steps: tuple[LearnerStep, ...]
    outcome: float


class PilotBudget:
    def __init__(self, protocol, prior, *, started=None):
        self.protocol = protocol
        self.started = time.monotonic() if started is None else started
        self.prior_wall = float(prior["wall_seconds"])
        self.charged = int(prior["admitted_learner_microsteps"])
        self.prior = dict(prior)
        self.stage_started = None
        self.stage_seconds = None

    def stage(self, seconds):
        self.stage_started = time.monotonic()
        self.stage_seconds = seconds

    def remaining(self, *, reserve=True):
        limits = self.protocol["limits"]
        arm = limits["wall_seconds"] - self.prior_wall - (time.monotonic() - self.started)
        stage = float("inf") if self.stage_started is None else self.stage_seconds - (time.monotonic() - self.stage_started)
        return min(arm, stage) - (limits["finalization_reserve_seconds"] if reserve else 0)

    def check(self, *, reserve=True):
        if self.remaining(reserve=reserve) <= 0:
            raise TimeoutError("shared arm or stage wall allocation exhausted")
        if self.charged >= self.protocol["limits"]["admitted_learner_microsteps"]:
            raise TimeoutError("shared learner microstep cap exhausted")

    def receipt(self):
        elapsed = self.prior_wall + time.monotonic() - self.started
        return {"wall_seconds": elapsed,
                "cumulative_wall_seconds": elapsed,
                "cumulative_admitted_microsteps": self.charged,
                "admitted_learner_microsteps": self.charged,
                "teacher_queries": self.prior.get("teacher_queries", 0),
                "teacher_seconds": self.prior.get("teacher_seconds", 0),
                "new_teacher_queries": 0, "stage_wall_seconds": None if self.stage_started is None
                else time.monotonic() - self.stage_started,
                "counting": "charged explicit learner atomic choices, including censored work; natural terminal admissions reported separately"}


def validate_inputs(protocol_path, bindings_path, stage1_run):
    protocol = json.loads(Path(protocol_path).read_bytes())
    bindings = json.loads(Path(bindings_path).read_bytes())
    if protocol.get("schema") != "86-tcg-agent/native-actorcritic/1" or protocol.get("sequence") != [3, 4]:
        raise ValueError("this runner implements stage 3 followed by stage 4 only")
    if protocol.get("skipped_stage") != 2 or protocol.get("teacher_access") != "forbidden":
        raise ValueError("critic transfer is skipped and all teacher access is forbidden")
    if protocol.get("opponent") != "fixed_uniform_native_legal":
        raise ValueError("the fixed uniform legal opponent cannot change")
    limits = protocol["limits"]
    ceilings = {"wall_seconds": 7200, "admitted_learner_microsteps": 100000,
                "stage3_wall_seconds": 1200, "stage4_wall_seconds": 2400,
                "per_game_wall_seconds": 120, "max_game_microsteps": 4096}
    for key, ceiling in ceilings.items():
        if type(limits[key]) not in (int, float) or not 0 < limits[key] <= ceiling:
            raise ValueError(f"invalid {key}")
    if not 0 < limits["finalization_reserve_seconds"] < min(limits["stage3_wall_seconds"], limits["stage4_wall_seconds"]):
        raise ValueError("invalid checkpoint reserve")
    calibration, ppo = protocol["calibration"], protocol["ppo"]
    if (calibration["training_games"] != 4 or calibration["heldout_games"] < 4
            or calibration["max_training_examples"] != 128 or calibration["epochs"] != 1
            or calibration["lr"] != 1e-4):
        raise ValueError("calibration differs from the bounded pilot recipe")
    exact = {"gamma": 1.0, "gae_lambda": 0.95, "clip": 0.2, "entropy_coef": 0.01,
             "value_coef": 0.5, "lr": 1e-4, "weight_decay": 0.0, "grad_clip": 0.5,
             "max_epochs": 4, "kl_stop": 0.02, "max_updates": 2, "episodes_per_update": 2}
    if any(ppo.get(key) != value for key, value in exact.items()):
        raise ValueError("PPO differs from the declared pilot recipe")
    artifacts = bindings["artifacts"]
    if not artifacts or hashlib.sha256(canonical(artifacts)).hexdigest() != bindings["artifact_manifest_sha256"]:
        raise ValueError("source closure manifest digest differs")
    closure = set()
    for source, expected in artifacts.items():
        path = Path(source).resolve(strict=True)
        if file_hash(path) != expected:
            raise ValueError(f"source closure mismatch for {path.name}")
        closure.add(path)
    root = Path(__file__).resolve().parents[1]
    required = [Path(__file__), root / "tcg/native_policy.py", root / "tcg/native_distillation.py",
                root / "scripts/train_native_actorcritic.py", Path(bindings["engine_library"]),
                Path(bindings["deck_manifest"])]
    if any(path.resolve(strict=True) not in closure for path in required):
        raise ValueError("actual executing sources, engine and decks must be pinned")
    for key, hash_key in (("engine_library", "engine_sha256"), ("deck_manifest", "deck_manifest_sha256")):
        if file_hash(bindings[key]) != bindings[hash_key]:
            raise ValueError(f"{key} identity differs")
    stage1 = Path(stage1_run).resolve(strict=True)
    prior = json.loads((stage1 / "training-result.json").read_bytes())
    prior_checkpoint = json.loads((stage1 / "student.json").read_bytes())
    prior_manifest = json.loads((stage1 / "manifest.json").read_bytes())
    stage1_bindings_path = Path(bindings["stage1_bindings"]).resolve(strict=True)
    stage1_protocol_path = Path(bindings["stage1_protocol"]).resolve(strict=True)
    if file_hash(stage1_bindings_path) != bindings["stage1_bindings_sha256"]:
        raise ValueError("stage 1 bindings file differs from its pinned identity")
    if file_hash(stage1_protocol_path) != bindings["stage1_protocol_sha256"]:
        raise ValueError("stage 1 protocol file differs from its pinned identity")
    original_bindings = json.loads(stage1_bindings_path.read_bytes())
    original_protocol = json.loads(stage1_protocol_path.read_bytes())
    original_closure = original_bindings["artifacts"]
    original_closure_sha = hashlib.sha256(canonical(original_closure)).hexdigest()
    original_protocol_sha = hashlib.sha256(canonical(original_protocol)).hexdigest()
    if (original_closure_sha != original_bindings["artifact_manifest_sha256"]
            or original_closure_sha != prior_manifest["artifact_manifest_sha256"]):
        raise ValueError("stage 1 manifest does not identify the supplied original source closure")
    if (prior_manifest.get("mode") != "train" or prior_manifest["protocol"] != original_protocol
            or prior_manifest["protocol_sha256"] != original_protocol_sha
            or prior_checkpoint["protocol_sha256"] != original_protocol_sha):
        raise ValueError("stage 1 protocol, training manifest and checkpoint lineage disagree")
    continuation_closure = {Path(source).resolve(strict=True): expected for source, expected in artifacts.items()}
    for source, expected in original_closure.items():
        if continuation_closure.get(Path(source).resolve(strict=True)) != expected:
            raise ValueError("continuation changed or omitted a frozen stage 1 source artifact")
    for key, hash_key in (("engine_library", "engine_sha256"), ("deck_manifest", "deck_manifest_sha256")):
        if original_bindings[hash_key] != bindings[hash_key]:
            raise ValueError(f"continuation changed the stage 1 {key}")
        if file_hash(original_bindings[key]) != original_bindings[hash_key]:
            raise ValueError(f"original stage 1 {key} no longer matches its binding")
    if prior.get("status") != "completed_bounded_stage" or prior.get("critic_parameters_unchanged") is not True:
        raise ValueError("stage 1 must have completed with its original critic unchanged")
    if file_hash(stage1 / "student.pt") != prior["student_weights_sha256"]:
        raise ValueError("stage 1 endpoint checkpoint differs from its receipt")
    if prior_checkpoint["weights_sha256"] != prior["student_weights_sha256"]:
        raise ValueError("stage 1 checkpoint and training receipts disagree")
    if not prior_checkpoint.get("protocol_sha256"):
        raise ValueError("stage 1 protocol lineage is missing")
    if prior["budget"]["wall_seconds"] > 1800 or prior["budget"]["admitted_learner_microsteps"] >= limits["admitted_learner_microsteps"]:
        raise ValueError("stage 1 exceeded its allocation or exhausted the shared microstep budget")
    manifest = json.loads(Path(bindings["deck_manifest"]).read_bytes())
    decks = tuple(tuple(row["card_ids"]) for row in manifest["decks"])
    if len(decks) != 6 or any(len(deck) != 60 for deck in decks):
        raise ValueError("the historical pilot requires six exact 60-card decks")
    return protocol, bindings, prior, decks


def actor_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("value_head."):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(canonical([name, str(value.dtype), list(value.shape)]))
        digest.update(value.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def set_trainable(model, *, critic_only):
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("value_head.") if critic_only else True)
        parameter.grad = None
    return tuple(parameter for parameter in model.parameters() if parameter.requires_grad)


def score_actorcritic(torch, native, model, frame, *, grad=False):
    if model.training or any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("native actor/critic requires deterministic CPU eval mode")
    if frame.schema_digest != native._schema_digest():
        raise ValueError("actor/critic frame schema differs")
    with torch.enable_grad() if grad else torch.no_grad():
        logits, raw_value = model(frame._payload(), validate=True)
        value = torch.tanh(raw_value)
    if logits.ndim != 2 or logits.shape[0] != frame.real_rows or value.shape != (frame.real_rows,):
        raise ValueError("native actor/critic returned unexpected shapes")
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for row, record in enumerate(frame.records):
        mask[row, list(record.eligible_columns)] = True
    if not torch.equal(torch.isfinite(logits), mask) or not torch.isneginf(logits[~mask]).all() or not torch.isfinite(value).all():
        raise ValueError("native actor/critic violates legal-logit/value contract")
    return logits, value, mask


def assignment(seed, domain, index):
    return {"index": index, "learner_deck": index % 6, "opponent_deck": (index // 2 + 1) % 6,
            "learner_seat": index % 2, "engine_seed": seed32(seed, domain, "engine", index),
            "learner_seed": seed32(seed, domain, "learner", index),
            "opponent_seed": seed32(seed, domain, "opponent", index)}


def collect_episode(torch, native, model, decks, item, protocol, budget, *, retain_limit=None):
    """Return a gradient-admissible episode only on a natural terminal outcome."""
    from fastenc.games import Games
    started, steps, attempted = time.monotonic(), [], 0
    status, reason, winner, stats = "censored", None, None, None
    learner_rng = torch.Generator(device="cpu").manual_seed(item["learner_seed"])
    opponent_rng = torch.Generator(device="cpu").manual_seed(item["opponent_seed"])
    digest = native.state_digest(model)
    seat = item["learner_seat"]
    pair = [None, None]
    pair[seat], pair[1 - seat] = decks[item["learner_deck"]], decks[item["opponent_deck"]]
    try:
        budget.check()
        with wall_guard(min(protocol["limits"]["per_game_wall_seconds"], budget.remaining())), Games(
                [pair], seed=item["engine_seed"], seeds=[item["engine_seed"]], threads=1) as games:
            capture = native.FrameCapture()
            for _ in range(protocol["limits"]["max_game_microsteps"]):
                budget.check()
                with games.encode() as encoded:
                    record = encoded.records[0]
                    if int(record["terminal"]) == 1:
                        winner = int(record["winner"])
                        if winner not in (0, 1, 2):
                            raise ValueError("unknown native terminal winner")
                        status = "terminal"
                        break
                    if int(record["decisionIndex"]) >= protocol["limits"]["max_game_microsteps"]:
                        reason = "native_decision_index_cap"
                        break
                    frame = capture.capture(encoded)
                if frame.real_rows != 1:
                    raise ValueError("sequential collector requires one encoded row")
                rec = frame.records[0]
                if rec.select_player == seat:
                    budget.charged += 1
                    logits, value, _ = score_actorcritic(torch, native, model, frame)
                    action = int(torch.multinomial(logits[0].softmax(-1), 1, generator=learner_rng).item())
                    if retain_limit is None or len(steps) < retain_limit:
                        steps.append(LearnerStep(frame, action, float(logits[0].log_softmax(-1)[action]),
                                                 float(value[0]), len(rec.eligible_columns) > 1))
                else:
                    action = rec.eligible_columns[int(torch.randint(len(rec.eligible_columns), (1,), generator=opponent_rng))]
                if action not in rec.eligible_columns:
                    raise ValueError("sampled illegal native action")
                games.step([action])
                attempted += 1
            else:
                reason = "explicit_game_microstep_cap"
            stats = games.stats()
    except TimeoutError as error:
        status, reason = "censored", str(error)
    except Exception as error:
        status, reason = "fault", f"{type(error).__name__}: {error}"
    episode = None
    outcome = None if status != "terminal" else (0.0 if winner == 2 else (1.0 if winner == seat else -1.0))
    if status == "terminal" and steps:
        episode = TerminalEpisode(str(item["engine_seed"]), seat, digest, tuple(steps), outcome)
    row = dict(item, status=status, reason=reason, winner=winner, outcome=outcome,
               explicit_microsteps=attempted, retained_learner_steps=len(steps),
               terminal_admitted_steps=len(steps) if episode is not None else 0,
               policy_digest=digest, wall_seconds=time.monotonic() - started, native_stats=stats)
    rng = {"learner": learner_rng.get_state(), "opponent": opponent_rng.get_state()}
    return row, episode, rng


def calibration_metrics(torch, native, model, examples, constant):
    rows = []
    for frame, target in examples:
        _, value, _ = score_actorcritic(torch, native, model, frame)
        prediction = float(value[0])
        rows.append({"frame_digest": frame.digest, "prediction": prediction, "target": target})
    count = len(rows)
    return {"count": count, "mse": sum((r["prediction"] - r["target"]) ** 2 for r in rows) / count if count else None,
            "constant_training_mean": constant,
            "constant_mse": sum((constant - r["target"]) ** 2 for r in rows) / count if count else None,
            "mean_bias": sum(r["prediction"] - r["target"] for r in rows) / count if count else None,
            "rows": rows}


def fit_critic(torch, native, model, optimizer, examples, budget):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for frame, target in examples:
        budget.check()
        _, value, _ = score_actorcritic(torch, native, model, frame, grad=True)
        loss = (value[0] - target).square()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite critic loss")
        (loss / len(examples)).backward()
        losses.append(float(loss.detach()))
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.5, error_if_nonfinite=True)
    budget.check()
    optimizer.step()
    return {"examples": len(examples), "mse_before_step": sum(losses) / len(losses), "gradient_norm": float(norm)}


def terminal_targets(episodes):
    rows, advantages, returns = [], [], []
    if len({episode.policy_digest for episode in episodes}) != 1:
        raise ValueError("PPO collection contains mixed learner versions")
    for episode in episodes:
        next_value, gae = 0.0, 0.0
        local = []
        for index in range(len(episode.steps) - 1, -1, -1):
            step = episode.steps[index]
            reward = episode.outcome if index == len(episode.steps) - 1 else 0.0
            delta = reward + next_value - step.old_value
            gae = delta + 0.95 * gae
            local.append(gae)
            next_value = step.old_value
        rows.extend(episode.steps)
        advantages.extend(reversed(local))
        returns.extend([episode.outcome] * len(episode.steps))
    genuine = [adv for step, adv in zip(rows, advantages) if step.genuine]
    mean = sum(genuine) / len(genuine) if genuine else 0.0
    variance = sum((value - mean) ** 2 for value in genuine) / len(genuine) if genuine else 0.0
    normalized = tuple((adv - mean) / (variance ** 0.5 + 1e-8) if step.genuine else 0.0
                       for step, adv in zip(rows, advantages))
    return tuple(rows), normalized, tuple(returns)


def ppo_update(torch, native, model, optimizer, episodes, config, budget):
    if not episodes or any(native.state_digest(model) != ep.policy_digest for ep in episodes):
        raise ValueError("PPO requires complete episodes from the current frozen student")
    rows, advantages, returns = terminal_targets(episodes)
    if not rows:
        raise ValueError("PPO requires nonempty natural terminal episodes")
    genuine_count = sum(step.genuine for step in rows)
    reports = []
    for epoch in range(config["max_epochs"]):
        budget.check()
        optimizer.zero_grad(set_to_none=True)
        sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for step, advantage, target in zip(rows, advantages, returns):
            budget.check()
            logits, values, mask = score_actorcritic(torch, native, model, step.frame, grad=True)
            value_loss = (values[0] - target).square()
            loss = config["value_coef"] * value_loss / len(rows)
            sums["value_loss"] += float(value_loss.detach()) / len(rows)
            if step.genuine:
                legal_logp = logits[0][mask[0]].log_softmax(-1)
                chosen_logp = logits[0].log_softmax(-1)[step.action]
                ratio = (chosen_logp - step.old_logp).exp()
                unclipped = ratio * advantage
                clipped = ratio.clamp(1 - config["clip"], 1 + config["clip"]) * advantage
                policy_loss = -torch.minimum(unclipped, clipped)
                entropy = -(legal_logp.exp() * legal_logp).sum()
                loss = loss + (policy_loss - config["entropy_coef"] * entropy) / genuine_count
                sums["policy_loss"] += float(policy_loss.detach()) / genuine_count
                sums["entropy"] += float(entropy.detach()) / genuine_count
            if not torch.isfinite(loss):
                raise ValueError("nonfinite PPO loss")
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
        budget.check()
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("nonfinite PPO parameters")
        kl = []
        for step in rows:
            budget.check()
            if not step.genuine:
                continue
            logits, _, _ = score_actorcritic(torch, native, model, step.frame)
            log_ratio = float(logits[0].log_softmax(-1)[step.action]) - step.old_logp
            kl.append(float(torch.expm1(torch.tensor(log_ratio))) - log_ratio)
        measured_kl = sum(kl) / len(kl) if kl else 0.0
        if not math.isfinite(measured_kl):
            raise ValueError("nonfinite full-batch PPO approximate KL")
        reports.append(dict(sums, epoch=epoch + 1, approximate_kl=measured_kl,
                            gradient_norm=float(norm), learner_steps=len(rows), genuine_steps=genuine_count))
        if measured_kl > config["kl_stop"]:
            break
    return reports
