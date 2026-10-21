"""Authorized small CABT baseline PPO updates. Importing performs no training."""

from dataclasses import dataclass
import hashlib
import io
import time
from pathlib import Path
from typing import Callable

import torch

from .collection import CollectionIdentity, EpochReport, EpochStopController, FrozenBatch
from .decision import integer
from .objective import minibatch_indices, ppo_objective
from .policy import Policy, PolicyConfig
from .rollout import RolloutTargets


class UpdateFailed(RuntimeError):
    """The affected update is unusable. Parameters may already have changed."""

    def __init__(self, phase: str, epoch: int = 0, minibatch: int = 0, optimizer_steps: int = 0):
        super().__init__(f"PPO update failed during {phase}; do not checkpoint or resume this update")
        self.phase = phase
        self.epoch = epoch
        self.minibatch = minibatch
        self.optimizer_steps = optimizer_steps
        self.resume_allowed = False
        self.checkpoint_allowed = False


@dataclass(frozen=True)
class UpdateReport:
    optimizer_steps: int
    epoch_reports: tuple[EpochReport, ...]
    learner_decisions: int
    minibatch_size: int = 256
    minibatch_metrics: tuple[dict, ...] = ()


def create_optimizer(policy: Policy, *, baseline_authorized: bool = False) -> torch.optim.AdamW:
    if baseline_authorized is not True:
        raise ValueError("small CABT baseline training authorization is required")
    if type(policy) is not Policy:
        raise ValueError("optimizer requires the reference Policy")
    return torch.optim.AdamW(policy.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-5,
                             weight_decay=0.0, amsgrad=False, maximize=False,
                             foreach=False, fused=False, capturable=False, differentiable=False)


def _validate_optimizer(policy, optimizer):
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise ValueError("reference update requires one AdamW parameter group")
    group = optimizer.param_groups[0]
    recipe = {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-5, "weight_decay": 0.0,
              "amsgrad": False, "maximize": False, "capturable": False, "differentiable": False}
    if any(group.get(name) != value for name, value in recipe.items()):
        raise ValueError("AdamW configuration differs from reference")
    if group.get("foreach") not in (None, False) or group.get("fused") not in (None, False):
        raise ValueError("reference update requires unfused AdamW")
    if tuple(map(id, group["params"])) != tuple(map(id, policy.parameters())):
        raise ValueError("optimizer parameters must match policy in native order")


def action_statistics(policy: Policy, targets: RolloutTargets, indices: tuple[int, ...]):
    steps = tuple(targets.steps[index] for index in indices)
    prediction = policy([step.decision for step in steps])
    if prediction.logits.dtype is not torch.float32 or prediction.values.dtype is not torch.float32:
        raise UpdateFailed("mixed_precision")
    mask = prediction.legal_mask
    if not mask.any(dim=1).all() or not torch.isfinite(prediction.logits[mask]).all():
        raise UpdateFailed("legal_logits")
    log_probabilities = prediction.logits.log_softmax(-1)
    safe_log_probabilities = torch.where(mask, log_probabilities, torch.zeros_like(log_probabilities))
    entropies = -(log_probabilities.exp() * safe_log_probabilities).sum(-1)
    selected = []
    for row, step in enumerate(steps):
        expected_legal = tuple(choice in step.decision.choices() for choice in prediction.option_ids[row])
        actual_legal = tuple(bool(value) for value in mask[row, :len(expected_legal)].tolist())
        if expected_legal != actual_legal or step.action_id not in step.decision.choices():
            raise UpdateFailed("recorded_legal_mask")
        try:
            column = prediction.option_ids[row].index(step.action_id)
        except ValueError as error:
            raise UpdateFailed("recorded_action") from error
        selected.append(log_probabilities[row, column])
    chosen = torch.stack(selected)
    if any(not torch.isfinite(value).all() for value in (chosen, prediction.values, entropies)):
        raise UpdateFailed("action_statistics")
    return chosen, prediction.values, entropies


def _accumulate_minibatch(parameters, optimizer, indices: tuple[int, ...], microbatch_size: int,
                          loss_for_indices: Callable):
    integer(microbatch_size, "microbatch size", 1)
    if not indices:
        raise ValueError("empty minibatch")
    parameters = tuple(parameters)
    optimizer.zero_grad(set_to_none=True)
    for start in range(0, len(indices), microbatch_size):
        chunk = indices[start:start + microbatch_size]
        loss = loss_for_indices(chunk)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise UpdateFailed("loss")
        (loss * (len(chunk) / len(indices))).backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in parameters):
            raise UpdateFailed("gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 0.5, error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise UpdateFailed("gradient_norm")
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in parameters):
        raise UpdateFailed("parameters")
    return float(norm)


def _minibatch_step(policy, optimizer, targets, indices, microbatch_size, check_deadline):
    metrics = {name: 0.0 for name in ("loss", "policy_loss", "value_loss", "entropy", "approximate_kl", "clip_fraction")}
    def loss_for_indices(chunk):
        check_deadline()
        logp, values, entropy = action_statistics(policy, targets, chunk)
        old = values.new_tensor([targets.steps[index].old_log_probability for index in chunk])
        returns = values.new_tensor([targets.returns[index] for index in chunk])
        advantages = values.new_tensor([targets.normalized_advantages[index] for index in chunk])
        objective = ppo_objective(logp, values, entropy, old, returns, advantages)
        for name in metrics:
            metrics[name] += float(getattr(objective, name).detach()) * len(chunk) / len(indices)
        return objective.loss

    metrics["gradient_norm_before_clip"] = _accumulate_minibatch(policy.parameters(), optimizer, indices, microbatch_size, loss_for_indices)
    metrics["learner_decisions"] = len(indices)
    return metrics


def update_policy(
    policy: Policy, optimizer: torch.optim.AdamW, batch: FrozenBatch, *,
    expected_identity: CollectionIdentity, shuffle_generator: torch.Generator,
    weights_path: str | Path,
    baseline_authorized: bool = False, microbatch_size: int = 8,
    expected_config: PolicyConfig = PolicyConfig(),
    deadline_monotonic: float | None = None, progress_callback: Callable | None = None,
) -> UpdateReport:
    """Mutate one learner from a frozen rollout; checkpoint only after this returns."""
    if baseline_authorized is not True:
        raise ValueError("small CABT baseline training authorization is required")
    if type(policy) is not Policy or policy.config != expected_config or expected_config != PolicyConfig():
        raise ValueError("policy configuration differs from reference")
    if type(batch) is not FrozenBatch or type(expected_identity) is not CollectionIdentity or batch.identity != expected_identity:
        raise ValueError("frozen rollout differs from the current learner/environment identity")
    if type(shuffle_generator) is not torch.Generator or str(shuffle_generator.device) != "cpu":
        raise ValueError("shuffle requires a caller-owned CPU torch Generator")
    integer(microbatch_size, "microbatch size", 1)
    if deadline_monotonic is not None and (type(deadline_monotonic) not in (float, int) or not torch.isfinite(torch.tensor(deadline_monotonic))):
        raise ValueError("training deadline must be finite monotonic seconds")
    def check_deadline():
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise UpdateFailed("training_wall_cap")
    check_deadline()
    _validate_optimizer(policy, optimizer)
    parameters = tuple(policy.parameters())
    if any(parameter.dtype is not torch.float32 or not parameter.requires_grad for parameter in parameters):
        raise ValueError("reference PPO requires trainable float32 parameters without AMP")
    if any(not torch.isfinite(parameter).all() for parameter in parameters):
        raise UpdateFailed("initial_parameters")
    weights_path = Path(weights_path)
    payload = weights_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_identity.learner_weights_sha256:
        raise ValueError("learner weights artifact differs from collection identity")
    recorded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    current = policy.state_dict()
    if (not isinstance(recorded, dict) or set(recorded) != set(current)
            or any(type(recorded[name]) is not torch.Tensor
                   or recorded[name].dtype != tensor.dtype or recorded[name].shape != tensor.shape
                   or not torch.equal(recorded[name], tensor.detach().cpu()) for name, tensor in current.items())):
        raise ValueError("current learner weights differ from the pinned artifact")
    controller = EpochStopController(batch)
    targets = batch.targets
    if any(len(values) != len(targets.steps) for values in
           (targets.returns, targets.advantages, targets.normalized_advantages)):
        raise ValueError("incomplete frozen targets")
    if any(step.provenance.policy_version != expected_identity.learner_policy_version for step in targets.steps):
        raise ValueError("mixed learner policy versions in frozen targets")
    reports = []
    metrics = []
    optimizer_steps = 0
    while controller.can_continue:
        epoch = controller.observed_epochs + 1
        order = tuple(torch.randperm(len(targets.steps), generator=shuffle_generator).tolist())
        for minibatch, indices in enumerate(minibatch_indices(order, 256), 1):
            try:
                started = time.monotonic()
                measured = _minibatch_step(policy, optimizer, targets, indices, microbatch_size, check_deadline)
                optimizer_steps += 1
                measured.update(epoch=epoch, minibatch=minibatch, wall_seconds=time.monotonic() - started)
                metrics.append(measured)
                if progress_callback is not None:
                    progress_callback(dict(measured))
            except (RuntimeError, ValueError, OverflowError) as error:
                phase = error.phase if isinstance(error, UpdateFailed) else "minibatch"
                raise UpdateFailed(phase, epoch, minibatch, optimizer_steps) from error
        observations = []
        try:
            with torch.inference_mode():
                for start in range(0, len(targets.steps), microbatch_size):
                    check_deadline()
                    indices = tuple(range(start, min(start + microbatch_size, len(targets.steps))))
                    current, _, _ = action_statistics(policy, targets, indices)
                    observations.extend((targets.steps[index].provenance.episode_id, targets.steps[index].sequence, value)
                                        for index, value in zip(indices, current.tolist()))
            reports.append(controller.observe_epoch(tuple(observations)))
        except (RuntimeError, ValueError, OverflowError) as error:
            phase = error.phase if isinstance(error, UpdateFailed) else "full_rollout_kl"
            raise UpdateFailed(phase, epoch, 0, optimizer_steps) from error
    return UpdateReport(optimizer_steps, tuple(reports), len(targets.steps), minibatch_metrics=tuple(metrics))
