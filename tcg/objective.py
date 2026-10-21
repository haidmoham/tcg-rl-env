"""R0 PPO forward objective. This module performs no optimization."""

from dataclasses import dataclass
from math import log

import torch
from torch import Tensor

from .decision import integer


@dataclass(frozen=True)
class Objective:
    loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor


def ppo_objective(
    log_probabilities: Tensor, values: Tensor, entropies: Tensor,
    old_log_probabilities: Tensor, returns: Tensor, normalized_advantages: Tensor,
) -> Objective:
    tensors = (log_probabilities, values, entropies, old_log_probabilities, returns, normalized_advantages)
    if any(type(tensor) is not Tensor or not tensor.is_floating_point() or tensor.ndim != 1
           or tensor.numel() == 0 for tensor in tensors):
        raise ValueError("objective inputs must be nonempty floating vectors")
    if any(tensor.shape != values.shape or tensor.dtype != values.dtype or tensor.device != values.device
           or not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("objective vectors must share shape, dtype, device and be finite")
    if any(tensor.requires_grad for tensor in (old_log_probabilities, returns, normalized_advantages)):
        raise ValueError("recorded policy and rollout targets must be detached")
    if (log_probabilities > 0).any() or (old_log_probabilities > 0).any() or (entropies < 0).any():
        raise ValueError("invalid log probability or entropy")
    log_ratio = log_probabilities - old_log_probabilities
    if (log_ratio.abs() >= log(torch.finfo(values.dtype).max)).any():
        raise ValueError("policy ratio cannot be represented safely")
    ratio = log_ratio.exp()
    policy_loss = -torch.minimum(
        ratio * normalized_advantages,
        ratio.clamp(0.8, 1.2) * normalized_advantages,
    ).mean()
    value_loss = (values - returns).square().mean()
    entropy = entropies.mean()
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
    approximate_kl = (-log_ratio).mean()
    clip_fraction = ((ratio < 0.8) | (ratio > 1.2)).to(values.dtype).mean()
    result = Objective(loss, policy_loss, value_loss, entropy, approximate_kl, clip_fraction)
    if any(not torch.isfinite(tensor).all() for tensor in vars(result).values()):
        raise ValueError("nonfinite PPO objective or diagnostics")
    return result


def minibatch_indices(permutation: tuple[int, ...], batch_size: int = 256) -> tuple[tuple[int, ...], ...]:
    integer(batch_size, "batch size", 1)
    if type(permutation) is not tuple or not permutation:
        raise ValueError("minibatch order must be a nonempty tuple")
    if any(type(index) is not int for index in permutation) or set(permutation) != set(range(len(permutation))):
        raise ValueError("minibatch order must permute every rollout index exactly once")
    return tuple(permutation[start:start + batch_size] for start in range(0, len(permutation), batch_size))
