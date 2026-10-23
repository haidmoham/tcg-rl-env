"""Masked candidate PPO, with explicit actor-decision trajectory boundaries."""

from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.distributions import Categorical

try:
    from . import card_action_torch as bc
except ImportError:
    import card_action_torch as bc

RUNS = Path(__file__).resolve().parents[1] / "runs"
CHECKPOINT_SCHEMA = "native-masked-ppo/1"


@dataclass(frozen=True)
class ModelConfig:
    state_dim: int = 224
    candidate_dim: int = 191
    hidden_sizes: tuple = (128, 64)
    feature_schema: str = "native-card-action-features/1"

    def __post_init__(self):
        sizes = (self.state_dim, self.candidate_dim, *self.hidden_sizes)
        if not self.hidden_sizes or any(type(x) is not int or not 1 <= x <= 8192 for x in sizes):
            raise ValueError("positive bounded model dimensions required")
        if not isinstance(self.feature_schema, str) or not self.feature_schema:
            raise ValueError("feature schema required")


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = .99
    gae_lambda: float = .95
    clip_coef: float = .2
    value_clip_coef: float = .2
    value_coef: float = .5
    entropy_coef: float = .01
    learning_rate: float = .0003
    max_grad_norm: float = .5
    epochs: int = 4
    minibatch_size: int = 64
    target_kl: float = .03
    normalize_advantages: bool = True
    max_records: int = 32768
    max_padded_candidates: int = 131072

    def __post_init__(self):
        for name in ("gamma", "gae_lambda", "clip_coef", "value_clip_coef", "value_coef",
                     "entropy_coef", "learning_rate", "max_grad_norm", "target_kl"):
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
                raise ValueError(f"invalid {name}")
        if self.gamma > 1 or self.gae_lambda > 1 or self.clip_coef >= 1:
            raise ValueError("discounts must be <=1 and policy clipping <1")
        if not self.learning_rate or not self.max_grad_norm or not self.target_kl:
            raise ValueError("learning rate, gradient norm, and target KL must be positive")
        for name in ("epochs", "minibatch_size", "max_records", "max_padded_candidates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"positive integer {name} required")
        if type(self.normalize_advantages) is not bool:
            raise ValueError("normalize_advantages must be boolean")


def _network(input_size, hidden_sizes):
    layers = []
    for size in hidden_sizes:
        layers.extend((nn.Linear(input_size, size), nn.Tanh()))
        input_size = size
    return nn.Sequential(*layers, nn.Linear(input_size, 1))


class _Actor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.network = _network(config.state_dim + config.candidate_dim, config.hidden_sizes)

    def forward(self, states, candidates, mask):
        context = states[:, None].expand(-1, candidates.shape[1], -1)
        return self.network(torch.cat((context, candidates), -1)).squeeze(-1).masked_fill(~mask, -torch.inf)


class MaskedActorCritic(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        self.actor = _Actor(config)
        self.critic = _network(config.state_dim, config.hidden_sizes)

    def load_actor_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict, strict=True)
        _finite_parameters(self.actor)

    def forward(self, states, candidates, mask):
        c = self.config
        if (states.ndim != 2 or states.shape[1] != c.state_dim or candidates.ndim != 3
                or candidates.shape[0] != states.shape[0] or candidates.shape[2] != c.candidate_dim
                or mask.shape != candidates.shape[:2] or mask.dtype != torch.bool or not states.shape[0]
                or not candidates.shape[1] or not mask.any(1).all()):
            raise ValueError("invalid state, candidate, or legal-mask shape")
        if (not states.is_floating_point() or not candidates.is_floating_point()
                or not torch.isfinite(states).all() or not torch.isfinite(candidates).all()
                or states.device != candidates.device or states.device != mask.device
                or states.device != next(self.parameters()).device
                or states.dtype != next(self.parameters()).dtype or candidates.dtype != states.dtype):
            raise ValueError("finite floating inputs on the model device required")
        logits, values = self.actor(states, candidates, mask), self.critic(states).squeeze(-1)
        if not torch.isfinite(logits[mask]).all() or not torch.isfinite(values).all():
            raise ValueError("nonfinite actor or critic output")
        return Categorical(logits=logits), values


def gae_targets(rewards, values, next_values, terminated, boundaries, gamma, lam):
    tensors = (rewards, values, next_values, terminated, boundaries)
    if (rewards.ndim not in (1, 2) or not rewards.shape[0]
            or any(x.shape != rewards.shape or x.device != rewards.device for x in tensors)
            or terminated.dtype != torch.bool or boundaries.dtype != torch.bool
            or any(not x.is_floating_point() or not torch.isfinite(x).all() for x in tensors[:3])
            or not 0 <= gamma <= 1 or not 0 <= lam <= 1):
        raise ValueError("invalid GAE tensor contract")
    with torch.no_grad():
        deltas = rewards + gamma * (~terminated) * next_values - values
        advantages, carry = torch.empty_like(values), torch.zeros_like(values[0])
        for t in range(len(values) - 1, -1, -1):
            carry = deltas[t] + gamma * lam * (~(boundaries[t] | terminated[t])) * carry
            advantages[t] = carry
        return advantages, advantages + values


def ppo_loss(new_logp, old_logp, advantages, new_values, old_values, returns, entropy, config):
    logratio = new_logp - old_logp
    ratio = logratio.exp()
    policy = -torch.minimum(ratio * advantages,
                            ratio.clamp(1 - config.clip_coef, 1 + config.clip_coef) * advantages).mean()
    value_error = (new_values - returns).square()
    clipped_values = old_values + (new_values - old_values).clamp(-config.value_clip_coef, config.value_clip_coef)
    value = .5 * torch.maximum(value_error, (clipped_values - returns).square()).mean()
    entropy_mean = entropy.mean()
    loss = policy + config.value_coef * value - config.entropy_coef * entropy_mean
    return loss, {"policy_loss": policy, "value_loss": value, "entropy": entropy_mean,
                  "approx_kl": ((ratio - 1) - logratio).mean(),
                  "clip_fraction": ((ratio - 1).abs() > config.clip_coef).float().mean()}


def explained_variance(predictions, targets):
    variance = targets.var(unbiased=False)
    return None if variance.item() == 0 else float((1 - (targets - predictions).var(unbiased=False) / variance).item())


def _number(x, name):
    if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or abs(x) > 1e6:
        raise ValueError(f"finite bounded {name} required")
    return x


def _features(record, config):
    if not isinstance(record, dict) or record.get("schema") != config.feature_schema:
        raise ValueError("feature schema mismatch")
    if record.get("kind", "ordinary") != "ordinary":
        raise ValueError("only ordinary learned decisions enter PPO")
    state, candidates = record.get("state"), record.get("candidates")
    if not isinstance(state, list) or len(state) != config.state_dim:
        raise ValueError("invalid state vector")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("nonempty complete legal candidates required")
    for row, size in [(state, config.state_dim), *((row, config.candidate_dim) for row in candidates)]:
        if not isinstance(row, list) or len(row) != size:
            raise ValueError("invalid feature vector dimension")
        for x in row:
            _number(x, "feature")


def pad_records(records, config, device="cpu", max_padded_candidates=131072):
    if not records:
        raise ValueError("nonempty records required")
    for record in records:
        _features(record, config)
    width = max(len(r["candidates"]) for r in records)
    if width * len(records) > max_padded_candidates:
        raise ValueError("padded candidate allocation budget exceeded; no candidates were truncated")
    states = torch.tensor([r["state"] for r in records], dtype=torch.float32, device=device)
    candidates = torch.zeros((len(records), width, config.candidate_dim), device=device)
    mask = torch.zeros((len(records), width), dtype=torch.bool, device=device)
    for i, record in enumerate(records):
        n = len(record["candidates"])
        candidates[i, :n] = torch.tensor(record["candidates"], dtype=torch.float32, device=device)
        mask[i, :n] = True
    return states, candidates, mask


@dataclass
class Rollout:
    records: list
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


def prepare_rollout(records, model_config, ppo_config, device="cpu"):
    if not isinstance(records, list) or not 1 <= len(records) <= ppo_config.max_records:
        raise ValueError("rollout record budget exceeded or empty")
    episodes = set()
    for i, record in enumerate(records):
        _features(record, model_config)
        if len(record["candidates"]) > ppo_config.max_padded_candidates:
            raise ValueError("single decision exceeds candidate budget")
        episode, step = record.get("episodeId"), record.get("stepIndex")
        if not isinstance(episode, str) or not episode or len(episode) > 512 or type(step) is not int or step < 0:
            raise ValueError("episodeId and nonnegative stepIndex required")
        for flag in ("terminated", "truncated", "segmentEnd"):
            if type(record.get(flag)) is not bool:
                raise ValueError(f"boolean {flag} required")
        if record["terminated"] and record["truncated"]:
            raise ValueError("a transition cannot terminate and truncate")
        if (record["terminated"] or record["truncated"]) and not record["segmentEnd"]:
            raise ValueError("episode endings require segmentEnd")
        if type(record.get("actionIndex")) is not int or not 0 <= record["actionIndex"] < len(record["candidates"]):
            raise ValueError("actionIndex must index a legal candidate")
        for key in ("oldLogProb", "oldValue", "reward", "nextValue"):
            _number(record.get(key), key)
        if record["oldLogProb"] > 1e-6:
            raise ValueError("log probability cannot be positive")
        if i and records[i - 1]["episodeId"] == episode:
            previous = records[i - 1]
            if previous["terminated"] or previous["truncated"] or step != previous["stepIndex"] + 1:
                raise ValueError("episode steps must be contiguous without reset")
            if not previous["segmentEnd"] and not math.isclose(previous["nextValue"], record["oldValue"], abs_tol=1e-4, rel_tol=1e-5):
                raise ValueError("nextValue must match next decision oldValue inside a segment")
        else:
            if episode in episodes or i and not records[i - 1]["segmentEnd"]:
                raise ValueError("episode regrouping or unmarked episode boundary")
            episodes.add(episode)
    if not records[-1]["segmentEnd"]:
        raise ValueError("last record must mark segmentEnd")
    tensor = lambda key, dtype=torch.float32: torch.tensor([r[key] for r in records], dtype=dtype, device=device)
    values, old_logp = tensor("oldValue"), tensor("oldLogProb")
    advantages, returns = gae_targets(tensor("reward"), values, tensor("nextValue"),
                                     tensor("terminated", torch.bool), tensor("segmentEnd", torch.bool),
                                     ppo_config.gamma, ppo_config.gae_lambda)
    return Rollout(records, old_logp, values, advantages, returns)


def _finite_parameters(model):
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("nonfinite model parameters")


def _private_path(path):
    return bc.private_checkpoint(path)


class PPOTrainer:
    def __init__(self, model_config=ModelConfig(), config=PPOConfig(), seed=1729, device="cpu"):
        if type(seed) is not int or not 0 <= seed <= 0xffffffff:
            raise ValueError("bounded integer seed required")
        if str(device) not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        self.model_config, self.config, self.seed = model_config, config, seed
        self.device = torch.device(device)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = MaskedActorCritic(model_config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.update_count, self.printing_aliases = 0, {}

    def warm_start(self, saved_bc_payload):
        if (saved_bc_payload.get("featureContract") != bc.CONTRACT
                or saved_bc_payload.get("contractSha256") != bc.CONTRACT_SHA256
                or self.model_config != ModelConfig()):
            raise ValueError("BC feature contract or model architecture mismatch")
        aliases = bc.printing_aliases(saved_bc_payload.get("printingAliases", {}))
        if self.update_count or self.optimizer.state:
            raise ValueError("warm start requires a fresh trainer")
        self.model.load_actor_state_dict(saved_bc_payload["state_dict"])
        self.printing_aliases = dict(aliases)

    def act(self, records, deterministic=False):
        if type(deterministic) is not bool:
            raise ValueError("deterministic must be boolean")
        self.model.eval()
        with torch.no_grad():
            dist, values = self.model(*pad_records(records, self.model_config, self.device, self.config.max_padded_candidates))
            actions = dist.logits.argmax(1) if deterministic else torch.multinomial(dist.probs.cpu(), 1, generator=self.generator).squeeze(1).to(self.device)
            logp = dist.log_prob(actions)
        return [{"actionIndex": int(a), "oldLogProb": float(p), "oldValue": float(v)}
                for a, p, v in zip(actions.cpu(), logp.cpu(), values.cpu())]

    def values(self, states):
        if not isinstance(states, list) or not states:
            raise ValueError("nonempty state list required")
        for state in states:
            if not isinstance(state, list) or len(state) != self.model_config.state_dim:
                raise ValueError("invalid state dimension")
            for x in state:
                _number(x, "state")
        with torch.no_grad():
            result = self.model.critic(torch.tensor(states, dtype=torch.float32, device=self.device)).squeeze(-1)
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite critic value")
        return result.cpu().tolist()

    def _evaluate(self, records):
        logps, values, entropies = [], [], []
        with torch.no_grad():
            for indices in self._minibatches(records, list(range(len(records)))):
                rows = [records[i] for i in indices]
                dist, value = self.model(*pad_records(rows, self.model_config, self.device, self.config.max_padded_candidates))
                actions = torch.tensor([r["actionIndex"] for r in rows], device=self.device)
                logps.append(dist.log_prob(actions))
                values.append(value)
                entropies.append(dist.entropy())
        return tuple(torch.cat(items) for items in (logps, values, entropies))

    def _minibatches(self, records, order):
        batch, width = [], 0
        for index in order:
            next_width = max(width, len(records[index]["candidates"]))
            if batch and (len(batch) == self.config.minibatch_size
                          or next_width * (len(batch) + 1) > self.config.max_padded_candidates):
                yield batch
                batch, width = [], 0
            batch.append(index)
            width = max(width, len(records[index]["candidates"]))
        if batch:
            yield batch

    def update(self, records):
        cfg = self.config
        rollout = prepare_rollout(records, self.model_config, cfg, self.device)
        initial_logp, initial_values, _ = self._evaluate(records)
        if not torch.allclose(initial_logp, rollout.old_log_probs, atol=1e-4, rtol=0) or not torch.allclose(initial_values, rollout.old_values, atol=1e-4, rtol=0):
            raise ValueError("stale or off-policy rollout log probabilities or values")
        advantages = rollout.advantages
        if cfg.normalize_advantages and len(records) > 1:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        initial = [p.detach().clone() for p in self.model.parameters()]
        steps, grad_norm, stop_reason = 0, 0., "epochs_completed"
        self.model.train()
        for epoch in range(cfg.epochs):
            order = torch.randperm(len(records), generator=self.generator).tolist()
            for indices in self._minibatches(records, order):
                rows = [records[i] for i in indices]
                dist, new_values = self.model(*pad_records(rows, self.model_config, self.device, cfg.max_padded_candidates))
                actions = torch.tensor([r["actionIndex"] for r in rows], device=self.device)
                loss, _ = ppo_loss(dist.log_prob(actions), rollout.old_log_probs[indices], advantages[indices],
                                   new_values, rollout.old_values[indices], rollout.returns[indices], dist.entropy(), cfg)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                    raise ValueError("missing or nonfinite PPO gradient")
                norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                grad_norm = max(grad_norm, float(norm))
                self.optimizer.step()
                _finite_parameters(self.model)
                steps += 1
            logp, values, entropy = self._evaluate(records)
            _, metrics = ppo_loss(logp, rollout.old_log_probs, advantages, values,
                                   rollout.old_values, rollout.returns, entropy, cfg)
            if not all(torch.isfinite(v) for v in metrics.values()):
                raise ValueError("nonfinite PPO diagnostics")
            if metrics["approx_kl"].item() > cfg.target_kl:
                stop_reason = "target_kl"
                break
        self.update_count += 1
        change = math.sqrt(sum(float((p.detach() - before).square().sum()) for p, before in zip(self.model.parameters(), initial)))
        return {**{k: float(v) for k, v in metrics.items()}, "optimizer_steps": steps,
                "epochs": epoch + 1, "stop_reason": stop_reason, "weight_change_l2": change,
                "grad_norm": grad_norm, "explained_variance": explained_variance(values, rollout.returns),
                "update_count": self.update_count, "records": len(records)}

    def save(self, path, metadata=None):
        path = _private_path(path)
        if path.exists():
            raise ValueError("checkpoint exists; choose a fresh path")
        _finite_parameters(self.model)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": CHECKPOINT_SCHEMA, "model_config": asdict(self.model_config),
                   "ppo_config": asdict(self.config), "seed": self.seed,
                   "state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                   "rng_state": self.generator.get_state(), "update_count": self.update_count,
                   "featureContract": bc.CONTRACT if self.model_config == ModelConfig() else asdict(self.model_config),
                   "contractSha256": bc.CONTRACT_SHA256 if self.model_config == ModelConfig() else None,
                   "sourceSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "printingAliases": self.printing_aliases, "metadata": metadata or {}}
        fd, temporary = tempfile.mkstemp(prefix=".ppo-", suffix=".pt", dir=path.parent)
        os.close(fd)
        try:
            torch.save(payload, temporary)
            if os.name == "nt":
                os.rename(temporary, path)
            else:
                os.link(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return str(path)

    @classmethod
    def load(cls, path, device="cpu"):
        payload = torch.load(_private_path(path), map_location=device, weights_only=True)
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("PPO checkpoint schema mismatch")
        trainer = cls(ModelConfig(**payload["model_config"]), PPOConfig(**payload["ppo_config"]), payload["seed"], device)
        expected = bc.CONTRACT if trainer.model_config == ModelConfig() else asdict(trainer.model_config)
        if payload.get("featureContract") != expected or trainer.model_config == ModelConfig() and payload.get("contractSha256") != bc.CONTRACT_SHA256:
            raise ValueError("PPO checkpoint feature contract mismatch")
        trainer.model.load_state_dict(payload["state_dict"], strict=True)
        _finite_parameters(trainer.model)
        trainer.optimizer.load_state_dict(payload["optimizer"])
        if any(not torch.isfinite(value).all() for state in trainer.optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor)):
            raise ValueError("nonfinite optimizer state")
        trainer.generator.set_state(payload["rng_state"].cpu())
        trainer.update_count = payload["update_count"]
        trainer.printing_aliases = dict(bc.printing_aliases(payload.get("printingAliases", {})))
        return trainer
