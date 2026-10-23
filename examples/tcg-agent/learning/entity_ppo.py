"""Entity-policy PPO with a bit-identical Model5 actor at warm start."""

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Categorical
from learning import model6_torch as bc
from learning.masked_ppo import PPOConfig, gae_targets, ppo_loss, explained_variance

SCHEMA = "entity-masked-ppo/1"
DEFAULT_CONFIG = {"gamma": .99, "gae_lambda": .95, "epochs": 4, "learning_rate": .0001,
                  "clip_coef": .15, "value_clip_coef": .2, "target_kl": .02, "entropy_coef": .001,
                  "minibatch_size": 32, "max_records": 16384, "max_padded_candidates": 32768}
DEFAULT_RATIONALE = "Control uses gamma .99 and entropy .001; the gamma one arm changes only discount. A separate gamma .99 entropy .005 arm changes only entropy. Lambda .95, four epochs, learning rate .0001, clip .15 and target KL .02 are shared considered starting choices, not proven optimal."


def finite_parameters(model):
    if any(not torch.isfinite(p).all() or (p.abs() > 1e6).any() for p in model.parameters()):
        raise ValueError("invalid PPO model parameters")


def pad_records(records, device, catalog):
    batch, _ = bc.padded_batch(records, device, catalog)
    for index, record in enumerate(records):
        mask = record.get("legalMask")
        if mask is not None:
            if (not isinstance(mask, list) or len(mask) != len(record["candidates"])
                    or any(type(flag) is not bool for flag in mask) or not any(mask)):
                raise ValueError("legalMask must cover every retained offer and admit at least one")
            batch["candidate_mask"][index, :len(mask)] &= torch.tensor(mask, dtype=torch.bool, device=device)
        if record["kind"] == "selection" and record.get("selectionCardinality") != 1:
            raise ValueError("PPO selection must explicitly have cardinality one")
    return batch


class EntityActorCritic(nn.Module):
    def __init__(self, catalog, architecture=None):
        super().__init__()
        self.actor = bc.EntityScorer(catalog, architecture=architecture)
        self.critic = nn.Sequential(nn.Linear(272, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, batch):
        logits = self.actor(**batch)
        context = self.actor.visible_context(batch)
        values = self.critic(context).squeeze(-1)
        if not torch.isfinite(logits[batch["candidate_mask"]]).all() or not torch.isfinite(values).all():
            raise ValueError("nonfinite entity PPO output")
        return Categorical(logits=logits), values


class EntityPPOTrainer:
    def __init__(self, catalog, config=None, seed=1729, device="cpu", architecture=None):
        self.device = bc.configure(device, seed)
        self.catalog, self.seed = catalog, seed
        self.architecture = bc.resolve_architecture(architecture)
        self.config = PPOConfig(**{**DEFAULT_CONFIG, **(config or {})})
        cfg = self.config
        if (not 1 <= cfg.epochs <= 4 or not 1 <= cfg.minibatch_size <= 64 or cfg.max_records > 16384
                or cfg.max_padded_candidates > 32768 or cfg.gamma not in (1, .99) or cfg.gae_lambda != .95
                or cfg.learning_rate not in (.0001, .00003) or cfg.clip_coef not in (.15, .2)
                or cfg.target_kl > .02 or cfg.entropy_coef not in (.001, .005)):
            raise ValueError("bounded initial entity PPO configuration required")
        self.model = EntityActorCritic(catalog, architecture=self.architecture).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.update_count = 0
        self.metadata = {}

    def warm_start(self, checkpoint, probe):
        if self.update_count or self.optimizer.state:
            raise ValueError("warm start requires a fresh trainer")
        baseline, saved = bc.load_checkpoint(checkpoint, self.device, self.catalog)
        if baseline.architecture != self.architecture:
            raise ValueError("pre-PPO actor architecture differs from initialized trainer")
        self.model.actor.load_state_dict(baseline.state_dict(), strict=True)
        self.model.requires_grad_(True)
        batch, _ = bc.padded_batch(probe, self.device, self.catalog)
        with torch.inference_mode():
            expected, actual = baseline(**batch), self.model.actor(**batch)
        if (not torch.equal(expected, actual) or any(not torch.equal(value, self.model.actor.state_dict()[name])
                                                   for name, value in baseline.state_dict().items())):
            raise ValueError("same-contract pre-PPO actor state or actual-record raw logits changed at initialization")
        return {"warmStartActorStateExact": True, "warmStartActorLogitsExact": True, "probeRecords": len(probe),
                "baseModel5CheckpointSha256": saved["baseModel5CheckpointSha256"]}

    def counts(self):
        actor = sum(p.numel() for p in self.model.actor.parameters())
        critic = sum(p.numel() for p in self.model.critic.parameters())
        base = sum(p.numel() for p in self.model.actor.base.parameters())
        return {"actorParameters": actor, "baseModel5Parameters": base, "energyExtraParameters": actor - base,
                "criticExtraParameters": critic,
                "trainableParameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)}

    def act(self, records, deterministic=False):
        if type(deterministic) is not bool:
            raise ValueError("explicit boolean deterministic required")
        self.model.eval()
        with torch.inference_mode():
            distribution, values = self.model(pad_records(records, self.device, self.catalog))
            actions = (distribution.logits.argmax(1) if deterministic else
                       torch.multinomial(distribution.probs.cpu(), 1, generator=self.generator).squeeze(1).to(self.device))
            logp = distribution.log_prob(actions)
        return [{"actionIndex": int(action), "oldLogProb": float(probability), "oldValue": float(value)}
                for action, probability, value in zip(actions.cpu(), logp.cpu(), values.cpu())]

    def values(self, records):
        self.model.eval()
        with torch.inference_mode():
            _, values = self.model(pad_records(records, self.device, self.catalog))
        return values.cpu().tolist()

    def evaluate(self, records):
        logps, values, entropies = [], [], []
        with torch.inference_mode():
            for start in range(0, len(records), self.config.minibatch_size):
                rows = records[start:start + self.config.minibatch_size]
                distribution, value = self.model(pad_records(rows, self.device, self.catalog))
                actions = torch.tensor([row["actionIndex"] for row in rows], device=self.device)
                logps.append(distribution.log_prob(actions)); values.append(value); entropies.append(distribution.entropy())
        return tuple(torch.cat(items) for items in (logps, values, entropies))

    def update(self, records):
        cfg = self.config
        if not isinstance(records, list) or not 2 <= len(records) <= cfg.max_records:
            raise ValueError("bounded PPO rollout with at least two decisions required")
        seen, previous = set(), None
        for row in records:
            pad_records([row], "cpu", self.catalog)
            identity, step = row.get("episodeId"), row.get("stepIndex")
            if not isinstance(identity, str) or not identity or type(step) is not int or step < 0:
                raise ValueError("episode identity and step required")
            if any(type(row.get(key)) is not bool for key in ("terminated", "truncated", "segmentEnd")):
                raise ValueError("explicit episode ending flags required")
            if row["terminated"] and row["truncated"] or (row["terminated"] or row["truncated"]) != row["segmentEnd"]:
                raise ValueError("only complete episode boundaries are admitted")
            if previous and identity == previous["episodeId"]:
                if previous["segmentEnd"] or step != previous["stepIndex"] + 1 or not math.isclose(previous["nextValue"], row.get("oldValue", math.inf), abs_tol=1e-4):
                    raise ValueError("episode continuity or bootstrap mismatch")
            elif identity in seen or step != 0 or previous and not previous["segmentEnd"]:
                raise ValueError("episode regrouping or incomplete episode")
            seen.add(identity)
            index = row.get("actionIndex")
            if type(index) is not int or not 0 <= index < len(row["candidates"]) or not row.get("legalMask", [True] * len(row["candidates"]))[index]:
                raise ValueError("executed action must be a legal retained offer")
            for key in ("oldLogProb", "oldValue", "reward", "nextValue"):
                number = row.get(key)
                if type(number) not in (int, float) or not math.isfinite(number) or abs(number) > 1e6:
                    raise ValueError("finite bounded rollout scalars required")
            if row["oldLogProb"] > 1e-6 or row["reward"] not in (-1, 0, 1):
                raise ValueError("invalid rollout probability or outcome reward")
            if not row["terminated"] and row["reward"] != 0 or row["terminated"] and row["nextValue"] != 0:
                raise ValueError("only natural outcomes receive reward; terminal bootstrap must be zero")
            previous = row
        if not records[-1]["segmentEnd"]:
            raise ValueError("complete final episode required")
        tensor = lambda key, dtype=torch.float32: torch.tensor([row[key] for row in records], dtype=dtype, device=self.device)
        old_logp, old_values = tensor("oldLogProb"), tensor("oldValue")
        self.model.eval()
        initial_logp, initial_values, _ = self.evaluate(records)
        if not torch.allclose(initial_logp, old_logp, atol=1e-4, rtol=0) or not torch.allclose(initial_values, old_values, atol=1e-4, rtol=0):
            raise ValueError("rollout differs from exact parent policy old log probabilities or values")
        advantages, returns = gae_targets(tensor("reward"), old_values, tensor("nextValue"),
            tensor("terminated", torch.bool), tensor("segmentEnd", torch.bool), cfg.gamma, cfg.gae_lambda)
        if cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        initial = [p.detach().clone() for p in self.model.parameters()]
        steps, maximum_norm, stop = 0, 0., "epochs-completed"
        for epoch in range(cfg.epochs):
            order = torch.randperm(len(records), generator=self.generator).tolist()
            self.model.train()
            for start in range(0, len(order), cfg.minibatch_size):
                indices = order[start:start + cfg.minibatch_size]
                rows = [records[index] for index in indices]
                distribution, values = self.model(pad_records(rows, self.device, self.catalog))
                actions = torch.tensor([row["actionIndex"] for row in rows], device=self.device)
                loss, _ = ppo_loss(distribution.log_prob(actions), old_logp[indices], advantages[indices],
                                  values, old_values[indices], returns[indices], distribution.entropy(), cfg)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True); loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                    raise ValueError("missing or nonfinite PPO gradient")
                norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                maximum_norm = max(maximum_norm, float(norm))
                self.optimizer.step(); finite_parameters(self.model); steps += 1
                self.model.eval()
                logp, values, entropy = self.evaluate(rows)
                _, metrics = ppo_loss(logp, old_logp[indices], advantages[indices], values,
                                      old_values[indices], returns[indices], entropy, cfg)
                if any(not torch.isfinite(item) for item in metrics.values()):
                    raise ValueError("nonfinite PPO diagnostics")
                if float(metrics["approx_kl"]) > cfg.target_kl:
                    stop = "target-kl"; break
                self.model.train()
            self.model.eval()
            logp, values, entropy = self.evaluate(records)
            _, metrics = ppo_loss(logp, old_logp, advantages, values, old_values, returns, entropy, cfg)
            if any(not torch.isfinite(item) for item in metrics.values()) or float(metrics["approx_kl"]) > cfg.target_kl * 4:
                raise ValueError("PPO final diagnostics nonfinite or hard KL limit exceeded")
            if float(metrics["approx_kl"]) > cfg.target_kl:
                stop = "target-kl"
            if stop == "target-kl":
                break
        self.update_count += 1
        return {**{key: float(value) for key, value in metrics.items()}, "optimizerSteps": steps,
                "epochs": epoch + 1, "stopReason": stop, "maxGradientNormBeforeClip": maximum_norm,
                "explainedVariance": explained_variance(values, returns), "updateCount": self.update_count,
                "weightChangeL2": math.sqrt(sum(float((p.detach() - before).square().sum()) for p, before in zip(self.model.parameters(), initial)))}

    def payload(self):
        finite_parameters(self.model)
        return {"schema": SCHEMA, "featureContract": bc.CONTRACT, "contractSha256": bc.CONTRACT_SHA256,
                "architecture": self.architecture, "catalog": self.catalog, "ppoConfig": asdict(self.config),
                "seed": self.seed, "updateCount": self.update_count, "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "rngState": self.generator.get_state(), "metadata": self.metadata}

    @classmethod
    def load(cls, path, catalog, device="cpu"):
        saved = torch.load(bc.private_checkpoint(path), map_location=device, weights_only=True)
        if (saved.get("schema") != SCHEMA or saved.get("featureContract") != bc.CONTRACT
                or saved.get("contractSha256") != bc.CONTRACT_SHA256
                or saved.get("catalog") != catalog):
            raise ValueError("entity PPO checkpoint contract mismatch")
        if saved.get("architecture") is None:
            raise ValueError("explicit saved PPO actor architecture required")
        architecture = bc.resolve_architecture(saved["architecture"])
        trainer = cls(catalog, saved["ppoConfig"], saved["seed"], device, architecture=architecture)
        trainer.model.load_state_dict(saved["state_dict"], strict=True)
        trainer.optimizer.load_state_dict(saved["optimizer"])
        finite_parameters(trainer.model)
        if any(not torch.isfinite(value).all() for state in trainer.optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor)):
            raise ValueError("nonfinite optimizer state")
        if type(saved.get("updateCount")) is not int or saved["updateCount"] < 0:
            raise ValueError("invalid update count")
        trainer.generator.set_state(saved["rngState"].cpu())
        trainer.update_count, trainer.metadata = saved["updateCount"], saved["metadata"]
        return trainer
