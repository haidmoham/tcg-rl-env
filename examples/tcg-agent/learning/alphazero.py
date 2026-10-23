"""Search-policy cross entropy and natural-outcome value learning on native public records."""

import math
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Categorical
from learning import entity_ppo as native
from learning import model6_torch as features
from learning.joint_choice_sets import action_count, checked_choice_sets, attach_choice_sets, aggregate_choice_logits

SCHEMA = "native-alphazero/1"
RECORD_SCHEMA = "native-alphazero-record/1"
METHOD = "alphazero-search-outcome/1"
CONFIG = {"learning_rate": .0001, "max_grad_norm": .5, "epochs": 1, "minibatch_size": 32,
          "value_coefficient": 1., "weight_decay": .0001}


class SearchActorCritic(native.EntityActorCritic):
    def forward(self, batch):
        base = {key: value for key, value in batch.items() if not key.startswith("choice_set_")}
        logits = self.actor(**base)
        values = self.critic(self.actor.visible_context(base)).squeeze(-1).tanh()
        if not torch.isfinite(logits[base["candidate_mask"]]).all() or not torch.isfinite(values).all():
            raise ValueError("nonfinite AlphaZero policy/value output")
        if "choice_set_indices" in batch:
            logits = aggregate_choice_logits(logits, batch["choice_set_indices"], batch["choice_set_members"], batch["choice_set_mask"])
        return Categorical(logits=logits), values


def pad_records(records, device, catalog):
    batch, _ = features.padded_batch(records, device, catalog)
    for index, record in enumerate(records):
        sets = checked_choice_sets(record)
        mask = record.get("legalMask")
        if mask is not None:
            if (not isinstance(mask, list) or len(mask) != action_count(record)
                    or any(type(flag) is not bool for flag in mask) or not any(mask)):
                raise ValueError("complete nonempty native legal mask required")
            if sets is None:
                batch["candidate_mask"][index, :len(mask)] &= torch.tensor(mask, device=device)
        if sets is not None and mask is None:
            raise ValueError("every joint choice set requires an explicit native validation mask")
        if record["kind"] == "selection" and sets is None and record.get("selectionCardinality") != 1:
            raise ValueError("selection needs a complete joint domain or explicit singleton")
    return attach_choice_sets(batch, records, device)


def validate_target(row):
    if (row.get("schema") != RECORD_SCHEMA or row.get("trainingMethod") != METHOD
            or row.get("completedNaturally") is not True or type(row.get("outcome")) not in (int, float)
            or row["outcome"] not in (-1, 0, 1) or not isinstance(row.get("episodeId"), str)
            or not row["episodeId"] or type(row.get("stepIndex")) is not int or row["stepIndex"] < 0
            or any(key in row for key in ("oldLogProb", "oldValue", "behaviorPolicy", "advantages"))):
        raise ValueError("natural-outcome search record required; PPO behavior data is a different objective")
    visits, target, count = row.get("searchVisits"), row.get("searchPolicy"), action_count(row)
    mask = row.get("legalMask", [True] * count)
    if (not isinstance(visits, list) or len(visits) != count
            or any(type(value) is not int or value < 0 for value in visits) or sum(visits) < 1
            or not isinstance(target, list) or len(target) != count
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in target)
            or not math.isclose(sum(target), 1., abs_tol=1e-6)
            or not isinstance(mask, list) or len(mask) != count or any(type(value) is not bool for value in mask)
            or sum(mask) < 2
            or any((not legal and (visit or probability)) for legal, visit, probability in zip(mask, visits, target))
            or any(not math.isclose(probability, visit / sum(visits), abs_tol=1e-6)
                   for probability, visit in zip(target, visits))):
        raise ValueError("policy target must be the complete normalized legal root-visit distribution")


def feature_record(row):
    return {**row, "schema": features.SCHEMA}


class AlphaZeroTrainer:
    def __init__(self, catalog, architecture=None, seed=81729, device="cpu", config=None):
        self.device = features.configure(device, seed)
        self.catalog, self.seed = catalog, seed
        self.architecture = features.resolve_architecture(architecture)
        self.config = {**CONFIG, **(config or {})}
        if self.config != CONFIG:
            raise ValueError("explicit initial AlphaZero configuration differs from the declared recipe")
        self.model = SearchActorCritic(catalog, self.architecture).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"],
                                          weight_decay=self.config["weight_decay"])
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.update_count, self.metadata = 0, {}

    def score(self, records):
        self.model.eval()
        with torch.inference_mode():
            distribution, values = self.model(pad_records(records, self.device, self.catalog))
        return {"logits": [[float(value) if math.isfinite(float(value)) else None for value in row]
                          for row in distribution.logits.cpu()], "values": values.cpu().tolist()}

    def loss(self, rows):
        for row in rows:
            validate_target(row)
        records = [feature_record(row) for row in rows]
        distribution, values = self.model(pad_records(records, self.device, self.catalog))
        target = torch.zeros_like(distribution.logits)
        for index, row in enumerate(rows):
            target[index, :len(row["searchPolicy"])] = torch.tensor(row["searchPolicy"], device=self.device)
        log_policy = torch.where(target > 0, distribution.logits, torch.zeros_like(distribution.logits))
        policy_loss = -(target * log_policy).sum(-1).mean()
        outcomes = torch.tensor([row["outcome"] for row in rows], dtype=torch.float32, device=self.device)
        value_loss = (values - outcomes).square().mean()
        return policy_loss + value_loss, {"policyCrossEntropy": policy_loss, "valueMse": value_loss,
                                          "entropy": distribution.entropy().mean()}

    def update(self, rows):
        if not isinstance(rows, list) or not 1 <= len(rows) <= 32768:
            raise ValueError("bounded actual search records required")
        for row in rows:
            validate_target(row)
            pad_records([feature_record(row)], "cpu", self.catalog)
        order = torch.randperm(len(rows), generator=self.generator).tolist()
        totals = {"policyCrossEntropy": 0., "valueMse": 0., "entropy": 0.}
        maximum_norm, steps = 0., 0
        self.model.train()
        for start in range(0, len(rows), self.config["minibatch_size"]):
            batch = [rows[index] for index in order[start:start + self.config["minibatch_size"]]]
            loss, metrics = self.loss(batch)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite AlphaZero loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"], error_if_nonfinite=True)
            maximum_norm = max(maximum_norm, float(norm))
            self.optimizer.step()
            native.finite_parameters(self.model)
            for name, value in metrics.items():
                totals[name] += float(value.detach()) * len(batch)
            steps += 1
        self.update_count += 1
        return {**{name: total / len(rows) for name, total in totals.items()}, "records": len(rows),
                "optimizerSteps": steps, "epochs": 1, "maxGradientNormBeforeClip": maximum_norm,
                "updateCount": self.update_count}

    def payload(self):
        native.finite_parameters(self.model)
        return {"schema": SCHEMA, "trainingMethod": METHOD, "featureContract": features.CONTRACT,
                "contractSha256": features.CONTRACT_SHA256, "catalog": self.catalog,
                "architecture": self.architecture, "valueTransform": "tanh/1", "config": self.config,
                "seed": self.seed, "updateCount": self.update_count, "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "rngState": self.generator.get_state(), "metadata": self.metadata}

    @classmethod
    def load(cls, path, catalog, device="cpu"):
        saved = torch.load(features.private_checkpoint(path), map_location=device, weights_only=True)
        if (saved.get("schema") != SCHEMA or saved.get("trainingMethod") != METHOD
                or saved.get("featureContract") != features.CONTRACT or saved.get("contractSha256") != features.CONTRACT_SHA256
                or saved.get("catalog") != catalog or saved.get("valueTransform") != "tanh/1"):
            raise ValueError("AlphaZero checkpoint contract mismatch")
        trainer = cls(catalog, saved["architecture"], saved["seed"], device, saved["config"])
        trainer.model.load_state_dict(saved["state_dict"], strict=True)
        trainer.optimizer.load_state_dict(saved["optimizer"])
        trainer.generator.set_state(saved["rngState"].cpu())
        if type(saved.get("updateCount")) is not int or saved["updateCount"] < 0:
            raise ValueError("invalid AlphaZero update count")
        trainer.update_count, trainer.metadata = saved["updateCount"], saved["metadata"]
        native.finite_parameters(trainer.model)
        return trainer
