import copy
import tempfile
from pathlib import Path
import unittest
import torch
from learning import alphazero as az
from learning import entity_ppo as native
from learning.test_joint_choice_sets import record


def target(outcome=1):
    return {**record(), "schema": az.RECORD_SCHEMA, "trainingMethod": az.METHOD,
            "episodeId": "mathematical-unit-check", "stepIndex": 0, "completedNaturally": True,
            "outcome": outcome, "searchVisits": [1, 0, 1, 1, 2, 2, 3],
            "searchPolicy": [.1, 0., .1, .1, .2, .2, .3]}


class AlphaZeroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_policy_and_outcome_targets_train_both_heads_without_ppo_data(self):
        trainer = az.AlphaZeroTrainer({"cardVocabularySize": 1, "actionVocabularySize": 1})
        before = copy.deepcopy(trainer.model.state_dict())
        metrics = trainer.update([target(1), target(-1)])
        after = trainer.model.state_dict()
        self.assertEqual(metrics["epochs"], 1)
        self.assertTrue(any(not torch.equal(value, after[name]) for name, value in before.items() if name.startswith("actor.")))
        self.assertTrue(any(not torch.equal(value, after[name]) for name, value in before.items() if name.startswith("critic.")))
        values = trainer.score([record()])["values"]
        self.assertTrue(all(-1 <= value <= 1 for value in values))
        self.assertEqual(trainer.config["weight_decay"], .0001)

    def test_fallback_forced_or_non_outcome_targets_are_rejected(self):
        for patch in ({"completedNaturally": False}, {"oldLogProb": -.1}, {"searchVisits": [0] * 7},
                      {"searchPolicy": [.5, 0., .5, 0., 0., 0., 0.]}, {"legalMask": [True] + [False] * 6}):
            row = {**target(), **patch}
            with self.assertRaises(ValueError): az.validate_target(row)

    def test_original_actor_critic_tensors_fit_new_bounded_value_model_exactly(self):
        catalog = {"cardVocabularySize": 1, "actionVocabularySize": 1}
        baseline = native.EntityActorCritic(catalog)
        trainer = az.AlphaZeroTrainer(catalog)
        trainer.model.load_state_dict(baseline.state_dict(), strict=True)
        self.assertFalse(trainer.optimizer.state)
        self.assertTrue(all(torch.equal(value, trainer.model.state_dict()[name]) for name, value in baseline.state_dict().items()))
        batch = az.pad_records([record()], "cpu", catalog)
        base = {key: value for key, value in batch.items() if not key.startswith('choice_set_')}
        with torch.no_grad():
            old_policy, old_value = baseline(base); new_policy, new_value = trainer.model(batch)
        expected = az.aggregate_choice_logits(baseline.actor(**base), batch['choice_set_indices'], batch['choice_set_members'], batch['choice_set_mask'])
        self.assertTrue(torch.equal(torch.distributions.Categorical(logits=expected).logits, new_policy.logits))
        self.assertTrue(torch.equal(old_value.tanh(), new_value))


if __name__ == "__main__": unittest.main()
