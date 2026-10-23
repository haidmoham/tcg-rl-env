import copy
import math
import unittest
import torch
from learning import alphazero as ppo
from learning import model6_torch as bc
from learning.joint_choice_sets import checked_choice_sets, aggregate_choice_logits


def record(kind="selection", joint=True):
    value = {"schema": bc.SCHEMA, "kind": kind, "globals": [0.] * 67,
             "entities": [], "entityCardIds": [], "entityParents": [],
             "candidates": [[0.] * 78 for _ in range(3)], "candidateSources": [-1] * 3,
             "candidateTargets": [-1] * 3, "candidateActors": [-1] * 3, "candidateActionIds": [0] * 3,
             "originSourceIndex": -1, "originActionId": 0, "energyEntities": [],
             "energyCandidates": [[0.] * 64 for _ in range(3)], "energyAttacks": [],
             "energyAttackEntities": [], "energyAttackActionIds": [], "energyCandidateAttacks": [],
             "energyCandidateAttackCandidates": [], "energyCandidateAttackActionIds": []}
    if joint:
        value.update(choiceSets=[[], [0], [1], [2], [0, 1], [0, 2], [1, 2]], choiceSetMinimum=0,
                     choiceSetMaximum=2, selectionCardinality="joint-unordered",
                     legalMask=[True, False, True, True, True, True, True])
    elif kind == "selection":
        value.update(selectionCardinality=1, legalMask=[True, False, True])
    return value


class JointChoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_set_logits_probabilities_and_gradient(self):
        logits = torch.tensor([[math.log(2), math.log(3), math.log(5)]], requires_grad=True)
        batch = ppo.pad_records([record()], "cpu", {"cardVocabularySize": 1, "actionVocabularySize": 1})
        actual = aggregate_choice_logits(logits, batch["choice_set_indices"], batch["choice_set_members"], batch["choice_set_mask"])
        expected_weights = torch.tensor([[1., 0., 3., 5., 6., 10., 15.]])
        self.assertTrue(torch.allclose(actual.softmax(-1), expected_weights / 40))
        distribution = torch.distributions.Categorical(logits=actual)
        self.assertAlmostEqual(distribution.log_prob(torch.tensor([6])).item(), math.log(15 / 40), places=6)
        (-distribution.log_prob(torch.tensor([6]))).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_full_domain_and_mask_validation(self):
        for mutate in (lambda r: r["choiceSets"].pop(), lambda r: r["choiceSets"].reverse(),
                       lambda r: r.update(legalMask=[True]), lambda r: r.update(legalMask=[False] * 7)):
            value = record(); mutate(value)
            with self.assertRaises(ValueError):
                ppo.pad_records([value], "cpu", {"cardVocabularySize": 1, "actionVocabularySize": 1})

    def test_existing_ordinary_and_singleton_are_bit_identical_and_mixed_batch_masks(self):
        catalog = {"cardVocabularySize": 1, "actionVocabularySize": 1}
        model = ppo.SearchActorCritic(catalog).eval()
        before = copy.deepcopy(model.state_dict())
        for row in (record("ordinary", False), record("selection", False)):
            batch = ppo.pad_records([row], "cpu", catalog)
            with torch.no_grad():
                expected_logits = model.actor(**batch)
                expected_value = model.critic(model.actor.visible_context(batch)).squeeze(-1).tanh()
                distribution, value = model(batch)
            self.assertTrue(torch.equal(distribution.logits, torch.distributions.Categorical(logits=expected_logits).logits))
            self.assertTrue(torch.equal(value, expected_value))
        mixed = ppo.pad_records([record(), record("ordinary", False)], "cpu", catalog)
        with torch.no_grad():
            distribution, values = model(mixed)
        self.assertEqual(distribution.probs.shape, (2, 7))
        self.assertEqual(distribution.probs[0, 1], 0)
        self.assertTrue(torch.equal(distribution.probs[1, 3:], torch.zeros(4)))
        self.assertTrue(all(torch.equal(before[key], value) for key, value in model.state_dict().items()))

if __name__ == "__main__":
    unittest.main()
