"""CPU mathematical and behavioral checks for outcome-based candidate PPO."""

import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from learning import card_action_torch as bc
from learning.masked_ppo import (
    ModelConfig, PPOConfig, MaskedActorCritic, PPOTrainer,
    gae_targets, ppo_loss, explained_variance, pad_records, prepare_rollout,
)


class MaskedPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.shape = ModelConfig(state_dim=2, candidate_dim=1, hidden_sizes=(8, 4),
                                 feature_schema="synthetic/1")
        self.config = PPOConfig(epochs=2, minibatch_size=4, target_kl=1.0)

    def observation(self, count=2):
        return {"schema": "synthetic/1", "kind": "ordinary", "state": [.2, -.1],
                "candidates": [[float(i)] for i in range(count)]}

    def rollout(self, trainer, count=8):
        observations = [self.observation(2 + i % 2) for i in range(count)]
        selected = [trainer.act([observation])[0] for observation in observations]
        return [{**observation, **action, "episodeId": f"game-{i}-seat-0",
                 "stepIndex": 0, "reward": 1.0 if action["actionIndex"] else -1.0,
                 "terminated": True, "truncated": False,
                 "segmentEnd": True, "nextValue": 0.0}
                for i, (observation, action) in enumerate(zip(observations, selected))]

    def test_terminal_gae_matches_hand_calculation_and_ignores_bootstrap(self):
        advantages, returns = gae_targets(
            torch.tensor([0., 1.]), torch.tensor([.2, .3]),
            torch.tensor([.3, 123.]), torch.tensor([False, True]),
            torch.tensor([False, True]), .9, .8)
        torch.testing.assert_close(advantages, torch.tensor([.574, .7]))
        torch.testing.assert_close(returns, torch.tensor([.774, 1.]))
        self.assertFalse(advantages.requires_grad)

    def test_truncated_bootstrap_stops_before_reset_episode(self):
        advantages, returns = gae_targets(
            torch.tensor([0., 0., 1000.]), torch.tensor([.2, .3, 0.]),
            torch.tensor([.3, .5, 0.]), torch.tensor([False, False, True]),
            torch.tensor([False, True, True]), .9, .8)
        torch.testing.assert_close(advantages[:2], torch.tensor([.178, .15]))
        torch.testing.assert_close(returns[:2], torch.tensor([.378, .45]))

    def test_gae_supports_parallel_streams_without_crossing(self):
        advantages, returns = gae_targets(
            torch.tensor([[0., 0.], [1., -1.]]), torch.zeros(2, 2),
            torch.tensor([[0., 0.], [100., 5.]]),
            torch.tensor([[False, False], [True, False]]),
            torch.tensor([[False, False], [True, True]]), 1., 1.)
        torch.testing.assert_close(advantages, torch.tensor([[1., 4.], [1., 4.]]))
        torch.testing.assert_close(returns, advantages)

    def test_ratio_clipping_handles_negative_advantages(self):
        config = PPOConfig(value_coef=0., entropy_coef=0., normalize_advantages=False)
        loss, metrics = ppo_loss(
            torch.log(torch.tensor([1.5, .5])), torch.zeros(2),
            torch.tensor([2., -2.]), torch.zeros(2), torch.zeros(2),
            torch.zeros(2), torch.zeros(2), config)
        self.assertAlmostEqual(float(loss), -.4, places=6)
        self.assertAlmostEqual(float(metrics["clip_fraction"]), 1.)
        expected_kl = ((1.5 - 1 - math.log(1.5)) + (.5 - 1 - math.log(.5))) / 2
        self.assertAlmostEqual(float(metrics["approx_kl"]), expected_kl, places=6)

    def test_value_clipping_and_entropy_objective(self):
        config = PPOConfig(value_coef=1., entropy_coef=.1)
        loss, metrics = ppo_loss(
            torch.zeros(1), torch.zeros(1), torch.zeros(1),
            torch.ones(1), torch.zeros(1), torch.full((1,), 2.),
            torch.full((1,), .7), config)
        self.assertAlmostEqual(float(metrics["value_loss"]), 1.62, places=6)
        self.assertAlmostEqual(float(loss), 1.55, places=6)
        self.assertIsNone(explained_variance(torch.zeros(2), torch.ones(2)))
        self.assertAlmostEqual(explained_variance(torch.tensor([1., 2.]),
                                                 torch.tensor([1., 2.])), 1.)

    def test_padding_permutation_and_state_only_critic(self):
        model = MaskedActorCritic(self.shape)
        states, candidates, mask = pad_records(
            [self.observation(1), self.observation(3)], self.shape)
        distribution, values = model(states, candidates, mask)
        self.assertEqual(distribution.probs[0].tolist(), [1., 0., 0.])
        self.assertTrue(torch.isfinite(distribution.entropy()).all())
        for _ in range(20):
            self.assertEqual(int(distribution.sample()[0]), 0)
        permutation = torch.tensor([2, 0, 1])
        permuted, values_after = model(states, candidates[:, permutation], mask[:, permutation])
        torch.testing.assert_close(permuted.probs, distribution.probs[:, permutation])
        torch.testing.assert_close(values_after, values, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model(states, candidates, torch.zeros_like(mask))

    def test_warm_start_preserves_exact_bc_scores(self):
        torch.manual_seed(33)
        actor = bc.CandidateScorer()
        trainer = PPOTrainer(seed=44)
        saved = {"featureContract": bc.CONTRACT, "contractSha256": bc.CONTRACT_SHA256,
                 "printingAliases": {}, "state_dict": actor.state_dict()}
        trainer.warm_start(saved)
        states = torch.randn(2, 224)
        candidates = torch.randn(2, 3, 191)
        mask = torch.tensor([[True, False, True], [True, True, True]])
        with torch.no_grad():
            logits = actor(states, candidates, mask)
            torch.testing.assert_close(trainer.model.actor(states, candidates, mask),
                                       logits, rtol=0, atol=0)
        invalid = copy.deepcopy(saved)
        invalid["contractSha256"] = "wrong"
        with self.assertRaises(ValueError):
            trainer.warm_start(invalid)

    def test_rollout_admission_rejects_missing_boundaries_gaps_and_selection(self):
        trainer = PPOTrainer(self.shape, self.config, seed=1)
        records = self.rollout(trainer, 2)
        accepted = prepare_rollout(records, self.shape, self.config)
        self.assertEqual(len(accepted.records), 2)
        for mutation in (lambda r: r[0].update(segmentEnd=False, terminated=False),
                         lambda r: r[0].update(kind="selection"),
                         lambda r: r[-1].pop("nextValue"),
                         lambda r: r[0].update(actionIndex=999),
                         lambda r: r[0].update(truncated=True),
                         lambda r: r[0].update(oldLogProb=float("nan")),
                         lambda r: r[0].update(terminated=1)):
            bad = copy.deepcopy(records)
            mutation(bad)
            with self.assertRaises(ValueError):
                prepare_rollout(bad, self.shape, self.config)
        records[0].update(terminated=False, segmentEnd=False, nextValue=records[1]["oldValue"])
        records[1].update(episodeId=records[0]["episodeId"], stepIndex=2)
        with self.assertRaises(ValueError):
            prepare_rollout(records, self.shape, self.config)

    def test_bootstrap_value_continuity_and_explicit_cut(self):
        trainer = PPOTrainer(self.shape, self.config, seed=2)
        records = self.rollout(trainer, 2)
        records[0].update(terminated=False, segmentEnd=False, nextValue=123.)
        records[1].update(episodeId=records[0]["episodeId"], stepIndex=1)
        with self.assertRaises(ValueError):
            prepare_rollout(records, self.shape, self.config)
        records[0].update(nextValue=records[1]["oldValue"])
        prepare_rollout(records, self.shape, self.config)
        records[0].update(segmentEnd=True, nextValue=.5, reward=0.)
        cut = prepare_rollout(records, self.shape, self.config)
        self.assertAlmostEqual(float(cut.returns[0]), self.config.gamma * .5, places=5)

    def test_stale_rollout_is_rejected_before_any_weight_change(self):
        trainer = PPOTrainer(self.shape, self.config, seed=3)
        records = self.rollout(trainer)
        records[-1]["oldLogProb"] += .1
        initial = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        with self.assertRaises(ValueError):
            trainer.update(records)
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, initial[key], rtol=0, atol=0)

    def test_outcome_update_has_finite_gradients_and_changes_weights(self):
        trainer = PPOTrainer(self.shape, self.config, seed=4)
        result = trainer.update(self.rollout(trainer))
        self.assertGreater(result["optimizer_steps"], 0)
        self.assertGreater(result["weight_change_l2"], 0.)
        for key in ("grad_norm", "approx_kl", "clip_fraction"):
            self.assertTrue(math.isfinite(result[key]))
        for parameter in trainer.model.parameters():
            self.assertTrue(torch.isfinite(parameter).all())
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_ragged_updates_split_to_respect_candidate_memory_bound(self):
        config = PPOConfig(epochs=1, minibatch_size=8, max_padded_candidates=3)
        trainer = PPOTrainer(self.shape, config, seed=40)
        records = self.rollout(trainer)
        result = trainer.update(records)
        self.assertEqual(result["records"], 8)
        self.assertEqual(result["optimizer_steps"], 8)
        self.assertGreater(result["weight_change_l2"], 0.)

    def test_full_rollout_kl_stops_later_epochs(self):
        config = PPOConfig(epochs=10, learning_rate=.03, minibatch_size=16,
                           target_kl=1e-12)
        trainer = PPOTrainer(self.shape, config, seed=41)
        result = trainer.update(self.rollout(trainer, 16))
        self.assertEqual(result["stop_reason"], "target_kl")
        self.assertEqual(result["epochs"], 1)
        self.assertGreater(result["approx_kl"], config.target_kl)

    def test_checkpoint_restores_sampling_optimizer_and_continuation(self):
        trainer = PPOTrainer(self.shape, self.config, seed=5)
        trainer.update(self.rollout(trainer))
        root = Path(__file__).resolve().parents[1] / "runs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            path = Path(temporary) / "model.pt"
            trainer.save(str(path), metadata={"sourceCommit": "synthetic-test"})
            restored = PPOTrainer.load(str(path))
            observations = [self.observation(2), self.observation(3)]
            self.assertEqual(trainer.act(observations), restored.act(observations))
            records = self.rollout(trainer)
            self.assertEqual(records, self.rollout(restored))
            left, right = trainer.update(records), restored.update(records)
            self.assertEqual(left["optimizer_steps"], right["optimizer_steps"])
            for key, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                trainer.save(str(path))

    def test_terminal_outcome_learning_increases_rewarded_action_probability(self):
        config = PPOConfig(learning_rate=.01, epochs=3, minibatch_size=32,
                           target_kl=.5, entropy_coef=.001)
        trainer = PPOTrainer(self.shape, config, seed=66)
        observation = self.observation()
        batch = pad_records([observation], self.shape)
        with torch.no_grad():
            before = float(trainer.model(*batch)[0].probs[0, 1])
        for iteration in range(12):
            rows = []
            for index, selected in enumerate(trainer.act([observation] * 64)):
                rows.append({**observation, **selected, "episodeId": f"{iteration}-{index}",
                             "stepIndex": 0, "reward": 1. if selected["actionIndex"] == 1 else -1.,
                             "terminated": True, "truncated": False,
                             "segmentEnd": True, "nextValue": 0.})
            trainer.update(rows)
        with torch.no_grad():
            after = float(trainer.model(*batch)[0].probs[0, 1])
        self.assertGreater(after, before + .2)
        self.assertGreater(after, .9)

    def test_jsonl_process_returns_sampled_action_logprob_value_and_bootstrap(self):
        trainer = PPOTrainer(self.shape, self.config, seed=77)
        observation = self.observation()
        root = Path(__file__).resolve().parents[1] / "runs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            checkpoint = str(Path(temporary) / "inference.pt")
            trainer.save(checkpoint)
            expected = trainer.act([observation], deterministic=True)[0]
            sampled = trainer.act([observation], deterministic=False)[0]
            requests = [
                {"id": 1, "op": "ping"},
                {"id": 2, "op": "infer", **observation, "deterministic": True},
                {"id": 3, "op": "infer", **observation, "deterministic": False},
                {"id": 4, "op": "values", "states": [observation["state"]]},
                {"id": 5, "op": "infer", **observation},
                {"id": 6, "op": "infer", **observation, "kind": "selection", "deterministic": True},
                {"id": 7, "op": "infer_batch", "items": [observation], "deterministic": True},
                {"id": float("nan"), "op": "ping"},
                {"id": 8, "op": "ping"},
                {"id": 9, "op": "close"},
            ]
            process = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("masked_ppo_service.py")),
                 "--checkpoint", checkpoint, "--device", "cpu"],
                input="\n".join(json.dumps(r) for r in requests) + "\n",
                text=True, capture_output=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            replies = [json.loads(line) for line in process.stdout.splitlines()]
            self.assertEqual(len(replies), len(requests))
            self.assertEqual(replies[0]["checkpointSchema"], "native-masked-ppo/1")
            self.assertEqual(replies[0]["checkpoint"], checkpoint)
            for key, value in expected.items():
                self.assertEqual(replies[1][key], value)
            for key, value in sampled.items():
                self.assertEqual(replies[2][key], value)
            self.assertEqual(replies[2]["index"], sampled["actionIndex"])
            self.assertEqual(replies[2]["selectionMode"], "sampled")
            self.assertEqual(replies[3]["values"], trainer.values([observation["state"]]))
            self.assertFalse(replies[4]["ok"])
            self.assertFalse(replies[5]["ok"])
            self.assertEqual(replies[6]["indices"], [expected["actionIndex"]])
            self.assertIsNone(replies[-3]["id"])
            self.assertFalse(replies[-3]["ok"])
            self.assertTrue(replies[-2]["ok"])
            self.assertTrue(replies[-1]["ok"])


if __name__ == "__main__":
    unittest.main()
