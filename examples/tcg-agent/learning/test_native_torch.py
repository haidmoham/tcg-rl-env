import math
from pathlib import Path
import tempfile
import unittest

import torch

from learning.native_torch import CandidateScorer, Learner, RUNS, padded_batch, record


def example(count=2, label=0):
    choices = [[0.0] * 27 for _ in range(count)]
    for index, choice in enumerate(choices):
        choice[index % 27] = 1.0
    return {"state": [0.0] * 32, "candidates": choices, "label": label}


class CandidateLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.learner = Learner("cpu", 1729)

    def test_padding_can_never_win(self):
        batch = padded_batch([example(1), example(3)], "cpu")
        scorer = CandidateScorer()
        scores = scorer(*batch[:3])
        self.assertEqual(scores[0, 1:].tolist(), [-math.inf, -math.inf])
        self.assertEqual(scores.argmax(dim=1)[0].item(), 0)

    def test_candidates_keep_input_order(self):
        value = example(3)
        original = self.learner.infer([value])[0]
        value["candidates"].reverse()
        reversed_index = self.learner.infer([value])[0]
        self.assertEqual(original, 2 - reversed_index)

    def test_bad_labels_and_features_are_rejected(self):
        for label in (-1, 2, True, 0.5):
            with self.assertRaises(ValueError):
                record(example(label=label), require_label=True)
        value = example()
        value["state"][0] = float("nan")
        with self.assertRaises(ValueError):
            record(value)
        with self.assertRaises(ValueError):
            record(example(count=0))

    def test_bounded_training_changes_weights_and_reloads(self):
        RUNS.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RUNS) as directory:
            result = self.learner.train({"records": [example(2), example(3, 1)],
                                         "updates": 1,
                                         "checkpoint": str(Path(directory) / "synthetic.pt")})
            self.assertEqual(result["updates"], 1)
            self.assertGreater(result["weightL1Change"], 0)
            self.assertTrue(result["reloadLogitsExact"])
            self.assertTrue(all(math.isfinite(value) for value in result["gradientNorms"]))
            with self.assertRaises(ValueError):
                self.learner.train({})


if __name__ == "__main__":
    unittest.main()
