import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from learning.card_action_torch import (CONTRACT, CONTRACT_SHA256, RUNS, CandidateScorer,
    padded_batch, record, supervised_loss)
from learning.train_card_action import evaluate, load_dataset, train


def row(label=0, count=2, selection=False):
    candidates = [[0.0] * 191 for _ in range(count)]
    for index, candidate in enumerate(candidates):
        candidate[index % 191] = 1.0
    result = {"state": [0.0] * 224, "candidates": candidates, "label": label}
    if selection:
        result.update(kind="selection", labels=[0, 1], label=0)
    return result


class CardActionTests(unittest.TestCase):
    def setUp(self):
        RUNS.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="card-action-test-", dir=RUNS)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(RUNS.resolve()))

    def tearDown(self):
        self.temporary.cleanup()

    def dataset(self):
        games = []
        for index, split in enumerate(("train", "validation")):
            path = self.root / f"game-{index}.jsonl"
            path.write_text("\n".join(json.dumps(row(index % 2, selection=i % 2 == 0)) for i in range(4)) + "\n")
            games.append({"id": f"game-{index}", "seed": 101 + index, "split": split, "recordsPath": path.name,
                "recordsSha256": hashlib.sha256(path.read_bytes()).hexdigest(), "completedNaturally": True,
                "rejected": 0, "commands": 4, "privacyChecks": 8, "realHandExchanges": 1})
        repo = Path(__file__).resolve().parents[1]
        source = "learning/card-action-features.mjs"
        receipt = {"schema": "native-card-action-dataset/1", "complete": True, "fatal": False,
            "featureContract": CONTRACT, "pins": {"testAuthority": "a" * 64},
            "sourceHashes": {source: hashlib.sha256((repo / source).read_bytes()).hexdigest()}, "games": games}
        (self.root / "receipt.json").write_text(json.dumps(receipt))
        return receipt

    def test_padding_and_soft_selection_targets(self):
        states, candidates, mask, targets = padded_batch([row(0, 1), row(selection=True)], "cpu", labelled=True)
        self.assertEqual(tuple(candidates.shape), (2, 2, 191))
        self.assertEqual(targets.tolist(), [[1, 0], [.5, .5]])
        model = CandidateScorer()
        scores = model(states, candidates, mask)
        self.assertTrue(torch.isneginf(scores[0, 1]))
        self.assertTrue(torch.isfinite(supervised_loss(scores, mask, targets)))

    def test_rejects_invalid_features_labels_and_capacity(self):
        bad = row(); bad["state"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            record(bad, labelled=True)
        bad = row(selection=True); bad["labels"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "unique"):
            record(bad, labelled=True)
        with self.assertRaisesRegex(ValueError, "512"):
            record(row(count=513), labelled=True)

    def test_selection_set_metric_distinguishes_top1_from_complete_chosen_set(self):
        class FixedScorer(torch.nn.Module):
            def forward(self, states, candidates, mask):
                return torch.tensor([[3., 2., 1.]]).expand(states.shape[0], -1).masked_fill(~mask, -torch.inf)
        selected = row(count=3, selection=True)
        selected["labels"] = [0, 2]
        measured = evaluate(FixedScorer(), [row(1, count=3), selected], "cpu")
        self.assertEqual(measured["ordinary"]["top1Agreement"], 0)
        self.assertEqual(measured["selection"]["top1InChosenSet"], 1)
        self.assertEqual(measured["selection"]["exactChosenSetAgreement"], 0)

    def test_dataset_admission_rejects_tampering_and_split_overlap(self):
        receipt = self.dataset()
        training, validation, _, _ = load_dataset(self.root)
        self.assertEqual((len(training), len(validation)), (4, 4))
        receipt["games"][1]["seed"] = receipt["games"][0]["seed"]
        (self.root / "receipt.json").write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            load_dataset(self.root)
        self.dataset()
        (self.root / "game-0.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_dataset(self.root)

    def test_checkpoint_training_reload_and_actual_stdio_rank(self):
        self.dataset()
        output = self.root / "model"
        result = train(self.root, output, device="cpu", updates=2, max_seconds=10)
        self.assertEqual(result["completedUpdates"], 2)
        self.assertTrue(result["reloadLogitsExact"])
        self.assertGreater(result["weightL1ChangeFromInitialization"], 0)
        self.assertEqual(result["contractSha256"], CONTRACT_SHA256)
        request = "\n".join(json.dumps(value) for value in [
            {"id": 1, "op": "ping"}, {"id": 2, "op": "infer", **row(selection=True)}, {"id": 3, "op": "close"}]) + "\n"
        process = subprocess.run([sys.executable, str(Path(__file__).with_name("card_action_torch.py")),
            "--checkpoint", str(output / "best-model.pt"), "--device", "cpu"], input=request,
            text=True, capture_output=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)
        replies = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual(replies[0]["contractSha256"], CONTRACT_SHA256)
        self.assertEqual(sorted(replies[1]["ranking"]), [0, 1])
        self.assertEqual(replies[1]["index"], replies[1]["ranking"][0])
        with self.assertRaisesRegex(ValueError, "fresh"):
            train(self.root, output, device="cpu", updates=1)

    def test_warm_start_remains_selected_when_every_update_validates_worse(self):
        self.dataset()
        initial = CandidateScorer()
        baseline = {key: tensor.clone() for key, tensor in initial.state_dict().items()}
        checkpoint = self.root / "initial.pt"
        torch.save({"featureContract": CONTRACT, "contractSha256": CONTRACT_SHA256,
            "state_dict": baseline, "printingAliases": {}}, checkpoint)
        def controlled_validation(model, values, device):
            measured = evaluate(model, values, device)
            unchanged = all(torch.equal(value, baseline[key]) for key, value in model.state_dict().items())
            return {**measured, "loss": 0.1 if unchanged else 1.0}
        with patch("learning.train_card_action.evaluate", side_effect=controlled_validation):
            result = train(self.root, self.root / "warm-start", device="cpu", updates=2, max_seconds=10,
                           initial_checkpoint=str(checkpoint))
        self.assertEqual(result["completedUpdates"], 2)
        self.assertEqual(result["bestUpdate"], 0)
        self.assertTrue(result["retainedInitialCheckpoint"])
        self.assertEqual(result["weightL1ChangeFromInitialization"], 0)
        selected = torch.load(result["checkpoint"], weights_only=True)["state_dict"]
        self.assertTrue(all(torch.equal(value, baseline[key]) for key, value in selected.items()))


if __name__ == "__main__":
    unittest.main()
