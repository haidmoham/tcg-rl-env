import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from learning.native_torch import RUNS, SCHEMA
from learning.train_second import fit, load_collection, resident_dataset, validate_plan


def write(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")


def fixture(root):
    pins = {key: "a" * 64 for key in ("workerSha256", "executableSha256", "manifestSha256", "policySha256")}
    sources = {"collector": "b" * 64, "features": "c" * 64,
               "learning/model-two-plan.json": validate_plan()}
    tasks = [{"id": f"seed-{seed}", "seed": seed,
              "seatOrder": [0, 1] if (seed - 91001) % 2 == 0 else [1, 0],
              "split": "train" if seed < 91019 else "validation"} for seed in range(91001, 91025)]
    completed, unattempted = [], []
    for task in tasks:
        if task["seed"] not in (*range(91001, 91009), 91019, 91020):
            unattempted.append({**task, "status": "unattempted", "reason": "global-wall-budget"})
            continue
        values = []
        for index in range(16 if task["split"] == "train" else 8):
            state = [0.0] * 32
            state[1] = index / 100
            choices = [[0.0] * 27, [1.0] + [0.0] * 26]
            values.append({"state": state, "candidates": choices, "label": 1})
        record_path = root / f"{task['id']}-records.jsonl"
        receipt_path = root / f"{task['id']}-receipt.json"
        lines = [json.dumps(value, separators=(",", ":")) for value in values]
        record_path.write_text("\n".join(lines) + "\n")
        canonical = hashlib.sha256(("[" + ",".join(lines) + "]").encode()).hexdigest()
        receipt = {**task, "completedNaturally": True, "rejected": 0, "realHandExchanges": 5,
                   "privacyChecks": 10, "pins": pins, "sourceHashes": sources,
                   "featureContract": {"schema": SCHEMA, "stateDimensions": 32, "candidateDimensions": 27},
                   "ordinaryDecisions": len(values), "commands": 50,
                   "terminalStateHash": "d" * 64, "trajectorySha256": "e" * 64,
                   "recordsSha256": canonical}
        write(receipt_path, receipt)
        completed.append({**task, "status": "completed", "recordsPath": record_path.name,
            "receiptPath": receipt_path.name, "receiptSha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "recordsFileSha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
            **{key: receipt[key] for key in ("ordinaryDecisions", "commands", "terminalStateHash", "trajectorySha256", "recordsSha256")}})
    manifest = {"schema": "native-supervised-collection/1", "complete": True, "fatal": False,
                "fatalFailures": [], "cancelled": [], "pins": pins, "sourceHashes": sources,
                "tasks": tasks, "completed": completed, "capacityFailures": [], "unattempted": unattempted}
    write(root / "receipt.json", manifest)
    return manifest


class SecondModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        RUNS.mkdir(exist_ok=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=RUNS)
        self.root = Path(self.temp.name)
        self.manifest = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        write(self.root / "receipt.json", self.manifest)

    def test_admits_only_new_whole_game_split(self):
        dataset = load_collection(self.root)
        self.assertEqual((len(dataset.training), len(dataset.validation)), (128, 16))
        self.assertEqual(dataset.provenance["trainingGames"], 8)
        self.assertEqual(dataset.provenance["validationGames"], 2)
        self.assertEqual(dataset.provenance["reservedFinalEvaluationSeeds"], [92001, 92002])

    def test_fatal_and_duplicate_outcomes_block_training(self):
        self.manifest["fatal"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "fatal"):
            load_collection(self.root)
        self.manifest["fatal"] = False
        self.manifest["completed"].append(self.manifest["completed"][0])
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_collection(self.root)

    def test_incomplete_data_cannot_meet_game_minimum(self):
        removed = self.manifest["completed"].pop(0)
        self.manifest["capacityFailures"].append({key: removed[key] for key in ("id", "seed", "seatOrder", "split")}
            | {"status": "capacity-timeout", "reason": "episode-wall-budget", "code": "NATIVE_EPISODE_TIME_BUDGET"})
        self.save()
        with self.assertRaisesRegex(ValueError, "at least 8"):
            load_collection(self.root)

    def test_record_tampering_is_rejected(self):
        item = self.manifest["completed"][0]
        path = self.root / item["recordsPath"]
        path.write_text(path.read_text().replace('"label":1', '"label":99'))
        with self.assertRaisesRegex(ValueError, "file hash"):
            load_collection(self.root)

    def test_source_mismatch_is_rejected_even_with_receipt_rehash(self):
        item = self.manifest["completed"][0]
        path = self.root / item["receiptPath"]
        receipt = json.loads(path.read_text())
        receipt["sourceHashes"] = {"wrong": "f" * 64}
        write(path, receipt)
        item["receiptSha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save()
        with self.assertRaisesRegex(ValueError, "matching native"):
            load_collection(self.root)

    def test_resident_padding_preserves_candidate_order(self):
        dataset = load_collection(self.root)
        values = dataset.training[:2]
        values[0]["candidates"] = values[0]["candidates"][:1]
        values[0]["label"] = 0
        _, candidates, mask, labels = resident_dataset(values, "cpu")
        self.assertEqual(mask.tolist(), [[True, False], [True, True]])
        self.assertEqual(candidates[1, 1, 0].item(), 1.0)
        self.assertEqual(labels.tolist(), [0, 1])

    def test_one_cpu_update_selects_trained_checkpoint_and_reloads(self):
        dataset = load_collection(self.root)
        output = self.root / "new-model2"
        result = fit(dataset, output, device="cpu", updates=1)
        self.assertEqual((result["parameters"], result["bestUpdate"]), (1953, 1))
        self.assertEqual(result["completedUpdates"], 1)
        self.assertTrue(result["reloadLogitsExact"])
        self.assertGreater(result["weightL1ChangeFromInitialization"], 0)
        self.assertEqual(result["checkpointSha256"], hashlib.sha256((output / "best-model.pt").read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            fit(dataset, output, device="cpu", updates=1)

    def test_validation_patience_selects_trained_candidate_without_initialization(self):
        dataset = load_collection(self.root)
        metrics = {"loss": 1.0, "teacherAgreement": 0.5, "records": 16}
        with patch("learning.train_second.evaluate", return_value=metrics):
            result = fit(dataset, self.root / "patience-model", device="cpu", updates=100)
        self.assertEqual(result["stopReason"], "validation-patience")
        self.assertEqual(result["completedUpdates"], 40)
        self.assertEqual(result["bestUpdate"], 10)
        self.assertGreater(result["weightL1ChangeFromInitialization"], 0)

    def test_hard_training_bounds_reject_before_output_creation(self):
        dataset = load_collection(self.root)
        for arguments in ({"updates": 1001}, {"max_seconds": 121}, {"seed": 1}):
            with self.assertRaises(ValueError):
                fit(dataset, self.root / "forbidden", device="cpu", **arguments)
        self.assertFalse((self.root / "forbidden").exists())


if __name__ == "__main__":
    unittest.main()
