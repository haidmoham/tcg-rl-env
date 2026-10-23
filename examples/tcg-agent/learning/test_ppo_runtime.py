"""Small CPU fixtures for the private PPO command and collection boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from learning import card_action_torch as bc
from learning import masked_ppo as ppo
from learning import ppo_runtime as runtime


def feature():
    return {"schema": bc.SCHEMA, "kind": "ordinary", "state": [0.] * 224,
            "candidates": [[0.] * 191, [0.25] * 191]}


class PPORuntimeTests(unittest.TestCase):
    def setUp(self):
        bc.RUNS.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="ppo-runtime-test-", dir=bc.RUNS)
        self.root = Path(self.temporary.name).resolve()
        self.bc_path = self.root / "bc.pt"
        torch.save({"featureContract": bc.CONTRACT, "contractSha256": bc.CONTRACT_SHA256,
                    "state_dict": bc.CandidateScorer().state_dict(), "printingAliases": {"a": "b"}}, self.bc_path)
        self.parent = self.root / "initial.pt"
        runtime.initialize(str(self.bc_path), str(self.parent), device="cpu", seed=13,
                           config={"epochs": 1, "minibatch_size": 4})

    def tearDown(self):
        self.temporary.cleanup()

    def collection(self):
        trainer = ppo.PPOTrainer.load(str(self.parent), device="cpu")
        rows, episodes = [], []
        for index, reward in enumerate((1., -1.)):
            row = feature()
            row.update(trainer.act([row], deterministic=False)[0])
            row.update(reward=reward, terminated=True, truncated=False, nextValue=0.,
                       segmentEnd=True, episodeId=f"toy-{index}", stepIndex=0)
            rows.append(row)
            episodes.append({"episodeId": row["episodeId"], "status": "terminated",
                "terminated": True, "truncated": False, "result": "win" if reward == 1 else "loss",
                "recordsCount": 1})
        directory = self.root / "rollout"
        directory.mkdir()
        path = directory / "records.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        receipt = {"schema": "native-ppo-rollout/1", "complete": True, "fatal": False,
            "featureContract": bc.CONTRACT, "checkpointsUnchanged": True,
            "parentCheckpointSha256": runtime.sha256(self.parent),
            "selectionCheckpointSha256": runtime.sha256(self.bc_path),
            "sourceHashes": runtime.source_hashes(), "pins": {"syntheticAuthority": "a" * 64},
            "recordsPath": path.name, "recordsSha256": runtime.sha256(path), "recordsCount": len(rows),
            "counts": {"episodes": 2, "terminated": 2, "truncated": 0, "excluded": 0,
                       "records": 2, "fatalErrors": 0, "errors": 0, "rejected": 0}, "episodes": episodes}
        (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        return directory, receipt

    def test_warm_start_and_real_stdio_actions_values(self):
        trainer = ppo.PPOTrainer.load(str(self.parent), device="cpu")
        saved = torch.load(self.bc_path, weights_only=True)
        self.assertTrue(all(torch.equal(value, trainer.model.actor.state_dict()[key])
                            for key, value in saved["state_dict"].items()))
        requests = [{"id": 1, "op": "ping"},
                    {"id": 2, "op": "act", "records": [feature()], "deterministic": False},
                    {"id": 3, "op": "values", "states": [feature()["state"]]},
                    {"id": 4, "op": "act", "records": [feature()]},
                    {"id": 5, "op": "seed", "seed": 42},
                    {"id": 6, "op": "ping"},
                    {"id": 7, "op": "close"}]
        process = subprocess.run([sys.executable, str(Path(runtime.__file__)), "serve", "--checkpoint",
            str(self.parent), "--device", "cpu", "--seed", "17"],
            input="".join(json.dumps(row) + "\n" for row in requests), capture_output=True, text=True, timeout=25)
        self.assertEqual(process.returncode, 0, process.stderr)
        replies = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual([reply["id"] for reply in replies], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual([reply["ok"] for reply in replies], [True, True, True, False, True, True, True])
        self.assertEqual(replies[0]["schema"], "native-masked-ppo/1")
        self.assertEqual(replies[0]["checkpointSha256"], runtime.sha256(self.parent))
        self.assertEqual(replies[0]["frozenSelectionSha256"], runtime.sha256(self.bc_path))
        self.assertIn(replies[1]["actions"][0]["actionIndex"], (0, 1))
        self.assertEqual(replies[1]["actions"][0]["oldValue"], replies[2]["values"][0])
        self.assertEqual(replies[4]["samplingSeed"], 42)
        self.assertEqual(replies[5]["samplingSeed"], 42)

    def test_single_update_preserves_parent_and_writes_verified_receipt(self):
        directory, data_receipt = self.collection()
        output = self.root / "updated.pt"
        before = runtime.sha256(self.parent)
        result = runtime.update_once(str(self.parent), directory, str(output), device="cpu")
        receipt = runtime.read_json(result["receipt"])
        self.assertEqual(receipt["parentCheckpointSha256"], before)
        self.assertEqual(receipt["dataSourceSha256"], data_receipt["recordsSha256"])
        self.assertEqual(receipt["metrics"]["update_count"], 1)
        self.assertGreater(receipt["metrics"]["weight_change_l2"], 0)
        self.assertTrue(receipt["reloadLogitsExact"] and receipt["reloadValuesExact"])
        self.assertEqual(runtime.sha256(self.parent), before)
        with self.assertRaisesRegex(ValueError, "fresh"):
            runtime.update_once(str(self.parent), directory, str(output), device="cpu")

    def test_seed_reset_repeats_sampling_without_changing_weights(self):
        trainer = ppo.PPOTrainer.load(str(self.parent), device="cpu")
        before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
        identity = {"samplingSeed": 13}
        rows = [feature() for _ in range(16)]
        for seed in (True, -1, 0x100000000, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                runtime.respond(trainer, {"id": 1, "op": "seed", "seed": seed}, identity)
        request = {"id": 2, "op": "seed", "seed": 0xffffffff}
        self.assertEqual(runtime.respond(trainer, request, identity), {"samplingSeed": 0xffffffff})
        first = runtime.respond(trainer, {"id": 3, "op": "act", "records": rows, "deterministic": False}, identity)
        runtime.respond(trainer, {"id": 4, "op": "act", "records": rows, "deterministic": False}, identity)
        runtime.respond(trainer, request, identity)
        repeated = runtime.respond(trainer, {"id": 5, "op": "act", "records": rows, "deterministic": False}, identity)
        self.assertEqual(first, repeated)
        self.assertEqual(runtime.respond(trainer, {"id": 6, "op": "ping"}, identity)["samplingSeed"], 0xffffffff)
        self.assertTrue(all(torch.equal(value, trainer.model.state_dict()[key]) for key, value in before.items()))

    def test_invalid_sha_result_and_source_freeze_reject_before_save(self):
        directory, receipt = self.collection()
        output = self.root / "never.pt"
        for change in ({"recordsSha256": "0" * 64}, {"parentCheckpointSha256": "0" * 64},
                       {"sourceHashes": {}}, {"counts": {**receipt["counts"], "rejected": 1}}):
            (directory / "receipt.json").write_text(json.dumps({**receipt, **change}), encoding="utf-8")
            with self.assertRaises(ValueError):
                runtime.update_once(str(self.parent), directory, str(output), device="cpu")
            self.assertFalse(output.exists())
        receipt["episodes"][0]["result"] = "loss"
        (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "terminal reward"):
            runtime.update_once(str(self.parent), directory, str(output), device="cpu")
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "allocation limits"):
            runtime.initialize(str(self.bc_path), str(output), device="cpu",
                               config={"max_padded_candidates": 1000000})
        self.assertFalse(output.exists())

    def test_explicit_truncation_bootstraps_without_inventing_a_result(self):
        directory, receipt = self.collection()
        records_path = directory / "records.jsonl"
        rows = [json.loads(line) for line in records_path.read_text().splitlines()]
        rows[-1].update(terminated=False, truncated=True, reward=0., nextValue=rows[-1]["oldValue"])
        receipt["episodes"][-1].update(status="truncated", terminated=False, truncated=True,
            result=None, auditSummaryAvailable=False, privacyChecks=None, opponentHandExchanges=None)
        receipt["counts"].update(terminated=1, truncated=1)
        records_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        receipt["recordsSha256"] = runtime.sha256(records_path)
        (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        result = runtime.update_once(str(self.parent), directory, str(self.root / "truncated.pt"), device="cpu")
        self.assertEqual(result["updateCount"], 1)

    def test_external_root_runtime_save_reload_and_boundary(self):
        with tempfile.TemporaryDirectory(prefix="ppo-external-test-", dir=runtime.REPO / ".cache") as directory:
            root = Path(directory).resolve()
            output = root / "fresh" / "initial.pt"
            with patch.dict(os.environ, {"PTCG_AGENT_MODEL_ROOT": str(root)}):
                result = runtime.initialize(str(self.bc_path), str(output), device="cpu", seed=13,
                                            config={"epochs": 1})
                self.assertEqual(result["checkpoint"], str(output))
                loaded = ppo.PPOTrainer.load(str(output), device="cpu")
                self.assertEqual(loaded.update_count, 0)
                with self.assertRaises(ValueError):
                    runtime.initialize(str(self.bc_path), str(root), device="cpu")
                with self.assertRaises(ValueError):
                    loaded.save(str(root / ".." / "escaped.pt"))
            with patch.dict(os.environ):
                os.environ.pop("PTCG_AGENT_MODEL_ROOT", None)
                with self.assertRaises(ValueError):
                    ppo.PPOTrainer.load(str(output), device="cpu")


if __name__ == "__main__":
    unittest.main()
