"""Explicit model storage roots, without external disks or real training data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from learning.card_action_torch import (CONTRACT, CONTRACT_SHA256, RUNS, CandidateScorer,
    load_checkpoint, model_root_for, private_checkpoint)
from learning.train_card_action import train


class ModelStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="model-storage-")
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "models"
        self.root.mkdir()
        self.checkpoint = self.root / "example.pt"
        torch.save({"featureContract": CONTRACT, "contractSha256": CONTRACT_SHA256,
                    "state_dict": CandidateScorer().state_dict()}, self.checkpoint)
        self.environment = patch.dict(os.environ)
        self.environment.start()
        os.environ.pop("PTCG_AGENT_MODEL_ROOT", None)

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def test_external_checkpoint_requires_explicit_root(self):
        with self.assertRaises(ValueError):
            load_checkpoint(str(self.checkpoint), "cpu")
        os.environ["PTCG_AGENT_MODEL_ROOT"] = str(self.root)
        model, _ = load_checkpoint(str(self.checkpoint), "cpu")
        self.assertFalse(model.training)
        self.assertEqual(private_checkpoint(str(self.checkpoint)), self.checkpoint)

    def test_invalid_roots_paths_and_bare_directories_are_rejected(self):
        for root in ("relative/models", "", self.root.anchor, str(RUNS.parent)):
            with self.subTest(root=root):
                os.environ["PTCG_AGENT_MODEL_ROOT"] = root
                with self.assertRaises(ValueError):
                    model_root_for(self.checkpoint)
        os.environ["PTCG_AGENT_MODEL_ROOT"] = str(self.root)
        for path in (self.root, RUNS, RUNS.parent, self.root / ".." / "escape.pt"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                model_root_for(path)
        with self.assertRaises(ValueError):
            private_checkpoint("relative.pt")
        with self.assertRaises(ValueError):
            private_checkpoint(str(self.root / "weights.bin"))

    def test_external_output_config_is_portable_and_fresh(self):
        os.environ["PTCG_AGENT_MODEL_ROOT"] = str(self.root)
        row = {"state": [0.] * 224, "candidates": [[0.] * 191, [1.] * 191], "label": 0}
        output = self.root / "new-model"
        with patch("learning.train_card_action.load_dataset", return_value=(
                [row], [row], {}, {"datasetSha256": "a" * 64})):
            train("synthetic", output, device="cpu", updates=1, max_seconds=10)
            config = json.loads((output / "model-config.json").read_text())
            self.assertEqual(config["modelRoot"], str(self.root))
            with self.assertRaisesRegex(ValueError, "fresh"):
                train("synthetic", output, device="cpu", updates=1, max_seconds=10)
        os.environ.pop("PTCG_AGENT_MODEL_ROOT")
        with self.assertRaises(ValueError):
            load_checkpoint(config["checkpoint"], "cpu")
        os.environ["PTCG_AGENT_MODEL_ROOT"] = config["modelRoot"]
        load_checkpoint(config["checkpoint"], "cpu")

    def test_node_client_passes_explicit_or_inherited_root_without_global_mutation(self):
        script = """
          import assert from 'node:assert/strict';
          import { CardActionModelClient } from './integrations/local-worker/card-action-model.mjs';
          const [executable, checkpoint, root, mode] = process.argv.slice(1);
          const before = process.env.PTCG_AGENT_MODEL_ROOT;
          for (const modelRoot of ['relative/models', new URL('file:///').pathname]) {
            assert.throws(() => new CardActionModelClient({ executable, checkpoint, modelRoot }));
          }
          const config = { executable, checkpoint, ...(mode === 'explicit' ? { modelRoot: root } : {}) };
          const client = new CardActionModelClient(config);
          try { await client.verify(); assert.equal(process.env.PTCG_AGENT_MODEL_ROOT, before); }
          finally { await client.shutdown(); }
        """
        for mode in ("explicit", "inherited"):
            environment = dict(os.environ)
            environment.pop("PTCG_AGENT_MODEL_ROOT", None)
            if mode == "inherited":
                environment["PTCG_AGENT_MODEL_ROOT"] = str(self.root)
            with self.subTest(mode=mode):
                result = subprocess.run(["node", "--input-type=module", "-e", script,
                    sys.executable, str(self.checkpoint), str(self.root), mode],
                    cwd=RUNS.parent, env=environment, capture_output=True, text=True, timeout=25)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
