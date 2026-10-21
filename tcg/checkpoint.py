"""CPU random-initialization artifacts with pinned schema and integrity checks."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from .decision import integer
from .policy import Policy, PolicyConfig, Vocabulary


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def save_initial_checkpoint(
    directory: str | Path, vocabulary: Vocabulary, config: PolicyConfig = PolicyConfig(),
    seed: int = 860001,
) -> Policy:
    integer(seed, "seed")
    if seed >= 2**63:
        raise ValueError("seed exceeds signed 64-bit range")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        policy = Policy(vocabulary, config).cpu().eval()
    weights = directory / "weights.pt"
    torch.save(policy.state_dict(), weights)
    manifest = {
        "schema": "86-tcg-agent/untrained-policy/1",
        "initialization": "random",
        "optimizer": None,
        "seed": seed,
        "torch_version": str(torch.__version__),
        "config": asdict(config),
        "vocabulary": asdict(vocabulary),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    manifest["manifest_sha256"] = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    (directory / "manifest.json").write_bytes(_json_bytes(manifest))
    return policy


def load_checkpoint(
    directory: str | Path, vocabulary: Vocabulary, config: PolicyConfig = PolicyConfig(),
) -> Policy:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    keys = {"schema", "initialization", "optimizer", "seed", "torch_version", "config",
            "vocabulary", "weights_sha256", "manifest_sha256"}
    if type(manifest) is not dict or set(manifest) != keys:
        raise ValueError("invalid checkpoint manifest keys")
    digest = manifest.pop("manifest_sha256")
    if digest != hashlib.sha256(_json_bytes(manifest)).hexdigest():
        raise ValueError("checkpoint manifest hash mismatch")
    if (manifest["schema"] != "86-tcg-agent/untrained-policy/1"
            or manifest["initialization"] != "random" or manifest["optimizer"] is not None
            or manifest["torch_version"] != str(torch.__version__)):
        raise ValueError("unsupported checkpoint schema, provenance, or runtime")
    integer(manifest["seed"], "seed")
    if manifest["seed"] >= 2**63:
        raise ValueError("seed exceeds signed 64-bit range")
    expected = {"config": asdict(config), "vocabulary": asdict(vocabulary)}
    if _json_bytes({key: manifest[key] for key in expected}) != _json_bytes(expected):
        raise ValueError("checkpoint configuration or vocabulary mismatch")
    weights = directory / "weights.pt"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("checkpoint weight hash mismatch")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        policy = Policy(vocabulary, config).cpu().eval()
    if not isinstance(state, dict):
        raise ValueError("checkpoint state must be a tensor mapping")
    expected_state = policy.state_dict()
    if set(state) != set(expected_state) or any(
        type(tensor) is not torch.Tensor or tensor.shape != expected_state[name].shape
        or tensor.dtype != expected_state[name].dtype or not torch.isfinite(tensor).all()
        for name, tensor in state.items()
    ):
        raise ValueError("checkpoint tensor schema mismatch")
    policy.load_state_dict(state, strict=True)
    return policy
