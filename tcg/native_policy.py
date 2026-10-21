"""CPU policy boundary for a private, externally configured Mew source tree."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


_mew_root: Path | None = None


def configure_mew(root: str | Path) -> str:
    """Select one private source tree before importing its model or native ABI."""
    global _mew_root
    candidate = Path(root).expanduser().resolve()
    lab = candidate / "agent-lab" if (candidate / "agent-lab").is_dir() else candidate
    if not (lab / "imitation" / "model_v4.py").is_file():
        raise ValueError("Mew root must contain agent-lab/imitation/model_v4.py")
    if _mew_root is not None and lab != _mew_root:
        raise ValueError("one Mew source tree is permitted per process")
    for name in ("fastenc", "cardrows", "imitation"):
        loaded = sys.modules.get(name)
        if loaded is not None:
            source = getattr(loaded, "__file__", None)
            if source is None or not Path(source).resolve().is_relative_to(lab):
                raise ValueError(f"{name} was imported from another source tree")
    if str(lab) not in sys.path:
        sys.path.insert(0, str(lab))
    _mew_root = lab
    return _schema_digest()


def _module(name: str) -> Any:
    if _mew_root is None:
        raise RuntimeError("configure_mew must run before using the native policy boundary")
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(_mew_root):
        raise ValueError(f"{name} resolved outside the configured Mew tree")
    return module


def _schema_digest() -> str:
    schema = _module("cardrows.schema")
    layout = _module("imitation.prepare_v4").layout_signature()
    arrays = [(family, name, schema.ARRAY_WIDTH[name], schema.ARRAY_DTYPE[name])
              for family in schema.FAMILY_ORDER for name in schema.FAMILY_ARRAYS[family]]
    return hashlib.sha256(json.dumps([layout, arrays], sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class _FrozenTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def tensor(self) -> Tensor:
        array = np.frombuffer(self.data, dtype=np.dtype(self.dtype)).reshape(self.shape).copy()
        return torch.from_numpy(array)


@dataclass(frozen=True)
class FrameRecord:
    game_slot: int
    batch_row: int
    select_player: int
    decision_index: int
    prefix_len: int
    option_count: int
    can_submit: bool
    eligible_columns: tuple[int, ...]
    turn: int
    ordered: bool


@dataclass(frozen=True)
class ActorFrame:
    """Immutable tensor bytes copied from a live batch, with no native handle."""

    records: tuple[FrameRecord, ...]
    schema_digest: str
    digest: str
    _arrays: tuple[_FrozenTensor, ...]
    _seq_len: int

    @property
    def real_rows(self) -> int:
        return len(self.records)

    def _payload(self) -> dict[str, Any]:
        return {**{item.name: item.tensor() for item in self._arrays}, "seq_len": self._seq_len}


class FrameCapture:
    """Own the packer's scratch buffers and copy every result before release."""

    def __init__(self) -> None:
        self._schema_digest = _schema_digest()
        self._schema = _module("cardrows.schema")
        self._packer = _module("fastenc.wide").Expander(torch.device("cpu"), pin=False)

    def capture(self, encoded: Any) -> ActorFrame:
        if encoded.batch is None or encoded.batch.n <= 0:
            raise ValueError("frame capture requires a nonempty encoded native batch")
        n = int(encoded.batch.n)
        payload = self._packer.tensors(encoded.batch)
        expected = {"family_count", "seq_len"} | {
            name for family in self._schema.FAMILY_ORDER
            for name in self._schema.FAMILY_ARRAYS[family]
        }
        if set(payload) != expected:
            raise ValueError("native packer arrays differ from the configured schema")
        arrays = []
        for name in sorted(expected - {"seq_len"}):
            array = payload[name][:n].detach().cpu().contiguous().numpy()
            arrays.append(_FrozenTensor(name, array.dtype.str, tuple(array.shape), array.tobytes()))
        f4_family = self._schema.FAMILY_ORDER.index(self._schema.FAMILY_F4)
        eligible_array, eligible_slot = self._schema.SLOT_OF[
            (self._schema.FAMILY_F4, None, "f4_eligible")]
        counts = payload["family_count"][:n, f4_family]
        eligible = payload[eligible_array][:n, :, eligible_slot].bool()
        records = []
        for slot, native in enumerate(encoded.records):
            row = int(native["batchRow"])
            if row < 0:
                continue
            if row >= n or int(native["terminal"]) or not int(native["live"]):
                raise ValueError("invalid native record for encoded batch row")
            physical, submit = int(native["optionCount"]), bool(native["canSubmit"])
            visible = physical + int(submit)
            if int(counts[row]) != visible:
                raise ValueError("native option/Submit count differs from encoded F4 rows")
            columns = tuple(torch.nonzero(eligible[row, :visible], as_tuple=False)[:, 0].tolist())
            if not columns or (submit and physical not in columns):
                raise ValueError("native encoded prompt has no legal action or missing Submit")
            records.append(FrameRecord(slot, row, int(native["selectPlayer"]),
                                       int(native["decisionIndex"]), int(native["prefixLen"]),
                                       physical, submit, columns, int(native["turn"]),
                                       bool(native["ordered"])))
        records.sort(key=lambda record: record.batch_row)
        if tuple(record.batch_row for record in records) != tuple(range(n)):
            raise ValueError("native batch rows must have exactly one game record each")
        seq_len = int(payload["seq_len"])
        digest = hashlib.sha256()
        digest.update(json.dumps([self._schema_digest, seq_len,
                                  [asdict(record) for record in records]], sort_keys=True).encode())
        for array in arrays:
            digest.update(json.dumps([array.name, array.dtype, array.shape]).encode())
            digest.update(array.data)
        return ActorFrame(tuple(records), self._schema_digest, digest.hexdigest(), tuple(arrays), seq_len)


@dataclass(frozen=True)
class SmallNativeConfig:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    ffn_mult: int = 2
    d_ref: int = 32
    card_id_dim: int = 32
    card_atom_dim: int = 32
    attack_id_dim: int = 32
    attack_atom_dim: int = 32
    dropout: float = 0.0


def policy_parameters(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    return tuple((name, parameter) for name, parameter in model.named_parameters()
                 if not name.startswith("value_head."))


def freeze_policy_only(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    if not any(name.startswith("value_head.") for name, _ in model.named_parameters()):
        raise ValueError("native student requires the separate ModelV4 value head")
    allowed = policy_parameters(model)
    names = {name for name, _ in allowed}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)
        parameter.grad = None
    model.eval()
    return allowed


def build_student(config: SmallNativeConfig = SmallNativeConfig(), *, seed: int) -> nn.Module:
    if type(seed) is not int or seed < 0:
        raise ValueError("student seed must be a nonnegative integer")
    if config.dropout != 0.0:
        raise ValueError("stage-1 student requires deterministic dropout=0")
    source = _module("imitation.model_v4")
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = source.ModelV4(source.ModelV4Config(**asdict(config)))
    freeze_policy_only(model)
    model._native_origin = {"kind": "fresh_student", "seed": seed}
    return model


def _checked_checkpoint(path: str | Path, expected_sha256: str) -> Path:
    checkpoint = Path(path).resolve()
    actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if len(expected_sha256) != 64 or actual != expected_sha256:
        raise ValueError("checkpoint SHA-256 differs from the pinned artifact")
    return checkpoint


def load_teacher(checkpoint: str | Path, *, expected_sha256: str) -> nn.Module:
    path = _checked_checkpoint(checkpoint, expected_sha256)
    with torch.device("cpu"):
        model = _module("imitation.model_v4").load_checkpoint(path, map_location="cpu")
    model.eval().requires_grad_(False)
    model._native_origin = {"kind": "frozen_teacher", "checkpoint_sha256": expected_sha256}
    return model


def load_student(checkpoint: str | Path, *, expected_sha256: str) -> nn.Module:
    path = _checked_checkpoint(checkpoint, expected_sha256)
    with torch.device("cpu"):
        model = _module("imitation.model_v4").load_checkpoint(path, map_location="cpu")
    if model.config.dropout != 0.0:
        raise ValueError("stage-1 student checkpoint requires deterministic dropout=0")
    freeze_policy_only(model)
    model._native_origin = {"kind": "student_checkpoint", "checkpoint_sha256": expected_sha256}
    return model


@dataclass(frozen=True)
class PolicyScores:
    logits: Tensor
    legal_mask: Tensor
    frame_digest: str


def score_policy(model: nn.Module, frame: ActorFrame, *, grad: bool = False) -> PolicyScores:
    """Read only copied actor tensors. This function never calls a game API."""
    if frame.schema_digest != _schema_digest():
        raise ValueError("frame schema differs from configured Mew source")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("native policy boundary is CPU only")
    if model.training:
        raise ValueError("native policies must use deterministic eval mode, including during distillation")
    if grad and getattr(model, "_native_origin", {}).get("kind") == "frozen_teacher":
        raise ValueError("teacher scoring cannot request gradients")
    with torch.enable_grad() if grad else torch.no_grad():
        logits, _ = model(frame._payload(), validate=True)
    if logits.ndim != 2 or logits.shape[0] != frame.real_rows:
        raise ValueError("native model output rows differ from captured frame")
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for row, record in enumerate(frame.records):
        if record.eligible_columns[-1] >= logits.shape[1]:
            raise ValueError("native model omitted a legal option column")
        mask[row, list(record.eligible_columns)] = True
    if not torch.equal(torch.isfinite(logits), mask) or not torch.isneginf(logits[~mask]).all():
        raise ValueError("native model mask or legal logits differ from captured frame")
    return PolicyScores(logits, mask, frame.digest)


def state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_identity(model: nn.Module) -> dict[str, Any]:
    named = dict(model.named_parameters())
    total = sum(parameter.numel() for parameter in named.values())
    vocabulary = sum(named[name].numel() for name in ("card.weight", "attack.weight"))
    return {"origin": dict(getattr(model, "_native_origin", {})),
            "schema_digest": _schema_digest(), "state_digest": state_digest(model),
            "config": asdict(model.config),
            "parameters": {"total": total, "vocabulary": vocabulary,
                           "model": total - vocabulary,
                           "trainable": sum(p.numel() for p in named.values() if p.requires_grad)},
            "vocab_padding": dict(model.vocab_padding),
            "reader_migration": dict(model.reader_migration),
            "trainable_names": [name for name, p in model.named_parameters() if p.requires_grad]}


def save_policy(model: nn.Module, path: str | Path) -> str:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("native policy checkpoint destination already exists")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("native policy checkpoints must be CPU artifacts")
    _module("imitation.model_v4").save_checkpoint(model, destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()
