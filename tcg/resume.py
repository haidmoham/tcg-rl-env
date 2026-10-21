"""Integrity-checked zero-update boundaries, without native or optimizer resume."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

import torch

from .checkpoint import load_checkpoint
from .decision import integer
from .policy import Policy, PolicyConfig, Vocabulary
from .population import Population
from .rollout import content_hash


_FILES = ("initial/manifest.json", "initial/weights.pt", "population.json", "random.json")


def _encode(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ResumeContract:
    protocol_sha256: str
    engine_sha256: str
    decks_sha256: str
    projection_sha256: str
    config_sha256: str
    weights_sha256: str

    def __post_init__(self):
        for value in asdict(self).values():
            content_hash(value)


@dataclass(frozen=True)
class BoundaryCounters:
    games_started: int = 0
    games_finished: int = 0
    learner_micro_decisions: int = 0
    optimizer_updates: int = 0

    def __post_init__(self):
        for name, value in asdict(self).items():
            integer(value, name)
        if self.games_started != self.games_finished:
            raise ValueError("boundary requires all started games to have finished")
        if self.optimizer_updates != 0:
            raise ValueError("only zero optimizer updates are supported")


@dataclass(frozen=True)
class RestoredBoundary:
    policy: Policy
    population: Population
    deck_rng: random.Random
    policy_rng: torch.Generator
    counters: BoundaryCounters
    contract: ResumeContract
    training_resume_verified: bool = False
    native_chance_exact_resume: bool = False


def _python_rng(state):
    if (type(state) not in (list, tuple) or len(state) != 3 or type(state[0]) is not int
            or state[0] != 3 or type(state[1]) not in (list, tuple) or len(state[1]) != 625
            or any(type(value) is not int or not 0 <= value < 2**32 for value in state[1][:-1])
            or not any(state[1][:-1]) or type(state[1][-1]) is not int
            or not 0 <= state[1][-1] <= 624
            or state[2] is not None and (type(state[2]) is not float or not math.isfinite(state[2]))):
        raise ValueError("invalid deck sampling RNG state")
    restored = random.Random(0)
    restored.setstate((3, tuple(state[1]), state[2]))
    return restored


def _torch_rng(state):
    expected_size = torch.Generator(device="cpu").get_state().numel()
    if (type(state) is not list or len(state) != expected_size
            or any(type(value) is not int or not 0 <= value <= 255 for value in state)):
        raise ValueError("invalid CPU policy sampling RNG state")
    restored = torch.Generator(device="cpu")
    try:
        restored.set_state(torch.tensor(state, dtype=torch.uint8))
    except RuntimeError as error:
        raise ValueError("invalid CPU policy sampling RNG state") from error
    return restored


def _population(data, contract):
    population = Population.from_bytes(data)
    if (population.accepted_updates != 0 or len(population.archive) != 1
            or population.archive[0].weights_sha256 != contract.weights_sha256):
        raise ValueError("population must contain only the matching random initialization")
    return population


def _read_file(directory: Path, name: str) -> bytes:
    if name not in (*_FILES, "boundary.json"):
        raise ValueError("unsupported package file path")
    path = directory / name
    if path.is_symlink() or (name.startswith("initial/") and (directory / "initial").is_symlink()):
        raise ValueError("package files must not be symbolic links")
    if not path.resolve(strict=True).is_relative_to(directory):
        raise ValueError("package file escapes its directory")
    return path.read_bytes()


def save_boundary(
    directory: str | Path, *, initial_checkpoint: str | Path, vocabulary: Vocabulary,
    contract: ResumeContract, population: Population, deck_rng: random.Random,
    policy_rng: torch.Generator, counters: BoundaryCounters,
    in_flight_games: int, partial_games: int, config: PolicyConfig = PolicyConfig(),
) -> None:
    """Write exclusive files and publish the integrity manifest last.

    The caller must stop workers and finish every game before taking this boundary.
    Failed writes retain an incomplete package that restore rejects.
    """
    for name, value in (("in-flight games", in_flight_games), ("partial games", partial_games)):
        integer(value, name)
        if value:
            raise ValueError("native games must finish before saving a boundary")
    if type(contract) is not ResumeContract or type(counters) is not BoundaryCounters:
        raise ValueError("typed contract and counters required")
    if type(population) is not Population or type(deck_rng) is not random.Random:
        raise ValueError("typed population and deck RNG required")
    if type(policy_rng) is not torch.Generator or policy_rng.device != torch.device("cpu"):
        raise ValueError("CPU policy sampling generator required")
    source = Path(initial_checkpoint).resolve(strict=True)
    checkpoint_files = {}
    for name in ("manifest.json", "weights.pt"):
        path = source / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(source):
            raise ValueError("checkpoint file escapes its directory")
        checkpoint_files["initial/" + name] = path.read_bytes()
    if _digest(checkpoint_files["initial/weights.pt"]) != contract.weights_sha256:
        raise ValueError("initial weights do not match the resume contract")
    load_checkpoint(source, vocabulary, config)
    population_data = population.to_bytes()
    _population(population_data, contract)
    deck_state = deck_rng.getstate()
    _python_rng(deck_state)
    policy_state = policy_rng.get_state().tolist()
    _torch_rng(policy_state)
    files = {**checkpoint_files, "population.json": population_data,
             "random.json": _encode({"deck_rng": deck_state, "policy_rng": policy_state})}
    payload = {
        "schema": "86-tcg-agent/zero-update-boundary/1",
        "contract": asdict(contract), "counters": asdict(counters),
        "files": {name: _digest(data) for name, data in files.items()},
        "python_version": list(sys.version_info[:3]), "torch_version": str(torch.__version__),
        "optimizer": None, "in_flight_games": 0, "partial_games": 0,
        "training_resume_verified": False, "native_chance_exact_resume": False,
    }
    envelope = _encode({"payload": payload, "sha256": _digest(_encode(payload))})
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "initial").mkdir()
    for name, data in files.items():
        with (directory / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    load_checkpoint(directory / "initial", vocabulary, config)
    with (directory / "boundary.json").open("xb") as stream:
        stream.write(envelope)
        stream.flush()
        os.fsync(stream.fileno())


def restore_boundary(
    directory: str | Path, *, expected_contract: ResumeContract,
    vocabulary: Vocabulary, config: PolicyConfig = PolicyConfig(),
) -> RestoredBoundary:
    """Validate the entire package before loading policy weights or returning RNGs."""
    if type(expected_contract) is not ResumeContract:
        raise ValueError("typed expected contract required")
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("package directory must not be a symbolic link")
    directory = directory.resolve(strict=True)
    try:
        if ({path.name for path in directory.iterdir()} != {"initial", "population.json", "random.json", "boundary.json"}
                or {path.name for path in (directory / "initial").iterdir()} != {"manifest.json", "weights.pt"}):
            raise ValueError("incomplete or unexpected package files")
        envelope = json.loads(_read_file(directory, "boundary.json"))
        if type(envelope) is not dict or set(envelope) != {"payload", "sha256"}:
            raise ValueError("invalid boundary envelope")
        payload = envelope["payload"]
        fields = {"schema", "contract", "counters", "files", "python_version", "torch_version",
                  "optimizer", "in_flight_games", "partial_games", "training_resume_verified",
                  "native_chance_exact_resume"}
        if type(payload) is not dict or set(payload) != fields:
            raise ValueError("invalid boundary payload")
        if envelope["sha256"] != _digest(_encode(payload)):
            raise ValueError("boundary integrity mismatch")
        if (payload["schema"] != "86-tcg-agent/zero-update-boundary/1"
                or payload["python_version"] != list(sys.version_info[:3])
                or payload["torch_version"] != str(torch.__version__)
                or payload["optimizer"] is not None
                or payload["training_resume_verified"] is not False
                or payload["native_chance_exact_resume"] is not False):
            raise ValueError("unsupported boundary provenance or runtime")
        for name in ("in_flight_games", "partial_games"):
            integer(payload[name], name)
            if payload[name]:
                raise ValueError("boundary contains unfinished native games")
        if type(payload["contract"]) is not dict:
            raise ValueError("invalid boundary contract")
        contract = ResumeContract(**payload["contract"])
        if contract != expected_contract:
            raise ValueError("resume contract mismatch")
        if type(payload["counters"]) is not dict:
            raise ValueError("invalid boundary counters")
        counters = BoundaryCounters(**payload["counters"])
        if type(payload["files"]) is not dict or set(payload["files"]) != set(_FILES):
            raise ValueError("invalid package file paths")
        files = {}
        for name, expected in payload["files"].items():
            content_hash(expected)
            files[name] = _read_file(directory, name)
            if _digest(files[name]) != expected:
                raise ValueError("package file integrity mismatch")
        if _digest(files["initial/weights.pt"]) != contract.weights_sha256:
            raise ValueError("initial weights do not match the resume contract")
        population = _population(files["population.json"], contract)
        random_data = json.loads(files["random.json"])
        if type(random_data) is not dict or set(random_data) != {"deck_rng", "policy_rng"}:
            raise ValueError("invalid random state payload")
        deck_rng = _python_rng(random_data["deck_rng"])
        policy_rng = _torch_rng(random_data["policy_rng"])
        policy = load_checkpoint(directory / "initial", vocabulary, config)
        return RestoredBoundary(policy, population, deck_rng, policy_rng, counters, contract)
    except (TypeError, KeyError, OverflowError, OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid zero-update boundary") from error
