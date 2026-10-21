"""Trained CABT baseline artifacts at complete PPO update boundaries."""

from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import sys

import torch

from .decision import identifier, integer
from .policy import Policy, PolicyConfig, Vocabulary
from .population import Population
from .rollout import content_hash


_FILES = ("weights.pt", "optimizer.pt", "population.json", "samplers.json")
_RECIPE = {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-5, "weight_decay": 0.0,
           "amsgrad": False, "maximize": False, "capturable": False, "differentiable": False}


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _tensor_bytes(state):
    stream = BytesIO()
    torch.save(state, stream)
    return stream.getvalue()


@dataclass(frozen=True)
class TrainingIdentity:
    protocol_sha256: str
    config_sha256: str
    engine_sha256: str
    decks_sha256: str
    projection_sha256: str
    initial_weights_sha256: str

    def __post_init__(self):
        for value in asdict(self).values():
            content_hash(value)


@dataclass(frozen=True)
class TrainingCounters:
    policy_version: int
    optimizer_steps: int
    games_started: int
    games_finished: int
    learner_micro_decisions: int

    def __post_init__(self):
        for name, value in asdict(self).items():
            integer(value, name, 1 if name in ("policy_version", "optimizer_steps") else 0)
        if self.optimizer_steps < self.policy_version:
            raise ValueError("every accepted policy update requires an optimizer step")
        if self.games_started != self.games_finished:
            raise ValueError("all native games must finish before checkpointing")


@dataclass(frozen=True)
class TrainedPolicy:
    policy: Policy
    weights_sha256: str
    identity: TrainingIdentity
    counters: TrainingCounters


@dataclass(frozen=True)
class TrainingCheckpoint:
    policy: Policy
    optimizer: torch.optim.AdamW
    weights_sha256: str
    identity: TrainingIdentity
    counters: TrainingCounters
    population: Population
    deck_rng: random.Random
    policy_rngs: dict[str, torch.Generator]
    native_chance_exact_resume: bool = False


def _policy_state(policy):
    if type(policy) is not Policy:
        raise ValueError("checkpoint requires the declared Policy model")
    state = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("policy weights must be finite float32 tensors")
    return state


def policy_weights_sha256(policy: Policy) -> str:
    """Digest the prospective learned artifact, not a pre-existing checkpoint file."""
    return _digest(_tensor_bytes(_policy_state(policy)))


def _optimizer_state(state, parameters, optimizer_steps):
    if type(state) is not dict or set(state) != {"state", "param_groups"}:
        raise ValueError("invalid AdamW state envelope")
    groups = state["param_groups"]
    required = {*_RECIPE, "params", "foreach", "fused"}
    if type(groups) is not list or len(groups) != 1 or type(groups[0]) is not dict or set(groups[0]) != required:
        raise ValueError("AdamW requires one canonical parameter group")
    group = groups[0]
    if any(type(group[name]) is not type(value) or group[name] != value for name, value in _RECIPE.items()):
        raise ValueError("AdamW recipe mismatch")
    if any(group[name] is not None and group[name] is not False for name in ("foreach", "fused")):
        raise ValueError("unsupported AdamW implementation settings")
    if (type(group["params"]) is not list or any(type(value) is not int for value in group["params"])
            or group["params"] != list(range(len(parameters)))):
        raise ValueError("AdamW parameter order mismatch")
    moments = state["state"]
    if (type(moments) is not dict or not moments
            or any(type(index) is not int or index not in group["params"] for index in moments)):
        raise ValueError("invalid AdamW parameter states")
    copied = {}
    for index, entry in moments.items():
        if type(entry) is not dict or set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("invalid AdamW moments")
        if any(type(value) is not torch.Tensor or not torch.isfinite(value).all() for value in entry.values()):
            raise ValueError("AdamW moments must be finite tensors")
        step = entry["step"]
        if (step.shape != torch.Size([]) or step.dtype != torch.float32
                or not 1 <= step.item() <= optimizer_steps or step.item() != int(step.item())):
            raise ValueError("invalid AdamW step counter")
        for name in ("exp_avg", "exp_avg_sq"):
            value = entry[name]
            if value.shape != parameters[index].shape or value.dtype != parameters[index].dtype:
                raise ValueError("AdamW tensor schema mismatch")
        if (entry["exp_avg_sq"] < 0).any():
            raise ValueError("negative AdamW second moments")
        copied[index] = {name: value.detach().cpu().clone() for name, value in entry.items()}
    if max(entry["step"].item() for entry in copied.values()) != optimizer_steps:
        raise ValueError("optimizer step count differs from the boundary counter")
    return {"state": copied, "param_groups": [dict(group)]}


def _deck_rng(state):
    if (type(state) not in (list, tuple) or len(state) != 3 or type(state[0]) is not int or state[0] != 3
            or type(state[1]) not in (list, tuple) or len(state[1]) != 625
            or any(type(value) is not int or not 0 <= value < 2**32 for value in state[1][:-1])
            or not any(state[1][:-1]) or type(state[1][-1]) is not int or not 0 <= state[1][-1] <= 624
            or state[2] is not None and (type(state[2]) is not float or not math.isfinite(state[2]))):
        raise ValueError("invalid deck sampling RNG state")
    result = random.Random(0)
    result.setstate((3, tuple(state[1]), state[2]))
    return result


def _policy_rngs(states):
    if type(states) is not dict or not states:
        raise ValueError("named CPU policy sampling generators required")
    generators = {}
    for name, state in states.items():
        identifier(name, "generator name")
        generator = torch.Generator(device="cpu")
        if (type(state) is not list or len(state) != generator.get_state().numel()
                or any(type(value) is not int or not 0 <= value <= 255 for value in state)):
            raise ValueError("invalid CPU policy sampling RNG state")
        try:
            generator.set_state(torch.tensor(state, dtype=torch.uint8))
        except RuntimeError as error:
            raise ValueError("invalid CPU policy sampling RNG state") from error
        generators[name] = generator
    return generators


def _population(data, identity, counters, weights_hash):
    population = Population.from_bytes(data)
    if (population.accepted_updates != counters.policy_version
            or population.archive[0].weights_sha256 != identity.initial_weights_sha256):
        raise ValueError("population origin or accepted update mismatch")
    if counters.policy_version % 10 == 0 and population.archive[-1].weights_sha256 != weights_hash:
        raise ValueError("current cadence snapshot must match the learned weights")
    return population


def save_training_checkpoint(
    directory: str | Path, *, policy: Policy, optimizer: torch.optim.AdamW,
    identity: TrainingIdentity, counters: TrainingCounters, population: Population,
    deck_rng: random.Random, policy_rngs: dict[str, torch.Generator],
    in_flight_games: int, partial_minibatch: bool,
) -> str:
    """Save only accepted update boundaries after all games and minibatches finish."""
    integer(in_flight_games, "in-flight games")
    if type(partial_minibatch) is not bool or partial_minibatch or in_flight_games:
        raise ValueError("checkpoint requires complete games and optimizer minibatches")
    if type(identity) is not TrainingIdentity or type(counters) is not TrainingCounters:
        raise ValueError("typed training identity and counters required")
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise ValueError("canonical AdamW optimizer required")
    policy_state = _policy_state(policy)
    parameters = tuple(policy.parameters())
    if len(optimizer.param_groups[0]["params"]) != len(parameters) or any(
        actual is not expected for actual, expected in zip(optimizer.param_groups[0]["params"], parameters)
    ):
        raise ValueError("optimizer must own every policy parameter in native order")
    weights = _tensor_bytes(policy_state)
    weights_hash = _digest(weights)
    optimizer_state = _optimizer_state(optimizer.state_dict(), parameters, counters.optimizer_steps)
    if type(population) is not Population or type(deck_rng) is not random.Random:
        raise ValueError("typed population and deck RNG required")
    population_data = population.to_bytes()
    _population(population_data, identity, counters, weights_hash)
    deck_state = deck_rng.getstate()
    _deck_rng(deck_state)
    if type(policy_rngs) is not dict or any(type(value) is not torch.Generator
                                           or value.device != torch.device("cpu") for value in policy_rngs.values()):
        raise ValueError("named CPU policy sampling generators required")
    states = {name: generator.get_state().tolist() for name, generator in policy_rngs.items()}
    _policy_rngs(states)
    files = {"weights.pt": weights, "optimizer.pt": _tensor_bytes(optimizer_state),
             "population.json": population_data, "samplers.json": _encode({"deck": deck_state, "policy": states})}
    payload = {
        "schema": "86-tcg-agent/trained-baseline/1", "provenance": "random-origin-terminal-outcome-ppo",
        "identity": asdict(identity), "counters": asdict(counters),
        "model_config": asdict(policy.config), "vocabulary": asdict(policy.vocabulary),
        "weights_sha256": weights_hash, "files": {name: _digest(data) for name, data in files.items()},
        "python_version": list(sys.version_info[:3]), "torch_version": str(torch.__version__),
        "platform": sys.platform, "cpu_threads": torch.get_num_threads(),
        "in_flight_games": 0, "partial_minibatch": False, "native_chance_exact_resume": False,
        "resume_scope": "same-runtime CPU optimizer and sampler state; native chance is not replayable",
    }
    manifest = _encode({"payload": payload, "sha256": _digest(_encode(payload))})
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name, data in (*files.items(), ("manifest.json", manifest)):
        with (directory / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    return weights_hash


def _read_package(directory, expected_identity, vocabulary, config):
    if type(expected_identity) is not TrainingIdentity:
        raise ValueError("typed expected training identity required")
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("checkpoint must not be a symbolic link")
    directory = directory.resolve(strict=True)
    if {path.name for path in directory.iterdir()} != {*_FILES, "manifest.json"}:
        raise ValueError("incomplete or unexpected checkpoint files")
    files = {}
    for name in (*_FILES, "manifest.json"):
        path = directory / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(directory):
            raise ValueError("checkpoint file escapes its directory")
        files[name] = path.read_bytes()
    envelope = json.loads(files.pop("manifest.json"))
    if type(envelope) is not dict or set(envelope) != {"payload", "sha256"}:
        raise ValueError("invalid trained checkpoint envelope")
    payload = envelope["payload"]
    fields = {"schema", "provenance", "identity", "counters", "model_config", "vocabulary", "weights_sha256",
              "files", "python_version", "torch_version", "in_flight_games", "partial_minibatch",
              "native_chance_exact_resume", "resume_scope", "platform", "cpu_threads"}
    if type(payload) is not dict or set(payload) != fields or envelope["sha256"] != _digest(_encode(payload)):
        raise ValueError("trained checkpoint integrity mismatch")
    if (payload["schema"] != "86-tcg-agent/trained-baseline/1"
            or payload["provenance"] != "random-origin-terminal-outcome-ppo"
            or payload["python_version"] != list(sys.version_info[:3])
            or payload["torch_version"] != str(torch.__version__)
            or payload["platform"] != sys.platform or type(payload["cpu_threads"]) is not int
            or payload["cpu_threads"] != torch.get_num_threads()
            or payload["partial_minibatch"] is not False or payload["native_chance_exact_resume"] is not False):
        raise ValueError("unsupported trained checkpoint provenance, boundary or runtime")
    integer(payload["in_flight_games"], "in-flight games")
    if payload["in_flight_games"]:
        raise ValueError("checkpoint contains active native games")
    if (type(payload["identity"]) is not dict or TrainingIdentity(**payload["identity"]) != expected_identity):
        raise ValueError("training identity mismatch")
    if type(payload["counters"]) is not dict:
        raise ValueError("invalid training counters")
    counters = TrainingCounters(**payload["counters"])
    if (_encode(payload["model_config"]) != _encode(asdict(config))
            or _encode(payload["vocabulary"]) != _encode(asdict(vocabulary))):
        raise ValueError("policy configuration or vocabulary mismatch")
    if type(payload["files"]) is not dict or set(payload["files"]) != set(_FILES):
        raise ValueError("invalid trained checkpoint file paths")
    for name, expected in payload["files"].items():
        content_hash(expected)
        if _digest(files[name]) != expected:
            raise ValueError("trained checkpoint file integrity mismatch")
    content_hash(payload["weights_sha256"])
    if payload["weights_sha256"] != _digest(files["weights.pt"]):
        raise ValueError("trained weight identity mismatch")
    return files, counters, payload["weights_sha256"]


def _load_policy(data, vocabulary, config):
    with torch.random.fork_rng(devices=[]):
        policy = Policy(vocabulary, config).cpu().eval()
    state = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
    expected = policy.state_dict()
    if type(state) is not dict or set(state) != set(expected) or any(
        type(value) is not torch.Tensor or value.shape != expected[name].shape
        or value.dtype != expected[name].dtype or not torch.isfinite(value).all()
        for name, value in state.items()
    ):
        raise ValueError("trained policy tensor schema mismatch")
    policy.load_state_dict(state, strict=True)
    return policy


def load_trained_policy(
    directory: str | Path, *, expected_identity: TrainingIdentity,
    vocabulary: Vocabulary, config: PolicyConfig = PolicyConfig(),
) -> TrainedPolicy:
    """Load inference weights after full file integrity checks, without an optimizer."""
    try:
        files, counters, weights_hash = _read_package(directory, expected_identity, vocabulary, config)
        policy = _load_policy(files["weights.pt"], vocabulary, config)
        policy.requires_grad_(False)
        return TrainedPolicy(policy, weights_hash, expected_identity, counters)
    except (TypeError, KeyError, OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid trained policy checkpoint") from error


def load_training_checkpoint(
    directory: str | Path, *, expected_identity: TrainingIdentity,
    vocabulary: Vocabulary, config: PolicyConfig = PolicyConfig(),
) -> TrainingCheckpoint:
    """Restore same-runtime CPU state; this does not restore native battle chance."""
    try:
        files, counters, weights_hash = _read_package(directory, expected_identity, vocabulary, config)
        population = _population(files["population.json"], expected_identity, counters, weights_hash)
        samplers = json.loads(files["samplers.json"])
        if type(samplers) is not dict or set(samplers) != {"deck", "policy"}:
            raise ValueError("invalid training samplers")
        deck_rng = _deck_rng(samplers["deck"])
        policy_rngs = _policy_rngs(samplers["policy"])
        policy = _load_policy(files["weights.pt"], vocabulary, config)
        parameters = tuple(policy.parameters())
        state = torch.load(BytesIO(files["optimizer.pt"]), map_location="cpu", weights_only=True)
        state = _optimizer_state(state, parameters, counters.optimizer_steps)
        optimizer = torch.optim.AdamW(parameters, **_RECIPE)
        optimizer.load_state_dict(state)
        return TrainingCheckpoint(policy, optimizer, weights_hash, expected_identity, counters,
                                  population, deck_rng, policy_rngs)
    except (TypeError, KeyError, OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid trained optimizer checkpoint") from error
