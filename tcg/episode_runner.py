"""Bounded frozen-policy episodes. This module performs no optimization."""

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Protocol

import torch

from . import cabt_observation
from .cabt import EngineError, EngineFrame
from .cabt_observation import CabtObservationAdapter
from .checkpoint import load_checkpoint
from .collection import GameAssignment
from .decision import ActorDecision, SUBMIT, integer
from .policy import Policy, Vocabulary
from .rollout import Episode, EpisodeEnd, LearnerStep, content_hash, finite
from .training_checkpoint import TrainingIdentity, load_trained_policy


class EpisodeBattle(Protocol):
    library_sha256: str

    def start(self, deck0, deck1, *, engine_seed: int | None = None) -> EngineFrame: ...

    def submit(self, indices) -> EngineFrame: ...


def _copy_state(policy: Policy) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}


def _matches_state(policy: Policy, expected: dict[str, torch.Tensor]) -> bool:
    current = policy.state_dict()
    return current.keys() == expected.keys() and all(torch.equal(current[name], value) for name, value in expected.items())


@dataclass(frozen=True, init=False, eq=False)
class FrozenPolicy:
    policy: Policy = field(repr=False)
    weights_sha256: str
    policy_version: int
    optimizer_steps: int
    training_identity: TrainingIdentity | None
    _checkpoint_state: dict[str, torch.Tensor] = field(repr=False)

    def __init__(self):
        raise TypeError("use a strict FrozenPolicy checkpoint factory")

    @classmethod
    def _bind(cls, policy: Policy, weights_sha256: str, policy_version: int, optimizer_steps: int,
              training_identity: TrainingIdentity | None) -> "FrozenPolicy":
        instance = object.__new__(cls)
        object.__setattr__(instance, "policy", policy.cpu().eval().requires_grad_(False))
        object.__setattr__(instance, "weights_sha256", weights_sha256)
        object.__setattr__(instance, "policy_version", policy_version)
        object.__setattr__(instance, "optimizer_steps", optimizer_steps)
        object.__setattr__(instance, "training_identity", training_identity)
        object.__setattr__(instance, "_checkpoint_state", _copy_state(policy))
        return instance

    @classmethod
    def from_initial_checkpoint(cls, path: str | Path, expected_weights_sha256: str,
                                vocabulary: Vocabulary) -> "FrozenPolicy":
        content_hash(expected_weights_sha256)
        path = Path(path)
        if hashlib.sha256((path / "weights.pt").read_bytes()).hexdigest() != expected_weights_sha256:
            raise ValueError("frozen policy checkpoint weight identity mismatch")
        policy = load_checkpoint(path, vocabulary).cpu().eval().requires_grad_(False)
        if hashlib.sha256((path / "weights.pt").read_bytes()).hexdigest() != expected_weights_sha256:
            raise ValueError("frozen policy checkpoint changed while loading")
        return cls._bind(policy, expected_weights_sha256, 0, 0, None)

    @classmethod
    def from_trained_checkpoint(cls, path: str | Path, expected_weights_sha256: str, vocabulary: Vocabulary, *,
                                expected_identity: TrainingIdentity) -> "FrozenPolicy":
        content_hash(expected_weights_sha256)
        path = Path(path)
        if hashlib.sha256((path / "weights.pt").read_bytes()).hexdigest() != expected_weights_sha256:
            raise ValueError("trained policy checkpoint weight identity mismatch")
        trained = load_trained_policy(path, expected_identity=expected_identity, vocabulary=vocabulary)
        if (trained.weights_sha256 != expected_weights_sha256
                or hashlib.sha256((path / "weights.pt").read_bytes()).hexdigest() != expected_weights_sha256):
            raise ValueError("trained policy checkpoint changed while loading")
        return cls._bind(trained.policy, trained.weights_sha256, trained.counters.policy_version,
                         trained.counters.optimizer_steps, trained.identity)

    def assert_identity(self, expected_weights_sha256: str, vocabulary: Vocabulary,
                        expected_policy_version: int | None = None) -> None:
        if (self.weights_sha256 != expected_weights_sha256 or type(self.policy) is not Policy
                or (expected_policy_version is not None and self.policy_version != expected_policy_version)
                or self.policy.vocabulary != vocabulary or any(module.training for module in self.policy.modules())
                or any(parameter.requires_grad for parameter in self.policy.parameters())
                or any(tensor.device.type != "cpu" for tensor in self.policy.state_dict().values())
                or not _matches_state(self.policy, self._checkpoint_state)):
            raise ValueError("policy weights, frozen CPU evaluation state or vocabulary changed")


@dataclass(frozen=True)
class EpisodeLimits:
    max_submissions: int = 4096
    max_submissions_per_turn: int = 512
    max_seconds: float = 60.0

    def __post_init__(self) -> None:
        integer(self.max_submissions, "submission cap", 1)
        integer(self.max_submissions_per_turn, "per-turn submission cap", 1)
        finite(self.max_seconds, "wall seconds cap")
        if self.max_submissions > 4096 or self.max_submissions_per_turn > 512 or not 0 < self.max_seconds <= 1800:
            raise ValueError("episode limits exceed bounded engineering caps")


@dataclass(frozen=True)
class EpisodeRun:
    episode: Episode
    submissions_attempted: int
    submissions_accepted: int
    learner_forward_calls: int
    opponent_forward_calls: int
    terminal_result: int | None
    reason: str | None
    parameters_unchanged: bool
    wall_seconds: float
    learner_rng_before_sha256: str
    learner_rng_after_sha256: str
    opponent_rng_before_sha256: str
    opponent_rng_after_sha256: str
    engine_seed_control_verified: bool = False
    native_seed_requested: int | None = None
    native_seed_api: str | None = None
    ordered_response_checks: int = 0
    multi_selection_submissions: int = 0


def deck_sha256(cards) -> str:
    return hashlib.sha256(json.dumps(list(cards), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def projection_sha256() -> str:
    return hashlib.sha256(Path(cabt_observation.__file__).read_bytes()).hexdigest()


def _rng_sha256(generator: torch.Generator) -> str:
    return hashlib.sha256(bytes(generator.get_state().tolist())).hexdigest()


def _sample(policy: FrozenPolicy, decision: ActorDecision, generator: torch.Generator) -> tuple[str, float, float]:
    with torch.inference_mode():
        prediction = policy.policy([decision])
        ids = prediction.option_ids[0]
        legal = decision.choices()
        if (len(ids) != len(decision.prompt.options) + 1 or len(set(ids)) != len(ids)
                or tuple(ids) != (*tuple(option.option_id for option in decision.prompt.options), SUBMIT)
                or prediction.logits.shape != (1, len(ids)) or prediction.values.shape != (1,)
                or tuple(bool(value) for value in prediction.legal_mask[0]) != tuple(option in legal for option in ids)):
            raise ValueError("policy output does not preserve the exact legal decision")
        logits = prediction.logits[0]
        for index, option in enumerate(ids):
            if option in legal:
                if not torch.isfinite(logits[index]):
                    raise ValueError("nonfinite legal policy logit")
            elif not torch.isneginf(logits[index]):
                raise ValueError("illegal policy option is not masked")
        log_probabilities = logits.log_softmax(-1)
        if len(legal) == 1:
            chosen = ids.index(legal[0])
        else:
            chosen = int(torch.multinomial(log_probabilities.exp(), 1, generator=generator))
        choice = ids[chosen]
        if choice not in legal:
            raise ValueError("policy selected an illegal micro-action")
        log_probability = float(log_probabilities[chosen])
        value = float(prediction.values[0])
        finite(log_probability, "selected log probability")
        finite(value, "critic value")
        if log_probability > 0 or abs(value) > 1:
            raise ValueError("invalid policy log probability or bounded critic value")
        return choice, log_probability, value


def run_episode(battle: EpisodeBattle, assignment: GameAssignment, learner: FrozenPolicy, opponent: FrozenPolicy,
                adapter: CabtObservationAdapter, exact_decks: tuple[tuple[int, ...], tuple[int, ...]],
                learner_generator: torch.Generator, opponent_generator: torch.Generator, *,
                protocol_path: str | Path, limits: EpisodeLimits = EpisodeLimits(),
                engine_seed: int | None = None) -> EpisodeRun:
    if engine_seed is not None:
        if type(engine_seed) is not int or not 0 <= engine_seed < 2 ** 64:
            raise ValueError("native engine seed must be an unsigned 64-bit integer")
        if getattr(battle, "seed_api", None) != "BattleStartSeeded64":
            raise ValueError("native engine seed requires BattleStartSeeded64")
    if (type(assignment) is not GameAssignment or type(learner) is not FrozenPolicy
            or type(opponent) is not FrozenPolicy or type(adapter) is not CabtObservationAdapter
            or type(limits) is not EpisodeLimits):
        raise ValueError("episode requires typed assignment, frozen policies, strict adapter and limits")
    if (type(exact_decks) is not tuple or len(exact_decks) != 2
            or any(type(deck) not in (list, tuple) or len(deck) != 60
                   or any(type(card) is not int or card <= 0 for card in deck) for deck in exact_decks)):
        raise ValueError("exact decks must be two 60-card lists in absolute seat order")
    decks = tuple(tuple(deck) for deck in exact_decks)
    if (deck_sha256(decks[assignment.learner_seat]) != assignment.learner_deck_sha256
            or deck_sha256(decks[1 - assignment.learner_seat]) != assignment.opponent_deck_sha256
            or battle.library_sha256 != assignment.identity.engine_sha256
            or projection_sha256() != assignment.identity.projection_sha256
            or hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest() != assignment.identity.protocol_sha256):
        raise ValueError("deck, engine, projection or protocol identity does not match the assignment")
    vocabulary = adapter.vocabulary()
    learner.assert_identity(assignment.identity.learner_weights_sha256, vocabulary, assignment.identity.learner_policy_version)
    opponent.assert_identity(assignment.opponent.weights_sha256, vocabulary, assignment.opponent.accepted_update)
    trained_identities = tuple(policy.training_identity for policy in (learner, opponent) if policy.training_identity is not None)
    for identity in trained_identities:
        if (identity.engine_sha256 != assignment.identity.engine_sha256
                or identity.projection_sha256 != assignment.identity.projection_sha256
                or identity.protocol_sha256 != assignment.identity.protocol_sha256):
            raise ValueError("trained policy context differs from the game assignment")
    if len(trained_identities) == 2 and trained_identities[0] != trained_identities[1]:
        raise ValueError("trained policies belong to different baseline lineages")
    if trained_identities and any(policy.training_identity is None
                                 and policy.weights_sha256 != trained_identities[0].initial_weights_sha256
                                 for policy in (learner, opponent)):
        raise ValueError("initial policy differs from the trained baseline origin")
    if (not isinstance(learner_generator, torch.Generator) or not isinstance(opponent_generator, torch.Generator)
            or learner_generator.device.type != "cpu" or opponent_generator.device.type != "cpu"
            or learner_generator is opponent_generator):
        raise ValueError("learner and opponent require separate CPU policy sampling generators")
    learner_rng_before, opponent_rng_before = _rng_sha256(learner_generator), _rng_sha256(opponent_generator)
    if learner_rng_before == opponent_rng_before:
        raise ValueError("learner and opponent sampling streams must have distinct states")
    before_learner, before_opponent = _copy_state(learner.policy), _copy_state(opponent.policy)
    started = time.monotonic()
    deadline = started + limits.max_seconds
    recorded = []
    attempted = accepted = learner_calls = opponent_calls = 0
    ordered_checks = multi_selections = 0
    turn_submissions = {}
    terminal_result = None
    end, reason = EpisodeEnd.TRUNCATION, None
    try:
        adapter.begin_episode(decks)
        frame = battle.start(*decks) if engine_seed is None else battle.start(*decks, engine_seed=engine_seed)
        while True:
            if type(frame) is not EngineFrame:
                raise EngineError("battle returned an unsupported frame")
            result = frame.result
            if type(result) is not int or result not in (-1, 0, 1, 2):
                raise EngineError("unknown absolute CABT winner/draw result")
            if result != -1:
                terminal_result = result
            if time.monotonic() >= deadline:
                reason = "administrative_wall_cap"
                break
            if result != -1:
                if not recorded:
                    raise EngineError("terminal game has no learner decisions to admit")
                reward = 0.0 if result == 2 else (1.0 if result == assignment.learner_seat else -1.0)
                recorded[-1] = replace(recorded[-1], reward=reward)
                end = EpisodeEnd.TERMINAL
                break
            if accepted >= limits.max_submissions:
                reason = "administrative_game_submission_cap"
                break
            integer(frame.actor, "native actor")
            if frame.actor not in (0, 1):
                raise EngineError("unknown native actor")
            turn = frame.wire_observation["current"]["turn"]
            integer(turn, "native turn")
            if turn_submissions.get(turn, 0) >= limits.max_submissions_per_turn:
                reason = "administrative_per_turn_submission_cap"
                break
            decision = adapter.project(frame.wire_observation, frame.actor)
            learner_turn = frame.actor == assignment.learner_seat
            while True:
                if time.monotonic() >= deadline:
                    reason = "administrative_wall_cap_during_prompt"
                    break
                choices = decision.choices()
                if not choices:
                    raise ValueError("all-invalid actor decision")
                if learner_turn:
                    learner_calls += 1
                    choice, old_log_probability, old_value = _sample(learner, decision, learner_generator)
                    recorded.append(LearnerStep(assignment.expected_provenance, len(recorded), decision,
                                                choice, old_log_probability, old_value))
                elif len(choices) == 1:
                    choice = choices[0]
                else:
                    opponent_calls += 1
                    choice, _, _ = _sample(opponent, decision, opponent_generator)
                if choice == SUBMIT:
                    selected_response = decision.response()
                    if not decision.prompt.ordered or selected_response != decision.selected:
                        raise ValueError("CABT response must preserve the exact picked sequence")
                    response = [int(option) for option in selected_response]
                    attempted += 1
                    frame = battle.submit(response)
                    accepted += 1
                    ordered_checks += 1
                    multi_selections += len(response) > 1
                    turn_submissions[turn] = turn_submissions.get(turn, 0) + 1
                    break
                decision = decision.pick(choice)
            if reason is not None:
                break
    except (RuntimeError, ValueError, KeyError, TypeError, OSError, OverflowError) as error:
        end, reason = EpisodeEnd.INFRASTRUCTURE, f"{type(error).__name__}: {error}"
    unchanged = _matches_state(learner.policy, before_learner) and _matches_state(opponent.policy, before_opponent)
    try:
        learner.assert_identity(assignment.identity.learner_weights_sha256, vocabulary, assignment.identity.learner_policy_version)
        opponent.assert_identity(assignment.opponent.weights_sha256, vocabulary, assignment.opponent.accepted_update)
    except ValueError as error:
        end, reason = EpisodeEnd.INFRASTRUCTURE, str(error)
    if not unchanged:
        end, reason = EpisodeEnd.INFRASTRUCTURE, "frozen policy parameters changed during episode"
    if end is not EpisodeEnd.TERMINAL:
        recorded = [replace(step, reward=0.0) for step in recorded]
    episode = Episode(assignment.expected_provenance, tuple(recorded), end)
    return EpisodeRun(episode, attempted, accepted, learner_calls, opponent_calls, terminal_result, reason,
                      unchanged, time.monotonic() - started, learner_rng_before, _rng_sha256(learner_generator),
                      opponent_rng_before, _rng_sha256(opponent_generator),
                      native_seed_requested=engine_seed,
                      native_seed_api="BattleStartSeeded64" if engine_seed is not None else None,
                      ordered_response_checks=ordered_checks,
                      multi_selection_submissions=multi_selections)
