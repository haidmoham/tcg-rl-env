"""Offline complete-game admission and full-rollout epoch stopping contracts."""

from dataclasses import dataclass, replace
from enum import Enum
from math import fsum

from .decision import identifier, integer
from .population import Population, Snapshot
from .rollout import Episode, EpisodeEnd, EpisodeProvenance, LearnerStep, RolloutTargets, admit_rollout, content_hash, finite


@dataclass(frozen=True)
class CollectionIdentity:
    learner_weights_sha256: str
    learner_policy_version: int
    engine_sha256: str
    projection_sha256: str
    protocol_sha256: str

    def __post_init__(self) -> None:
        integer(self.learner_policy_version, "learner policy version")
        for value in (self.learner_weights_sha256, self.engine_sha256, self.projection_sha256, self.protocol_sha256):
            content_hash(value)


@dataclass(frozen=True)
class GameAssignment:
    episode_id: str
    learner_seat: int
    learner_deck_sha256: str
    opponent_deck_sha256: str
    opponent: Snapshot
    identity: CollectionIdentity

    def __post_init__(self) -> None:
        identifier(self.episode_id, "episode ID")
        integer(self.learner_seat, "learner seat")
        if self.learner_seat not in (0, 1) or type(self.opponent) is not Snapshot or type(self.identity) is not CollectionIdentity:
            raise ValueError("invalid game assignment seat, opponent or collection identity")
        content_hash(self.learner_deck_sha256)
        content_hash(self.opponent_deck_sha256)

    @property
    def expected_provenance(self) -> EpisodeProvenance:
        return EpisodeProvenance(self.episode_id, self.learner_seat, self.learner_deck_sha256,
                                 self.identity.learner_policy_version, self.opponent.weights_sha256)


@dataclass(frozen=True)
class CollectionCounts:
    started_games: int = 0
    truncated_games: int = 0

    def __post_init__(self) -> None:
        integer(self.started_games, "started games")
        integer(self.truncated_games, "truncated games")
        if self.truncated_games > self.started_games:
            raise ValueError("truncated games exceed started games")

    @property
    def truncation_limit_exceeded(self) -> bool:
        return self.started_games >= 100 and self.truncated_games * 100 > self.started_games


class BatchState(str, Enum):
    COLLECTING = "collecting"
    FROZEN = "frozen"
    INVALID = "invalid"


class CollectionFault(str, Enum):
    INFRASTRUCTURE = "infrastructure_failure"
    INVALID_ACTION = "invalid_action"
    NONFINITE = "nonfinite_learning_value"
    INVALID_OLD_STATISTICS = "invalid_old_policy_statistics"
    LEAKAGE = "actor_information_leakage"
    ALL_INVALID_MASK = "all_invalid_mask"


class BatchInvalidated(ValueError):
    pass


@dataclass(frozen=True)
class FrozenBatch:
    identity: CollectionIdentity
    assignments: tuple[GameAssignment, ...]
    episodes: tuple[Episode, ...]
    targets: RolloutTargets
    cumulative_counts: CollectionCounts


class BatchCollector:
    def __init__(self, identity: CollectionIdentity, population: Population,
                 prior_counts: CollectionCounts = CollectionCounts()) -> None:
        if type(identity) is not CollectionIdentity or type(population) is not Population or type(prior_counts) is not CollectionCounts:
            raise ValueError("collector requires typed identity, population and cumulative counts")
        self._identity = identity
        self._population = population
        self._counts = prior_counts
        self._assignments: dict[str, GameAssignment] = {}
        self._episodes: dict[str, Episode] = {}
        self._learner_decisions = 0
        self._state = BatchState.COLLECTING
        self._invalid_reason: str | None = None
        self._check_truncation_rate()

    @property
    def state(self) -> BatchState:
        return self._state

    @property
    def invalid_reason(self) -> str | None:
        return self._invalid_reason

    @property
    def cumulative_counts(self) -> CollectionCounts:
        return self._counts

    @property
    def assignments(self) -> tuple[GameAssignment, ...]:
        return tuple(self._assignments.values())

    @property
    def episodes(self) -> tuple[Episode, ...]:
        return tuple(self._episodes[key] for key in self._assignments if key in self._episodes)

    @property
    def learner_decisions(self) -> int:
        return self._learner_decisions

    @property
    def ready(self) -> bool:
        return (self._state is BatchState.COLLECTING and self._learner_decisions >= 8192
                and len(self._episodes) == len(self._assignments))

    def _invalidate(self, reason: str) -> None:
        self._state = BatchState.INVALID
        self._invalid_reason = reason
        raise BatchInvalidated(reason)

    def _require_collecting(self) -> None:
        if self._state is BatchState.INVALID:
            raise BatchInvalidated(self._invalid_reason)
        if self._state is not BatchState.COLLECTING:
            raise ValueError("batch is already frozen")

    def _check_truncation_rate(self) -> None:
        if self._counts.truncation_limit_exceeded:
            self._invalidate("cumulative administrative truncation rate exceeds 1% after 100 started games")

    def _require_assignment(self, assignment: GameAssignment) -> None:
        if (type(assignment) is not GameAssignment
                or self._assignments.get(assignment.episode_id) != assignment):
            self._invalidate("episode does not match its registered immutable game assignment")
        if assignment.episode_id in self._episodes:
            self._invalidate("episode has already ended")

    def start_game(self, episode_id: str, learner_seat: int, learner_deck_sha256: str,
                   opponent_deck_sha256: str) -> GameAssignment:
        self._require_collecting()
        if self._learner_decisions >= 8192:
            raise ValueError("collection threshold reached; finish pending games and freeze the batch")
        validated = GameAssignment(episode_id, learner_seat, learner_deck_sha256, opponent_deck_sha256,
                                   self._population.archive[0], self._identity)
        if episode_id in self._assignments:
            self._invalidate("episode ID cannot be reused or reassigned")
        next_counts = CollectionCounts(self._counts.started_games + 1, self._counts.truncated_games)
        if next_counts.truncation_limit_exceeded:
            self._invalidate("cumulative administrative truncation rate exceeds 1% after 100 started games")
        current = (Snapshot(self._identity.learner_policy_version, self._identity.learner_weights_sha256)
                   if self._population.sampling_mode == "current_pfsp" else None)
        assignment = replace(validated, opponent=self._population.sample_opponent(current=current))
        self._assignments[episode_id] = assignment
        self._counts = next_counts
        self._check_truncation_rate()
        return assignment

    def complete_game(self, assignment: GameAssignment, steps: tuple[LearnerStep, ...], end: EpisodeEnd) -> None:
        self._require_collecting()
        self._require_assignment(assignment)
        try:
            episode = Episode(assignment.expected_provenance, steps, end)
        except (ValueError, TypeError, OverflowError):
            self._invalidate("invalid episode steps, old statistics or provenance")
        self._episodes[assignment.episode_id] = episode
        if end is EpisodeEnd.INFRASTRUCTURE:
            self._invalidate("infrastructure failure invalidates the affected batch")
        if end is EpisodeEnd.TRUNCATION:
            self._counts = CollectionCounts(self._counts.started_games, self._counts.truncated_games + 1)
            self._check_truncation_rate()
        else:
            self._population.record_result(assignment.opponent, (episode.steps[-1].reward + 1) / 2)
            self._learner_decisions += len(steps)

    def fail_game(self, assignment: GameAssignment, fault: CollectionFault) -> None:
        self._require_collecting()
        self._require_assignment(assignment)
        if type(fault) is not CollectionFault:
            raise ValueError("failure must name a collection fault")
        self._invalidate(f"{assignment.episode_id}: {fault.value}")

    def freeze(self) -> FrozenBatch:
        self._require_collecting()
        if not self.ready:
            raise ValueError("freeze requires 8192 complete learner decisions and no pending games")
        targets = admit_rollout(self.episodes)
        result = FrozenBatch(self._identity, self.assignments, self.episodes, targets, self._counts)
        self._state = BatchState.FROZEN
        return result


@dataclass(frozen=True)
class EpochReport:
    observed_epochs: int
    mean_old_minus_current_logp: float
    stop_remaining_epochs: bool
    stop_reason: str | None


class EpochStopController:
    def __init__(self, batch: FrozenBatch) -> None:
        if type(batch) is not FrozenBatch or len(batch.targets.steps) < 8192:
            raise ValueError("epoch stopping requires a frozen complete rollout")
        self._steps = batch.targets.steps
        self._observed_epochs = 0
        self._stop_reason: str | None = None

    @property
    def can_continue(self) -> bool:
        return self._stop_reason is None

    @property
    def observed_epochs(self) -> int:
        return self._observed_epochs

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    def observe_epoch(self, observations: tuple[tuple[str, int, float], ...]) -> EpochReport:
        if not self.can_continue:
            raise ValueError("remaining epochs are stopped")
        try:
            if type(observations) is not tuple or len(observations) != len(self._steps):
                raise ValueError("epoch observation must cover the entire admitted rollout")
            differences = []
            for step, observation in zip(self._steps, observations):
                if (type(observation) is not tuple or len(observation) != 3
                        or type(observation[0]) is not str or type(observation[1]) is not int
                        or observation[:2] != (step.provenance.episode_id, step.sequence)):
                    raise ValueError("epoch log probabilities are missing, duplicated or out of order")
                finite(observation[2], "current log probability")
                if observation[2] > 0:
                    raise ValueError("current log probability must be nonpositive")
                difference = step.old_log_probability - observation[2]
                finite(difference, "old minus current log probability")
                differences.append(difference)
            kl = fsum(differences) / len(differences)
            finite(kl, "full rollout KL")
        except (ValueError, TypeError, OverflowError):
            self._stop_reason = "invalid_full_rollout_observation"
            raise ValueError("invalid full-rollout epoch observation") from None
        self._observed_epochs += 1
        if kl > 0.02:
            self._stop_reason = "full_rollout_kl_exceeds_0.02"
        elif self._observed_epochs == 4:
            self._stop_reason = "four_epochs_observed"
        return EpochReport(self._observed_epochs, kl, not self.can_continue, self._stop_reason)
