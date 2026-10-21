"""Complete learner-seat episodes and fixed R0 return targets."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
import re

from .decision import ActorDecision, identifier, integer


def content_hash(value: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected lowercase SHA-256 content hash")


def finite(value: float, name: str) -> None:
    if type(value) not in (float, int) or not isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class EpisodeProvenance:
    episode_id: str
    learner_seat: int
    deck_hash: str
    policy_version: int
    opponent_hash: str

    def __post_init__(self) -> None:
        identifier(self.episode_id, "episode ID")
        integer(self.learner_seat, "learner seat")
        if self.learner_seat not in (0, 1):
            raise ValueError("learner seat must be 0 or 1")
        integer(self.policy_version, "policy version")
        content_hash(self.deck_hash)
        content_hash(self.opponent_hash)


@dataclass(frozen=True)
class LearnerStep:
    provenance: EpisodeProvenance
    sequence: int
    decision: ActorDecision
    action_id: str
    old_log_probability: float
    old_value: float
    reward: float = 0.0

    def __post_init__(self) -> None:
        if type(self.provenance) is not EpisodeProvenance or type(self.decision) is not ActorDecision:
            raise ValueError("step requires typed provenance and actor decision")
        integer(self.sequence, "step sequence")
        if self.action_id not in self.decision.choices():
            raise ValueError("recorded action must be legal at the recorded decision")
        for name in ("old_log_probability", "old_value", "reward"):
            finite(getattr(self, name), name)
        if self.old_log_probability > 0 or abs(self.old_value) > 1:
            raise ValueError("invalid old policy log probability or bounded value")


class EpisodeEnd(str, Enum):
    TERMINAL = "terminal"
    TRUNCATION = "administrative_truncation"
    INFRASTRUCTURE = "infrastructure_failure"


@dataclass(frozen=True)
class Episode:
    provenance: EpisodeProvenance
    steps: tuple[LearnerStep, ...]
    end: EpisodeEnd

    def __post_init__(self) -> None:
        if type(self.provenance) is not EpisodeProvenance or type(self.end) is not EpisodeEnd:
            raise ValueError("episode requires typed provenance and ending")
        if type(self.steps) is not tuple or any(type(step) is not LearnerStep for step in self.steps):
            raise ValueError("episode steps must be an immutable tuple")
        if self.end is EpisodeEnd.TERMINAL and not self.steps:
            raise ValueError("terminal episode must contain learner decisions")
        for index, step in enumerate(self.steps):
            if step.provenance != self.provenance or step.sequence != index:
                raise ValueError("mixed provenance or discontinuous learner step sequence")
            last_terminal = self.end is EpisodeEnd.TERMINAL and index == len(self.steps) - 1
            if (last_terminal and step.reward not in (-1, 0, 1)) or (not last_terminal and step.reward != 0):
                raise ValueError("only terminal learner step may carry win/loss/draw reward")


@dataclass(frozen=True)
class RolloutTargets:
    steps: tuple[LearnerStep, ...]
    advantages: tuple[float, ...]
    returns: tuple[float, ...]
    normalized_advantages: tuple[float, ...]
    censored_episode_ids: tuple[str, ...]


def admit_rollout(episodes: tuple[Episode, ...]) -> RolloutTargets:
    if type(episodes) is not tuple or not episodes or any(type(ep) is not Episode for ep in episodes):
        raise ValueError("rollout requires typed episodes")
    if len({ep.provenance.episode_id for ep in episodes}) != len(episodes):
        raise ValueError("duplicate episode ID")
    if len({ep.provenance.policy_version for ep in episodes}) != 1:
        raise ValueError("rollout must contain a single learner policy version")
    if any(episode.end is EpisodeEnd.INFRASTRUCTURE for episode in episodes):
        raise ValueError("infrastructure failure invalidates the entire rollout")
    steps, advantages, returns, censored = [], [], [], []
    for episode in episodes:
        if episode.end is EpisodeEnd.TRUNCATION:
            censored.append(episode.provenance.episode_id)
            continue
        episode_advantages = [0.0] * len(episode.steps)
        next_value, gae = 0.0, 0.0
        for index in range(len(episode.steps) - 1, -1, -1):
            step = episode.steps[index]
            delta = step.reward + next_value - step.old_value
            gae = delta + 0.95 * gae
            finite(gae, "advantage")
            episode_advantages[index] = gae
            next_value = step.old_value
        steps.extend(episode.steps)
        advantages.extend(episode_advantages)
        returns.extend(adv + step.old_value for adv, step in zip(episode_advantages, episode.steps))
    if not steps:
        raise ValueError("rollout contains no complete learner episodes")
    mean = sum(advantages) / len(advantages)
    variance = sum((adv - mean) ** 2 for adv in advantages) / len(advantages)
    scale = sqrt(variance) + 1e-8
    normalized = tuple((adv - mean) / scale for adv in advantages)
    for value in (*returns, *normalized):
        finite(value, "rollout target")
    return RolloutTargets(tuple(steps), tuple(advantages), tuple(returns), normalized, tuple(censored))
