"""Hash-only self-play history and reproducible per-game opponent draws."""

from dataclasses import asdict, dataclass
import hashlib
import json
from math import fsum
from pathlib import Path
import random

from .decision import integer
from .rollout import content_hash


@dataclass(frozen=True)
class Snapshot:
    accepted_update: int
    weights_sha256: str

    def __post_init__(self) -> None:
        integer(self.accepted_update, "accepted update")
        content_hash(self.weights_sha256)

    @classmethod
    def from_file(cls, accepted_update: int, path: str | Path) -> "Snapshot":
        return cls(accepted_update, hashlib.sha256(Path(path).read_bytes()).hexdigest())


def _encode(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


class Population:
    def __init__(self, initial: Snapshot, seed: int = 860001,
                 sampling_mode: str = "legacy_latest_uniform") -> None:
        if type(initial) is not Snapshot or initial.accepted_update != 0:
            raise ValueError("population requires an initial snapshot")
        integer(seed, "population seed")
        if type(sampling_mode) is not str or sampling_mode not in ("legacy_latest_uniform", "current_pfsp"):
            raise ValueError("unsupported population sampling mode")
        self._archive = [initial]
        self._accepted_updates = 0
        self._rng = random.Random(seed)
        self._sampling_mode = sampling_mode
        self._score_histories = {initial.weights_sha256: []}

    @property
    def sampling_mode(self) -> str:
        return self._sampling_mode

    @property
    def score_histories(self) -> dict[str, tuple[float, ...]]:
        return {key: tuple(scores) for key, scores in self._score_histories.items()}

    @property
    def accepted_updates(self) -> int:
        return self._accepted_updates

    @property
    def archive(self) -> tuple[Snapshot, ...]:
        return tuple(self._archive)

    @property
    def sampling_pool(self) -> tuple[Snapshot, ...]:
        return (self._archive[0], *self._archive[1:][-31:])

    def accept_update(self, snapshot: Snapshot | None = None) -> None:
        next_update = self._accepted_updates + 1
        if next_update % 10 == 0:
            if type(snapshot) is not Snapshot or snapshot.accepted_update != next_update:
                raise ValueError("tenth accepted update requires its immutable snapshot")
        elif snapshot is not None:
            raise ValueError("snapshot outside the tenth-update cadence")
        if snapshot is not None:
            self._archive.append(snapshot)
            self._score_histories.setdefault(snapshot.weights_sha256, [])
        self._accepted_updates = next_update

    def _validate_current(self, current: Snapshot) -> None:
        if type(current) is not Snapshot or current.accepted_update != self._accepted_updates:
            raise ValueError("current opponent must match the accepted learner update")

    def historical_weights(self, current: Snapshot) -> tuple[tuple[Snapshot, float], ...]:
        """Local PFSP reconstruction: squared loss rate over each opponent's last 256 scores.

        The author describes PFSP-style sampling but does not disclose this formula or window.
        """
        self._validate_current(current)
        weighted = []
        for snapshot in self.sampling_pool:
            if snapshot.weights_sha256 == current.weights_sha256:
                continue
            scores = self._score_histories.get(snapshot.weights_sha256, [])
            mean = fsum(scores) / len(scores) if scores else 0.5
            weighted.append((snapshot, (1.0 - mean) ** 2))
        return tuple(weighted)

    def record_result(self, opponent: Snapshot, score: float) -> None:
        if (type(opponent) is not Snapshot or opponent.accepted_update > self._accepted_updates
                or opponent.weights_sha256 not in self._score_histories):
            raise ValueError("result requires a known current or historical opponent")
        if type(score) not in (int, float) or score not in (0, 0.5, 1):
            raise ValueError("learner game score must be loss 0, draw 0.5, or win 1")
        scores = self._score_histories[opponent.weights_sha256]
        self._score_histories[opponent.weights_sha256] = [*scores, float(score)][-256:]

    def sample_opponent(self, current: Snapshot | None = None) -> Snapshot:
        if self._sampling_mode == "current_pfsp":
            self._validate_current(current)
            weighted = self.historical_weights(current)
            self._score_histories.setdefault(current.weights_sha256, [])
            if not weighted or self._rng.random() < 0.5:
                return current
            snapshots, weights = zip(*weighted)
            if not any(weights):
                return self._rng.choice(snapshots)
            return self._rng.choices(snapshots, weights=weights, k=1)[0]
        if current is not None:
            raise ValueError("legacy sampling does not accept a current opponent")
        pool = self.sampling_pool
        if len(pool) == 1 or self._rng.random() < 0.5:
            return pool[-1]
        return self._rng.choice(pool[:-1])

    def to_bytes(self) -> bytes:
        payload = {
            "schema": "86-tcg-agent/population/2",
            "accepted_updates": self._accepted_updates,
            "archive": [asdict(snapshot) for snapshot in self._archive],
            "rng_state": self._rng.getstate(),
            "sampling_mode": self._sampling_mode,
            "score_histories": self._score_histories,
        }
        return _encode({"payload": payload, "sha256": hashlib.sha256(_encode(payload)).hexdigest()})

    @classmethod
    def from_bytes(cls, data: bytes) -> "Population":
        try:
            envelope = json.loads(data)
            if type(envelope) is not dict or set(envelope) != {"payload", "sha256"}:
                raise ValueError("invalid population envelope")
            payload = envelope["payload"]
            if type(payload) is not dict:
                raise ValueError("invalid population payload")
            if envelope["sha256"] != hashlib.sha256(_encode(payload)).hexdigest():
                raise ValueError("population integrity mismatch")
            schema = payload.get("schema")
            keys = {"schema", "accepted_updates", "archive", "rng_state"}
            if schema == "86-tcg-agent/population/2":
                keys |= {"sampling_mode", "score_histories"}
            elif schema != "86-tcg-agent/population/1":
                raise ValueError("unsupported population schema")
            if set(payload) != keys:
                raise ValueError("invalid population payload")
            integer(payload["accepted_updates"], "accepted updates")
            archive_data = payload["archive"]
            if type(archive_data) is not list or not archive_data:
                raise ValueError("missing population archive")
            if any(type(item) is not dict or set(item) != {"accepted_update", "weights_sha256"}
                   for item in archive_data):
                raise ValueError("invalid snapshot metadata")
            archive = [Snapshot(**item) for item in archive_data]
            expected_updates = list(range(0, payload["accepted_updates"] + 1, 10))
            if [snapshot.accepted_update for snapshot in archive] != expected_updates:
                raise ValueError("archive does not match accepted-update cadence")
            rng_state = payload["rng_state"]
            if (type(rng_state) is not list or len(rng_state) != 3 or type(rng_state[0]) is not int or rng_state[0] != 3
                    or type(rng_state[1]) is not list or len(rng_state[1]) != 625
                    or any(type(x) is not int or not 0 <= x < 2**32 for x in rng_state[1][:-1])
                    or type(rng_state[1][-1]) is not int or not 0 <= rng_state[1][-1] <= 624
                    or rng_state[2] is not None):
                raise ValueError("invalid population RNG state")
            population = cls(archive[0], sampling_mode=payload.get("sampling_mode", "legacy_latest_uniform"))
            histories = payload.get("score_histories", {snapshot.weights_sha256: [] for snapshot in archive})
            if type(histories) is not dict:
                raise ValueError("invalid opponent score histories")
            for key, scores in histories.items():
                content_hash(key)
                if (type(scores) is not list or len(scores) > 256
                        or any(type(score) not in (int, float) or score not in (0, 0.5, 1) for score in scores)):
                    raise ValueError("invalid opponent score history")
            if not {snapshot.weights_sha256 for snapshot in archive} <= set(histories):
                raise ValueError("archive score histories are missing")
            population._archive = archive
            population._accepted_updates = payload["accepted_updates"]
            population._score_histories = histories
            population._rng.setstate((3, tuple(rng_state[1]), None))
            return population
        except (TypeError, KeyError, OverflowError, json.JSONDecodeError) as error:
            raise ValueError("invalid population metadata") from error
