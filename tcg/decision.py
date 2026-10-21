"""Validated actor-visible decisions, independent of simulator dictionaries."""

from dataclasses import dataclass, replace
from math import isfinite

SUBMIT = "__submit__"


def integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def identifier(value: str, name: str) -> None:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"invalid {name}")


@dataclass(frozen=True)
class Token:
    card_id: str | None = None
    owner: int = 0
    zone: str = "global"
    role: str = "state"
    features: tuple[float, ...] = ()
    position: int | None = None

    def __post_init__(self) -> None:
        if self.card_id is not None:
            identifier(self.card_id, "card ID")
        integer(self.owner, "owner")
        if self.owner > 2:
            raise ValueError("owner must be 0 neutral, 1 self, or 2 opponent")
        identifier(self.zone, "zone")
        identifier(self.role, "role")
        if type(self.features) is not tuple or any(
            type(x) not in (float, int) or not isfinite(x) for x in self.features
        ):
            raise ValueError("features must be a tuple of finite numbers")
        if self.position is not None:
            integer(self.position, "position")


@dataclass(frozen=True)
class Option:
    option_id: str
    token: Token
    state_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        identifier(self.option_id, "option ID")
        if self.option_id == SUBMIT or type(self.token) is not Token:
            raise ValueError("invalid option token or reserved ID")
        if type(self.state_indices) is not tuple:
            raise ValueError("state indices must be a tuple")
        for index in self.state_indices:
            integer(index, "state index")


@dataclass(frozen=True)
class LegalPrompt:
    options: tuple[Option, ...]
    minimum: int = 1
    maximum: int = 1
    ordered: bool = False
    allow_repeats: bool = False
    allocation_total: int | None = None
    semantics: str = "independent"

    def __post_init__(self) -> None:
        if type(self.options) is not tuple or any(type(x) is not Option for x in self.options):
            raise ValueError("options must be an immutable tuple of Options")
        ids = tuple(x.option_id for x in self.options)
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate option IDs")
        integer(self.minimum, "minimum")
        integer(self.maximum, "maximum")
        if type(self.ordered) is not bool or type(self.allow_repeats) is not bool:
            raise ValueError("prompt flags must be boolean")
        if self.semantics != "independent":
            raise ValueError("unsupported conditional prompt semantics")
        if self.maximum < self.minimum or (not self.options and self.maximum):
            raise ValueError("impossible cardinality")
        if not self.allow_repeats and self.maximum > len(self.options):
            raise ValueError("unique selection maximum exceeds option count")
        if self.allocation_total is not None:
            integer(self.allocation_total, "allocation total")
            if not self.allow_repeats or self.minimum != self.maximum or self.maximum != self.allocation_total:
                raise ValueError("allocation requires repeated unit picks and an exact total")


@dataclass(frozen=True)
class ActorDecision:
    tokens: tuple[Token, ...]
    prompt: LegalPrompt
    selected: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.tokens) is not tuple or any(type(x) is not Token for x in self.tokens):
            raise ValueError("actor tokens must be an immutable tuple of Tokens")
        if type(self.prompt) is not LegalPrompt or type(self.selected) is not tuple:
            raise ValueError("invalid prompt or selected tuple")
        ids = {x.option_id for x in self.prompt.options}
        if any(type(x) is not str or x not in ids for x in self.selected):
            raise ValueError("unknown selected option")
        if len(self.selected) > self.prompt.maximum:
            raise ValueError("too many selections")
        if not self.prompt.allow_repeats and len(set(self.selected)) != len(self.selected):
            raise ValueError("repeated selection forbidden")
        if any(index >= len(self.tokens) for option in self.prompt.options for index in option.state_indices):
            raise ValueError("option points outside visible state")

    def choices(self) -> tuple[str, ...]:
        available = tuple(
            option.option_id for option in self.prompt.options
            if len(self.selected) < self.prompt.maximum
            and (self.prompt.allow_repeats or option.option_id not in self.selected)
        )
        return available + ((SUBMIT,) if len(self.selected) >= self.prompt.minimum else ())

    def pick(self, option_id: str) -> "ActorDecision":
        if option_id == SUBMIT or option_id not in self.choices():
            raise ValueError("pick must name a currently legal option")
        return replace(self, selected=self.selected + (option_id,))

    def response(self) -> tuple[str, ...] | tuple[tuple[str, int], ...]:
        if SUBMIT not in self.choices():
            raise ValueError("response is incomplete")
        if self.prompt.allocation_total is not None:
            return tuple((o.option_id, self.selected.count(o.option_id)) for o in self.prompt.options)
        if self.prompt.ordered:
            return self.selected
        return tuple(o.option_id for o in self.prompt.options for _ in range(self.selected.count(o.option_id)))
