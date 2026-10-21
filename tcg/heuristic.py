"""Frozen development-first comparator over the actor-visible decision only."""

from .decision import ActorDecision, SUBMIT


IDENTITY = "development-first/1"
ACTION_CAP = 12
MAIN_PRIORITY = ((9, "evolve"), (8, "attach"), (7, "play"),
                 (13, "attack"), (14, "end"))
RULE_TABLE = {
    "forced": "the sole currently legal microchoice",
    "main_cap_attack": "at turnActionCount >= 12 prefer the first attack",
    "main_cap_end": "at turnActionCount >= 12 otherwise prefer end",
    "main_evolve": "first evolution",
    "main_attach_active": "first attachment whose visible target is active",
    "main_attach": "first other attachment",
    "main_play": "first play",
    "main_attack": "first attack, without estimating damage",
    "main_end": "first end, avoiding optional ability, retreat and discard",
    "setup_bench_fill": "first available setup bench option until maximum",
    "promotion_visible_hp": "highest visible referenced Pokemon HP, native-order ties",
    "neutral_submit": "submit once minimum is satisfied",
    "neutral_native_order": "first available option, without card-value inference",
}
NEUTRAL_RULES = frozenset(("neutral_submit", "neutral_native_order"))


def _scalar(decision: ActorDecision, role: str) -> int:
    try:
        return int(next(token.features[0] for token in decision.tokens if token.role == role))
    except (StopIteration, IndexError) as error:
        raise ValueError(f"heuristic requires the pinned projected {role} token") from error


def _hp(decision, option):
    cards = [decision.tokens[index] for index in option.state_indices
             if decision.tokens[index].role == "card"]
    if len(cards) != 1 or cards[0].zone not in ("active", "bench"):
        return None
    card = cards[0]
    return next((token.features[1] for token in decision.tokens
                 if token.role == "pokemon.hp" and token.owner == card.owner
                 and token.zone == card.zone and token.features[0] == card.features[0]), None)


def selection(decision: ActorDecision) -> tuple[str, str]:
    """Return the legal choice and its frozen rule label for experiment accounting."""
    choices = decision.choices()
    if not choices:
        raise ValueError("heuristic received no legal microchoice")
    if len(choices) == 1:
        return choices[0], "forced"
    available = tuple(option for option in decision.prompt.options if option.option_id in choices)
    context = _scalar(decision, "prompt.context")
    if not 0 <= context <= 48:
        raise ValueError("unqualified heuristic wire context")
    if _scalar(decision, "prompt.type") == 0:
        if _scalar(decision, "state.turnActionCount") >= ACTION_CAP:
            for kind, rule in ((13, "main_cap_attack"), (14, "main_cap_end")):
                for option in available:
                    if option.token.features[0] == kind:
                        return option.option_id, rule
        for kind, label in MAIN_PRIORITY:
            matching = tuple(option for option in available if option.token.features[0] == kind)
            if kind == 8:
                for option in matching:
                    if any(decision.tokens[index].role == "option.inPlayArea"
                           and decision.tokens[index].features == (4,)
                           for index in option.state_indices):
                        return option.option_id, "main_attach_active"
            if matching:
                return matching[0].option_id, "main_" + label
    elif context == 2 and available:
        return available[0].option_id, "setup_bench_fill"
    elif context in (1, 4) and available:
        values = tuple(_hp(decision, option) for option in available)
        if all(value is not None for value in values):
            index = max(range(len(available)), key=lambda index: values[index])
            return available[index].option_id, "promotion_visible_hp"
    if SUBMIT in choices:
        return SUBMIT, "neutral_submit"
    return available[0].option_id, "neutral_native_order"


def choose(decision: ActorDecision) -> str:
    return selection(decision)[0]
