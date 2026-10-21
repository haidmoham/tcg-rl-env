"""Strict CABT wire-to-policy projection. Native qualification is separate."""

from dataclasses import replace
from collections import Counter

from .decision import ActorDecision, LegalPrompt, Option, Token
from .policy import Vocabulary


STATE_FIELDS = (
    "turn", "turnActionCount", "yourIndex", "firstPlayer", "supporterPlayed",
    "stadiumPlayed", "energyAttached", "retreated", "result", "stadium", "looking", "players",
)
PLAYER_FIELDS = (
    "active", "bench", "benchMax", "deckCount", "discard", "prize", "handCount", "hand",
    "poisoned", "burned", "asleep", "paralyzed", "confused",
)
CARD_FIELDS = ("id", "serial", "playerIndex")
POKEMON_FIELDS = (
    "id", "serial", "playerIndex", "hp", "maxHp", "appearThisTurn", "energies", "energyCards", "tools", "preEvolution",
)
SELECT_FIELDS = (
    "type", "context", "minCount", "maxCount", "remainDamageCounter", "remainEnergyCost",
    "option", "deck", "contextCard", "effect",
)
OPTION_FIELDS = (
    "type", "number", "area", "index", "playerIndex", "toolIndex", "energyIndex", "count",
    "inPlayArea", "inPlayIndex", "attackId", "cardId", "serial", "specialConditionType",
)
OPTION_PARAMETERS = {
    0: ("number",), 1: (), 2: (), 3: ("area", "index", "playerIndex"),
    4: ("area", "index", "playerIndex", "toolIndex"),
    5: ("area", "index", "playerIndex", "energyIndex"),
    6: ("area", "index", "playerIndex", "energyIndex", "count"),
    7: ("index",), 8: ("area", "index", "inPlayArea", "inPlayIndex"),
    9: ("area", "index", "inPlayArea", "inPlayIndex"),
    10: ("area", "index"), 11: ("area", "index"), 12: (),
    13: ("attackId",), 14: (), 15: ("cardId", "serial"), 16: ("specialConditionType",),
}
LOG_FIELDS = (
    "type", "playerIndex", "hasBasicPokemon", "cardId", "serial", "fromArea", "toArea",
    "cardIdActive", "serialActive", "cardIdBench", "serialBench", "cardIdBefore", "serialBefore",
    "cardIdAfter", "serialAfter", "cardIdTarget", "serialTarget", "attackId", "value",
    "putDamageCounter", "isRecover", "head", "result", "reason",
)
LOG_PARAMETERS = {
    0: ("playerIndex",), 1: ("playerIndex", "hasBasicPokemon"),
    2: ("playerIndex",), 3: ("playerIndex",),
    4: ("playerIndex", "cardId", "serial"), 5: ("playerIndex",),
    6: ("playerIndex", "cardId", "serial", "fromArea", "toArea"),
    7: ("playerIndex", "fromArea", "toArea"),
    8: ("playerIndex", "cardIdActive", "serialActive", "cardIdBench", "serialBench"),
    9: ("playerIndex", "cardIdBefore", "serialBefore", "cardIdAfter", "serialAfter"),
    10: ("playerIndex", "cardId", "serial"),
    11: ("playerIndex", "cardId", "serial", "cardIdTarget", "serialTarget"),
    12: ("playerIndex", "cardId", "serial", "cardIdTarget", "serialTarget"),
    13: ("playerIndex", "cardId", "serial", "cardIdTarget", "serialTarget"),
    14: ("playerIndex", "cardId", "serial", "cardIdBefore", "serialBefore", "cardIdAfter", "serialAfter"),
    15: ("playerIndex", "cardId", "serial", "attackId"),
    16: ("playerIndex", "cardId", "serial", "value", "putDamageCounter"),
    **{kind: ("playerIndex", "isRecover", "cardId", "serial") for kind in range(17, 22)},
    22: ("playerIndex", "head"), 23: ("result", "reason"),
}
AREAS = {
    14: "deck_bottom",
    1: "deck", 2: "hand", 3: "discard", 4: "active", 5: "bench", 6: "prize",
    7: "stadium", 8: "energyCards", 9: "tools", 10: "preEvolution", 11: "player", 12: "looking",
}
ZONES = ("global", "player", "active", "bench", "deck", "discard", "prize", "hand",
         "stadium", "looking", "energyCards", "tools", "preEvolution", "prompt", "history", "memory")
ROLES = tuple(dict.fromkeys((
    "card", "unknown", "energy", "option", "event", "attachment", "registered_deck", "seen_card",
    *("state." + key for key in STATE_FIELDS[:9] if key != "yourIndex"),
    *("player." + key for key in PLAYER_FIELDS if key not in ("active", "bench", "discard", "prize", "hand")),
    *("pokemon." + key for key in ("hp", "maxHp", "appearThisTurn")),
    *("prompt." + key for key in ("type", "context", "remainDamageCounter", "remainEnergyCost", "contextCard", "effect")),
    *("option." + key for key in OPTION_FIELDS if key != "serial"),
    *("event." + key for key in LOG_FIELDS if not key.startswith("serial")),
)))


def _shape(value, fields, name, required=()):
    if type(value) is not dict or set(value) - set(fields) or set(required) - set(value):
        raise ValueError(f"unsupported {name} shape")
    return value


def _int(value, name, minimum=0, maximum=None):
    if (type(value) is not int or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"invalid {name}: {value!r}")
    return value


def _bool(value, name):
    if type(value) is not bool:
        raise ValueError(f"invalid {name}")
    return int(value)


def _list(value, name):
    if type(value) is not list:
        raise ValueError(f"invalid {name}")
    return value


def _owner(seat, actor):
    return 1 if _int(seat, "playerIndex", 0, 1) == actor else 2


def _result(value, actor):
    _int(value, "result", -1, 2)
    return value if value in (-1, 2) else (1 if value == actor else 0)


class CabtObservationAdapter:
    """Project once per native frame. Use ActorDecision.pick for policy microchoices.

    Histories contain only each seat's own observation deltas. Reset between games.
    Remember seen card identities, not hidden locations. All prize identities are masked.
    """

    def __init__(self, card_ids: tuple[str, ...]):
        self._vocabulary = Vocabulary(card_ids, ZONES, ROLES)
        self.reset()

    def reset(self) -> None:
        self._history = {0: [], 1: []}
        self._seen = {0: set(), 1: set()}
        self._registered_decks = None

    def begin_episode(self, registered_decks=None) -> None:
        self.reset()
        if registered_decks is not None:
            if len(registered_decks) != 2 or any(len(deck) != 60 for deck in registered_decks):
                raise ValueError("registered decks must contain two exact 60-card lists")
            self._registered_decks = tuple(Counter(self._card_id(card) for card in deck)
                                           for deck in registered_decks)

    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    def _card_id(self, value):
        card_id = str(_int(value, "card ID", 1))
        if card_id not in self._vocabulary.card_ids:
            raise ValueError(f"unknown card ID {card_id}")
        return card_id

    def project(self, wire_observation: dict, actor: int) -> ActorDecision:
        _int(actor, "actor", 0, 1)
        wire = _shape(wire_observation, ("current", "select", "logs", "search_begin_input"),
                      "observation", ("current", "select", "logs"))
        current = _shape(wire["current"], STATE_FIELDS, "current", STATE_FIELDS)
        select = _shape(wire["select"], SELECT_FIELDS, "select", SELECT_FIELDS)
        if current["yourIndex"] != actor or type(current["yourIndex"]) is not int:
            raise ValueError("actor must match current.yourIndex")
        if current["result"] != -1:
            raise ValueError("terminal observation has no actor decision")
        tokens = []
        entities = {}
        serials = {}
        seen = set(self._seen[actor])

        def add(role, value=(), owner=0, zone="global", card_id=None, position=None):
            index = len(tokens)
            tokens.append(Token(card_id, owner, zone, role, tuple(value), position))
            return index

        def card(value, owner, zone, index, parent=None, pokemon=False):
            if value is None or zone == "prize":
                return add("unknown", (index,), owner, zone)
            fields = POKEMON_FIELDS if pokemon else CARD_FIELDS
            required = tuple(key for key in fields if not pokemon or key != "playerIndex")
            value = _shape(value, fields, "pokemon" if pokemon else "card", required)
            if pokemon and "playerIndex" in value and _owner(value["playerIndex"], actor) != owner:
                raise ValueError("pokemon owner differs from board location")
            if not pokemon:
                owner = _owner(value["playerIndex"], actor)
            card_id = self._card_id(value["id"])
            seen.add((owner, card_id))
            serial = _int(value["serial"], "serial")
            if serial in serials and serials[serial][0] != card_id:
                raise ValueError("inconsistent visible card serial")
            location = (index,) if parent is None else (index, *parent)
            token_index = add("card", location, owner, zone, card_id)
            serials[serial] = (card_id, token_index)
            if parent is not None:
                add("attachment", location, owner, zone)
            if pokemon:
                for key in ("hp", "maxHp", "appearThisTurn"):
                    number = (_bool(value[key], key) if key == "appearThisTurn"
                              else _int(value[key], key, None if key == "hp" else 0))
                    add("pokemon." + key, (index, number), owner, zone)
                for energy_index, energy in enumerate(_list(value["energies"], "energies")):
                    add("energy", (index, energy_index, _int(energy, "energy", 0, 11)), owner, zone)
                for attachment in ("energyCards", "tools", "preEvolution"):
                    for child_index, child in enumerate(_list(value[attachment], attachment)):
                        child_token = card(child, owner, attachment, child_index, (ZONES.index(zone), index))
                        entities[(owner, zone, index, attachment, child_index)] = child_token
            return token_index

        for key in STATE_FIELDS[:9]:
            if key == "yourIndex":
                continue
            value = current[key]
            if key in ("supporterPlayed", "stadiumPlayed", "energyAttached", "retreated"):
                value = _bool(value, key)
            elif key == "firstPlayer":
                value = _int(value, key, -1, 1)
                value = 0 if value == -1 else _owner(value, actor)
            elif key == "result":
                value = _result(value, actor)
            else:
                value = _int(value, key)
            add("state." + key, (value,))
        players = _list(current["players"], "players")
        if len(players) != 2:
            raise ValueError("expected two players")
        for seat in (actor, 1 - actor):
            owner = _owner(seat, actor)
            player = _shape(players[seat], PLAYER_FIELDS, "player", PLAYER_FIELDS)
            for key in ("benchMax", "deckCount", "handCount", "poisoned", "burned", "asleep", "paralyzed", "confused"):
                number = _int(player[key], key) if key in ("benchMax", "deckCount", "handCount") else _bool(player[key], key)
                add("player." + key, (number,), owner, "player")
            for zone in ("active", "bench", "discard", "prize", "hand"):
                if zone == "hand" and owner == 2:
                    continue
                values = _list(player[zone], zone)
                if zone == "active" and len(values) > 1:
                    raise ValueError("multiple active pokemon")
                if zone == "hand" and len(values) != player["handCount"]:
                    raise ValueError("hand count mismatch")
                for index, value in enumerate(values):
                    entities[(owner, zone, index)] = card(value, owner, zone, index, pokemon=zone in ("active", "bench"))
        for zone in ("stadium", "looking"):
            values = current[zone] if current[zone] is not None else []
            for index, value in enumerate(_list(values, zone)):
                entities[(0, zone, index)] = card(value, 0, zone, index)
        if select["deck"] is not None:
            for index, value in enumerate(_list(select["deck"], "select.deck")):
                entities[(1, "deck", index)] = card(value, 1, "deck", index)
        for key in ("contextCard", "effect"):
            if select[key] is not None:
                index = card(select[key], 0, "prompt", 0)
                add("prompt." + key, (index,), zone="prompt")

        select_type = _int(select["type"], "select.type", 0, 10)
        context = _int(select["context"], "select.context", 0, 48)
        if context == 10:
            raise ValueError("unsupported ordering or allocation semantics")
        for key in ("type", "context", "remainDamageCounter", "remainEnergyCost"):
            add("prompt." + key, (_int(select[key], key, None if key == "remainEnergyCost" else 0),), zone="prompt")
        single_choice = select["minCount"] in (0, 1) and select["maxCount"] == 1
        allocation = context == 14 or select_type == 4
        if allocation and not single_choice:
            raise ValueError("unsupported multi-choice allocation semantics")
        if (select["remainDamageCounter"] and not (context == 14 and single_choice)
                or select["remainEnergyCost"] and not (select_type == 4 and single_choice)):
            raise ValueError("unsupported conditional prompt semantics")

        options = []
        for option_index, raw in enumerate(_list(select["option"], "option")):
            raw = _shape(raw, OPTION_FIELDS, "option", ("type",))
            kind = _int(raw["type"], "option.type", 0, 16)
            parameters = OPTION_PARAMETERS[kind]
            if any(raw.get(key) is not None for key in OPTION_FIELDS if key not in ("type", *parameters)):
                raise ValueError("unexpected option parameter")
            if any(raw.get(key) is None for key in parameters):
                raise ValueError("missing option parameter")
            indices = []
            owner = _owner(raw["playerIndex"], actor) if "playerIndex" in parameters else 1
            for key in parameters:
                if key == "serial":
                    continue
                value = raw[key]
                if key == "playerIndex":
                    value = owner
                elif key == "cardId":
                    value = _int(value, key)
                    if value:
                        identity = self._card_id(value)
                        visible = serials.get(raw["serial"])
                        if visible is None or visible[0] != identity:
                            raise ValueError("skill refers to a nonvisible card")
                        indices.append(visible[1])
                        indices.append(add("option.cardId", (), owner, "prompt", identity))
                        continue
                elif key in ("area", "inPlayArea"):
                    if type(value) is not int or value not in AREAS:
                        raise ValueError("unknown area")
                else:
                    value = _int(value, key, 0, 4 if key == "specialConditionType" else None)
                indices.append(add("option." + key, (value,), owner, "prompt"))
            if "index" in parameters:
                zone = "hand" if kind == 7 else AREAS[raw["area"]]
                lookup_owner = 0 if zone in ("looking", "stadium") else owner
                lookup = (lookup_owner, zone, raw["index"])
                if kind in (4, 5, 6):
                    attachment = "tools" if kind == 4 else "energyCards"
                    lookup += (attachment, raw["toolIndex" if kind == 4 else "energyIndex"])
                if lookup not in entities:
                    raise ValueError("option refers outside visible state")
                indices.append(entities[lookup])
            if "inPlayIndex" in parameters:
                lookup = (1, AREAS[raw["inPlayArea"]], raw["inPlayIndex"])
                if lookup not in entities:
                    raise ValueError("option target refers outside visible state")
                indices.append(entities[lookup])
            options.append(Option(str(option_index), Token(owner=owner, zone="prompt", role="option", features=(kind,)), tuple(indices)))

        logs = _list(wire["logs"], "logs")
        new_events = []
        for event in logs:
            start = len(tokens)
            event = _shape(event, LOG_FIELDS, "log", ("type",))
            kind = _int(event["type"], "log.type", 0, 23)
            parameters = LOG_PARAMETERS[kind]
            if any(event.get(key) is not None for key in LOG_FIELDS if key not in ("type", *parameters)):
                raise ValueError("unexpected log parameter")
            if any(event.get(key) is None for key in parameters):
                raise ValueError("missing log parameter")
            owner = _owner(event["playerIndex"], actor) if "playerIndex" in parameters else 0
            add("event", (kind,), owner, "history")
            hide_card = (kind == 4 and owner == 2) or (kind == 6 and
                        (event["fromArea"] == 6 or event["toArea"] == 6 or
                         owner == 2 and event["toArea"] in (1, 2, 14)))
            for key in parameters:
                if key.startswith("serial") or key == "playerIndex":
                    continue
                value = event[key]
                identity = None
                if key.startswith("cardId"):
                    if hide_card:
                        continue
                    identity = self._card_id(value)
                    if key == "cardId" and owner:
                        seen.add((owner, identity))
                    values = ()
                elif key in ("hasBasicPokemon", "putDamageCounter", "isRecover", "head"):
                    values = (_bool(value, key),)
                elif key == "result":
                    values = (_result(value, actor),)
                else:
                    values = (_int(value, key, -1000000 if key == "value" else 0),)
                    if key in ("fromArea", "toArea") and value not in AREAS:
                        raise ValueError(f"unknown event area {value}")
                    if key == "reason" and value not in (1, 2, 3, 4):
                        raise ValueError("unknown result reason")
                add("event." + key, values, owner, "history", identity)
            new_events.append(tuple(tokens[start:]))
            del tokens[start:]
        history = (self._history[actor] + new_events)[-32:]
        for position, event_tokens in enumerate(history):
            tokens.extend(replace(token, position=position) for token in event_tokens)
        for owner, identity in sorted(seen):
            add("seen_card", (), owner, "memory", identity)
        if self._registered_decks is not None:
            for identity, count in sorted(self._registered_decks[actor].items()):
                add("registered_deck", (count,), 1, "deck", identity)
        minimum = _int(select["minCount"], "minCount")
        maximum = _int(select["maxCount"], "maxCount")
        decision = ActorDecision(tuple(tokens), LegalPrompt(tuple(options), minimum, maximum, ordered=True))
        self._history[actor] = history
        self._seen[actor] = seen
        return decision
