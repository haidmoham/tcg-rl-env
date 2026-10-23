import ctypes
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import subprocess

from tcg.live_parity import LIVE_PINS, digest


DRAW_FACETS = ("effect", "legal_prompts", "information_boundaries", "interactions")
DRAW_MANIFEST_SHA256 = "a27c8874662c2a707274ef0e8d29f54fc35f3a38dd4a12e0d7583b72f40a4ee3"
DRAW_MODES = (
    ("regular", 12, 3, 11, 2, 3, False),
    ("six-prizes", 12, 3, 11, 2, 6, False),
    ("short", 2, 0, 2, 0, 3, False),
    ("empty", 0, 0, 0, 0, 3, False),
    ("hand-only", 0, 3, 0, 2, 3, False),
    ("opponent-only", 0, 0, 8, 2, 3, False),
    ("owner-only", 8, 2, 0, 0, 3, False),
    ("one-card", 1, 0, 1, 0, 3, False),
    ("exact-four", 4, 0, 4, 0, 3, False),
    ("exact-six", 6, 0, 6, 0, 3, False),
    ("exact-eight-six-prizes", 8, 0, 8, 0, 6, False),
    ("short-six-prizes", 2, 1, 2, 1, 6, False),
    ("large-hand", 4, 12, 4, 11, 5, False),
    ("first-turn", 12, 3, 11, 2, 6, True),
)
REVIEWED_SOURCES = {
    "CardImpl.h": "286a51820d36f9b60b5b13c58bc2edff352eb4050581ddadf1960f13fd6f21a9",
    "RuntimeParityCorrections.h": "163955867e50b2d32d37cafa056860fe7911720cdd3c40a3d59f53b2e1a1c396",
    "TargetList.h": "c93bea62ea3de7b733e3cbbc8d75cde9a612c3dd956516597a8fe8858dc8eac7",
    "experiments/draw_supporters/cabt.cpp": "c1ff0778d64df3984f05bd0b15e8b56314081accb4ea00e8328ad499a3015557",
    "experiments/draw_supporters/Commands.h": "5c313b6314dcd5daacd190459d4117b1f687e5935bc605fa1821e06fd2dd6ee2",
}
EXPANDED_REVIEWED_SOURCES = {
    "TargetList.h": "7e48c15e8568fbdd32d8d32257b00e16c57709832bddf8761b0e7321c2965198",
    "experiments/draw_supporters/cabt.cpp": "3938f7b2daa082b6f3b1620e5a77a6592d3409858c8b86a358fbc4f4fb920efd",
}
RESPONSE_FIELDS = {
    "programSha256": str, "cardSourceID": str, "mode": str, "canPlay": bool,
    "prompted": bool, "completed": bool, "hand": int, "deck": int,
    "enemyHand": int, "enemyDeck": int, "secondSupporter": bool,
    "turnEnded": bool, "gameOver": bool,
}
ZONE_FIELDS = ("ownUnion", "enemyUnion", "discard", "enemyDiscard")
VIEW_FIELDS = ("ownerView", "opponentView", "enemyOwnerView", "enemyOpponentView")
RAW_ZONES = ("hand", "deck", "enemyHand", "enemyDeck")


@dataclass(frozen=True)
class DrawProgram:
    source_id: str
    native_id: int
    name: str
    lillie: bool
    rows: int


DRAW_PROGRAMS = {
    "457be63d5ab63e5bf308df8e59516aee0c86a88fe74fa46f1ce118672721e9a4":
        DrawProgram("me1_119", 1227, "Lillie's Determination", True, 6),
    "c1e081044abea73c900453a8379a1747a428832afcdf063c42a0b065c85f91e4":
        DrawProgram("me3_76", 1213, "Judge", False, 16),
}


def draw_commands(item):
    program = item["programSha256"]
    domain = DRAW_PROGRAMS[program]
    return [
        {"programSha256": program, "cardSourceID": source_id, "cabtCardId": domain.native_id,
         "mode": mode, "deck": deck, "hand": hand, "enemyDeck": enemy_deck,
         "enemyHand": enemy_hand, "prizes": prizes, "firstTurn": first, "seed": 17}
        for source_id in item["cardSourceIDs"]
        for mode, deck, hand, enemy_deck, enemy_hand, prizes, first in DRAW_MODES
    ]


def draw_metadata_scenario(item, source_id):
    return {"suite": "draw-supporters-metadata/1", "commands": [
        {"operation": "nativeDrawMetadata", "programSha256": item["programSha256"],
         "cardSourceID": source_id, "cardSha256": item["cardSha256"][source_id]}]}


def draw_scenario(item, facet):
    if facet not in DRAW_FACETS:
        raise ValueError("draw Supporters admit only the four measured first-action facets")
    return {"suite": f"draw-supporters-{facet}/1", "commands": draw_commands(item)}


def _source_programs(artifacts, candidate, cards, source_rows):
    raw = artifacts["drawManifest"].read_bytes()
    if sha256(raw).hexdigest() != DRAW_MANIFEST_SHA256:
        raise ValueError("draw manifest differs from the frozen two-program native matrix")
    if any(candidate["sourceSha256"].get(path) not in {expected, EXPANDED_REVIEWED_SOURCES.get(path, expected)}
           for path, expected in REVIEWED_SOURCES.items()):
        raise ValueError("draw candidate differs from the reviewed native fixture and Lillie correction")
    manifest = json.loads(raw)
    if (manifest["schema"] != "draw-supporters-inputs/1"
            or manifest["corpusSha256"] != LIVE_PINS.corpus_sha256
            or manifest["denominator"] != {"rows": 22945, "programs": 9001, "requiredCases": 93514}
            or len(manifest["selected"]) != 2):
        raise ValueError("draw manifest differs from the pinned source domain")
    by_source, program_sources = {}, {}
    for card, _, row_hash, program in source_rows:
        source_id = card["cardSourceID"]
        if source_id in by_source:
            raise ValueError("draw corpus contains duplicate source identities")
        by_source[source_id] = (card, row_hash, program)
        program_sources.setdefault(program, set()).add(source_id)
    items, row_items = {}, {}
    for item in manifest["selected"]:
        program = item["programSha256"]
        domain = DRAW_PROGRAMS.get(program)
        source_ids = item["cardSourceIDs"]
        if (domain is None or program in items or type(item["cabtCardId"]) is not int
                or item["cabtCardId"] != domain.native_id or domain.native_id not in cards
                or item["sourceID"] != domain.source_id or type(item["actionIndex"]) is not int
                or item["actionIndex"] != 0 or not isinstance(source_ids, list)
                or len(source_ids) != domain.rows or len(set(source_ids)) != len(source_ids)
                or set(source_ids) != program_sources.get(program)
                or set(item["cardSha256"]) != set(source_ids)):
            raise ValueError("draw manifest contains unsupported programs, printings, or native identities")
        for source_id in source_ids:
            card, row_hash, source_program = by_source[source_id]
            actions = card.get("useActions", [])
            if (source_id in row_items or source_program != program
                    or digest({k: v for k, v in card.items() if k not in {"cardSourceID", "releaseDate"}}) != program
                    or row_hash != item["cardSha256"][source_id]
                    or card.get("cardName") != domain.name or card.get("cardFormat") != 5
                    or card.get("cardStage") != 0 or card.get("cardProperties") != ([25] if domain.lillie else [0])
                    or len(actions) != 1 or actions[0].get("conditions", []) != item["actionConditions"]
                    or [a.get("conditions", []) for a in actions[0].get("subActions", [])] != item["effectLocalConditions"]
                    or any(a.get("$type") != "MatchLogic.MoveCards, MatchLogic"
                           for a in actions[0].get("subActions", []))):
                raise ValueError("draw printing differs from its full reviewed program and raw row")
            row_items[source_id] = item
        items[program] = item
    if (set(items) != set(DRAW_PROGRAMS) or len(row_items) != 22
            or manifest["commands"] != [command for item in items.values() for command in draw_commands(item)]):
        raise ValueError("draw family inventory differs from all 308 commands and 22 printings")
    return items, row_items


def _known_cards(value):
    return isinstance(value, list) and all(
        isinstance(card, dict) and type(card.get("id")) is int and card["id"] > 0 for card in value)


def draw_response(row, command):
    fields = set(RESPONSE_FIELDS) | set(ZONE_FIELDS) | set(VIEW_FIELDS) | {"prompt", "rawZones", "traitConsumerLillie"}
    if (not isinstance(row, dict) or set(row) != fields
            or any(type(row[key]) is not kind for key, kind in RESPONSE_FIELDS.items())
            or type(row["traitConsumerLillie"]) is not bool
            or any(not isinstance(row[key], list) or any(type(value) is not str for value in row[key])
                   or row[key] != sorted(row[key]) for key in ZONE_FIELDS)
            or row["prompted"] or row["prompt"] is not None):
        raise ValueError("draw fixture differs from the measured typed prompt-free response shape")
    domain = DRAW_PROGRAMS[command["programSha256"]]
    if row["traitConsumerLillie"] is not domain.lillie:
        raise ValueError("native Lillie trait consumer differs from the exact source property")
    raw = row["rawZones"]
    if (not isinstance(raw, dict) or set(raw) != set(RAW_ZONES)
            or any(not isinstance(raw[key], list) or any(type(value) is not str for value in raw[key])
                   or len(raw[key]) != row[key] for key in RAW_ZONES)
            or sorted(raw["hand"] + raw["deck"]) != row["ownUnion"]
            or sorted(raw["enemyHand"] + raw["enemyDeck"]) != row["enemyUnion"]):
        raise ValueError("draw raw zone inventories differ from the conserved native projection")
    owner, opponent, enemy_owner, enemy_opponent = (row[key] for key in VIEW_FIELDS)
    required_view = {"active", "bench", "discard", "hand", "deckCount", "handCount", "prize"}
    for view, hand, deck, discard in ((owner, "hand", "deck", "discard"), (opponent, "hand", "deck", "discard"),
                                      (enemy_owner, "enemyHand", "enemyDeck", "enemyDiscard"),
                                      (enemy_opponent, "enemyHand", "enemyDeck", "enemyDiscard")):
        if (not isinstance(view, dict) or not required_view <= view.keys()
                or type(view["deckCount"]) is not int or view["deckCount"] != row[deck]
                or type(view["handCount"]) is not int or view["handCount"] != row[hand]
                or any(not isinstance(view[key], list) for key in ("active", "bench", "discard", "prize"))
                or len(view["active"]) != 1 or view["bench"] or len(view["discard"]) != len(row[discard])):
            raise ValueError("draw fixture omitted actual private views or changed their zone inventories")
    privacy = {
        "ownerHandKnown": _known_cards(owner["hand"]) and len(owner["hand"]) == row["hand"],
        "opponentHandHidden": opponent["hand"] is None,
        "enemyHandHidden": enemy_opponent["hand"] is None,
        "enemyOwnHandKnown": _known_cards(enemy_owner["hand"]) and len(enemy_owner["hand"]) == row["enemyHand"],
        "deckHiddenFromBoth": all("deck" not in view for view in (owner, opponent, enemy_owner, enemy_opponent)),
        "discardPublic": _known_cards(opponent["discard"]) and _known_cards(enemy_opponent["discard"]),
    }
    total = command["deck"] + command["hand"]
    enemy_total = command["enemyDeck"] + command["enemyHand"]
    playable = not command["firstTurn"] and (total + (enemy_total if not domain.lillie else 0)) > 0
    draw = (8 if command["prizes"] == 6 else 6) if domain.lillie else 4
    expected = {
        "canPlay": playable, "completed": playable, "turnEnded": False, "gameOver": False,
        "hand": min(draw, total) if playable else command["hand"] + 1,
        "deck": max(0, total - draw) if playable else command["deck"],
        "enemyHand": min(4, enemy_total) if playable and not domain.lillie else command["enemyHand"],
        "enemyDeck": max(0, enemy_total - 4) if playable and not domain.lillie else command["enemyDeck"],
        "discard": ["played"] if playable else [], "enemyDiscard": [],
    }
    if playable:
        expected["secondSupporter"] = False
    own_labels = [f"deck{index}" for index in range(command["deck"])] + [f"hand{index}" for index in range(command["hand"])]
    enemy_labels = [f"enemy-deck{index}" for index in range(command["enemyDeck"])] + [f"enemy-hand{index}" for index in range(command["enemyHand"])]
    if not playable:
        own_labels.append("played")
    expected["ownUnion"], expected["enemyUnion"] = sorted(own_labels), sorted(enemy_labels)
    if any(row[key] != value for key, value in expected.items()) or not all(privacy.values()):
        raise ValueError("draw observations violate the measured draw, discard, legality, conservation, or privacy boundaries")
    hand_ids = Counter(domain.native_id if label == "played" else 9990302 for label in raw["hand"])
    if (Counter(card["id"] for card in owner["hand"]) != hand_ids
            or any(card["id"] != 9990302 for card in enemy_owner["hand"])
            or any(card["id"] != domain.native_id for card in opponent["discard"])
            or any(view["active"][0].get("id") != 9990301 for view in (owner, opponent, enemy_owner, enemy_opponent))):
        raise ValueError("draw visible card identities differ from the measured source and synthetic marker IDs")
    return {**{key: row[key] for key in (*RESPONSE_FIELDS, *ZONE_FIELDS)}, "privacy": privacy}


def draw_metadata(item, source_id, card, native):
    domain = DRAW_PROGRAMS[item["programSha256"]]
    if (card["cardId"] != domain.native_id or card["name"] != domain.name
            or card["cardType"] != 3 or card["evolutionType"] != 0 or card["attacks"]
            or len(card["skills"]) != 1 or card["skills"][0]["name"] != domain.name
            or native["lillie"] is not domain.lillie or native["canPlayFirstTurn"] is not False):
        raise ValueError("draw native metadata differs from the reviewed Supporter identity")
    return {"programSha256": item["programSha256"], "cardSourceID": source_id,
            "name": card["name"], "format": 5, "stage": card["evolutionType"],
            "properties": [25] if native["lillie"] else [0],
            "actions": [{"name": skill["name"], "energy": ""} for skill in card["skills"]]}


def run_draw_family(cases, artifacts, candidate, cards, source_rows):
    if not {"drawManifest", "drawFixture", "library"} <= artifacts.keys():
        raise ValueError("draw cases require a pinned manifest, native fixture, and candidate DLL")
    items, row_items = _source_programs(artifacts, candidate, cards, source_rows)
    selected_cases, seen = [], set()
    for case in cases:
        parts = case["caseId"].split("/")
        if len(parts) == 3 and parts[0] == "row" and parts[2] == "metadata_identity" and parts[1] in row_items:
            item = row_items[parts[1]]
            scenario = draw_metadata_scenario(item, parts[1])
        elif (len(parts) == 5 and parts[0] == "program" and parts[1] in items
              and parts[2:4] == ["useActions", "0"] and parts[4] in DRAW_FACETS):
            item = items[parts[1]]
            scenario = draw_scenario(item, parts[4])
        else:
            raise ValueError("draw fixture cannot admit rules, second actions, or unsupported cases")
        if (case["caseId"] in seen or case["scenario"] != scenario
                or type(case["cabtCardId"]) is not int or case["cabtCardId"] != item["cabtCardId"]):
            raise ValueError("draw case differs from its exact source identity and command matrix")
        selected_cases.append((case, item, scenario, parts))
        seen.add(case["caseId"])
    fixture = artifacts["drawFixture"]

    def execute(*arguments):
        return json.loads(subprocess.run([str(fixture), *arguments], capture_output=True, text=True,
                                         encoding="utf-8", timeout=60, check=True).stdout)

    registry = execute("registry")
    library = ctypes.CDLL(str(artifacts["library"]))
    library.AllCard.argtypes = []
    library.AllCard.restype = ctypes.c_char_p
    library.AllAttack.argtypes = []
    library.AllAttack.restype = ctypes.c_char_p
    cards_raw, attacks_raw = library.AllCard(), library.AllAttack()
    if (not cards_raw or not attacks_raw or json.loads(cards_raw) != list(cards.values())
            or registry != {"entries": json.loads(cards_raw), "attacks": json.loads(attacks_raw)}):
        raise ValueError("draw fixture catalogs differ from the actual candidate DLL")
    raw_metadata = execute("metadata")
    if (not isinstance(raw_metadata, list) or len(raw_metadata) != 2
            or any(not isinstance(row, dict) or set(row) != {"cardId", "lillie", "canPlayFirstTurn", "effects"}
                   or type(row["cardId"]) is not int or type(row["lillie"]) is not bool
                   or type(row["canPlayFirstTurn"]) is not bool or not isinstance(row["effects"], list)
                   for row in raw_metadata)):
        raise ValueError("draw fixture omitted actual native metadata and effect layouts")
    native_metadata = {row["cardId"]: row for row in raw_metadata}
    if set(native_metadata) != {domain.native_id for domain in DRAW_PROGRAMS.values()}:
        raise ValueError("draw native metadata contains duplicate or unsupported identities")
    if any(len(row["effects"]) != (7 if row["cardId"] == 1227 else 5) for row in raw_metadata):
        raise ValueError("draw native action and effect-local condition layout changed")
    metadata = {source_id: draw_metadata(item, source_id, cards[item["cabtCardId"]], native_metadata[item["cabtCardId"]])
                for source_id, item in row_items.items()}
    commands = {(command["programSha256"], command["cardSourceID"], command["mode"]): command
                for item in items.values() for command in draw_commands(item)}
    raw = execute()
    if not isinstance(raw, list):
        raise ValueError("draw fixture must execute actual native observation rows")
    responses = {}
    for row in raw:
        key = (row["programSha256"], row["cardSourceID"], row["mode"])
        if key not in commands or key in responses:
            raise ValueError("draw fixture contains unknown or duplicate command identities")
        responses[key] = draw_response(row, commands[key])
    if responses.keys() != commands.keys() or len(responses) != 308 or len(metadata) != 22:
        raise ValueError("draw native inventory differs from all 308 commands and 22 printings")
    observations = []
    for case, item, scenario, parts in selected_cases:
        selected = ([metadata[parts[1]]] if parts[0] == "row" else
                    [responses[(item["programSha256"], command["cardSourceID"], command["mode"])]
                     for command in scenario["commands"]])
        observations.append({"caseId": case["caseId"], "cabtCardId": item["cabtCardId"],
                             "scenarioSha256": digest(scenario),
                             "observation": {"commands": scenario["commands"], "responses": selected},
                             "scenarioComponents": [{"suite": scenario["suite"], "scenarioSha256": digest(scenario)}]})
    return observations
