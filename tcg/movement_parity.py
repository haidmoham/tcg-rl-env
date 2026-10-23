import ctypes
from dataclasses import dataclass
from hashlib import sha256
import json
import subprocess

from tcg.live_parity import digest


MOVEMENT_FACETS = ("effect", "legal_prompts", "information_boundaries", "interactions")
MOVEMENT_MANIFEST_SHA256 = "a4645739d8ca6b2963711eebea8645015a53ec3329a0cc764924e0f1a4cd0e64"
MOVEMENT_MODES = ("empty", "no-eligible", "one", "mixed-one", "mixed-zero", "mixed-max",
                  "full-bench", "one-slot", "reverse-max")
BACKGROUND_KINDS = ("basic70", "basic80", "stage1_70", "ex70", "item", "tool", "supporter",
                    "basicEnergy", "specialEnergy")
RESPONSE_FIELDS = {"programSha256": str, "mode": str, "canPlay": bool, "prompted": bool,
                   "selectedValid": bool, "completed": bool, "turnEnded": bool, "gameOver": bool}
ZONE_FIELDS = ("hand", "deck", "discard", "bench", "optionLabels")
NATIVE_TYPES = {0: 1, 1: 3, 2: 2, 3: 4, 4: 8, 5: 11, 6: 5, 7: 7, 8: 10, 9: 9}
NATIVE_FORMATS = {0: 3, 1: 4, 2: 6, 3: 5, 4: 7, 5: 1, 6: 2}
REVIEWED_SOURCES = {
    "CardImpl.h": "286a51820d36f9b60b5b13c58bc2edff352eb4050581ddadf1960f13fd6f21a9",
    "experiments/movement_family/cabt.cpp": "3d6f85bf172b502f66cc44a907fc09edb56b05602e4995192793122f23b290ef",
    "experiments/movement_family/Commands.h": "8711ca9e4dc370c945e49c4b8a99179dd8cb483303719493c238eb9658e20567",
}
REVIEWED_CORRECTIONS = {
    "ac373ae1117e78d3b29f10a6c5b18f6c9be87684aa79346445c98cd096e9e1af",
    "163955867e50b2d32d37cafa056860fe7911720cdd3c40a3d59f53b2e1a1c396",
}


@dataclass(frozen=True)
class MovementProgram:
    source_id: str
    native_id: int
    attack: bool = False

    @property
    def source_zone(self):
        return "discard" if self.native_id in (1097, 162) else "deck"


MOVEMENT_PROGRAMS = {
    "9bc7a5248a14e46e7f7f88ae892b9d311becb0e030d9a230032ad0eb9a108118": MovementProgram("me3_81", 1152),
    "a427a14217d1534e621a62feaba0a64a3dcb403c7ac9300003d87ec34d602562": MovementProgram("me2-5_196", 1097),
    "7dbd189921a15f5a26638ad837ddf70456e2cd88319e858caf4a3de25fbf24eb": MovementProgram("sv5_144", 1086),
    "65fe2df781f9668885084ce9bc9f04e0e2da81f5586852799d951810eefd2a44": MovementProgram("sv8_170", 1205),
    "5791b8c10735841fb37c73f8e9b54f8a8c825d8f68ca8e74d28404e6f14fc8bb": MovementProgram("sv6_126", 42, True),
    "9bc94ea1cce4846087f8e38ad18ef26e7934542913e07e185764cb3e651612ff": MovementProgram("sv7_57", 162, True),
}


def movement_commands(item):
    program = item["programSha256"]
    domain = MOVEMENT_PROGRAMS[program]
    modes = MOVEMENT_MODES + (("one-energy",) if domain.native_id == 1097 else ())
    commands = []
    for mode in modes:
        if mode == "empty":
            mixture = []
        elif mode == "no-eligible":
            mixture = ["item", "tool", "supporter", "specialEnergy"]
        elif mode == "one-energy":
            mixture = ["basicEnergy"]
        elif mode == "one":
            mixture = ["ex70" if domain.native_id == 1205 else "basic70"]
        else:
            mixture = [*BACKGROUND_KINDS, "basic70", "ex70", "ex70"]
        count = 1
        if mode == "mixed-zero":
            count = 0
        elif mode in ("mixed-max", "reverse-max") and domain.native_id == 1205:
            count = 3
        elif mode in ("mixed-max", "one-slot") or mode == "reverse-max" and domain.native_id == 1086:
            count = 2
        commands.append({"programSha256": program, "cardSourceID": domain.source_id,
                         "cabtCardId": domain.native_id, "mode": mode, "sourceZone": domain.source_zone,
                         "sourceCards": mixture, "selectedCount": count,
                         "benchCount": 5 if mode == "full-bench" else 4 if mode == "one-slot" else 0,
                         "seed": 17})
    return commands


def movement_metadata_scenario(item, source_id):
    return {"suite": "movement-metadata/1", "commands": [
        {"operation": "nativeMovementMetadata", "programSha256": item["programSha256"],
         "cardSourceID": source_id, "cardSha256": item["cardSha256"][source_id]}]}


def movement_scenario(item, facet):
    if facet not in MOVEMENT_FACETS:
        raise ValueError("movement family admits only the measured first-action facets")
    return {"suite": f"movement-family-{facet}/1", "commands": movement_commands(item)}


def _source_programs(artifacts, candidate, cards, source_rows):
    raw = artifacts["movementManifest"].read_bytes()
    if sha256(raw).hexdigest() != MOVEMENT_MANIFEST_SHA256:
        raise ValueError("movement manifest differs from the frozen measured input matrix")
    if (any(candidate["sourceSha256"].get(path) != expected for path, expected in REVIEWED_SOURCES.items())
            or candidate["sourceSha256"].get("RuntimeParityCorrections.h") not in REVIEWED_CORRECTIONS):
        raise ValueError("movement candidate differs from the reviewed native fixture and corrections")
    manifest = json.loads(raw)
    by_source, program_sources = {}, {}
    for card, _, row_hash, program in source_rows:
        source_id = card["cardSourceID"]
        if source_id in by_source:
            raise ValueError("movement corpus contains duplicate source identities")
        by_source[source_id] = (card, row_hash, program)
        program_sources.setdefault(program, set()).add(source_id)
    items, row_items = {}, {}
    for item in manifest["selected"]:
        program = item["programSha256"]
        domain = MOVEMENT_PROGRAMS.get(program)
        if (domain is None or program in items or item["cabtCardId"] != domain.native_id
                or item["sourceID"] != domain.source_id or item["attack"] is not domain.attack
                or type(item["actionIndex"]) is not int or item["actionIndex"] != 0
                or domain.native_id not in cards
                or set(item["cardSourceIDs"]) != program_sources.get(program)
                or set(item["cardSha256"]) != set(item["cardSourceIDs"])):
            raise ValueError("movement manifest contains unsupported source programs or native identities")
        for source_id in item["cardSourceIDs"]:
            card, row_hash, source_program = by_source[source_id]
            actual_program = digest({key: value for key, value in card.items()
                                     if key not in {"cardSourceID", "releaseDate"}})
            actions = card.get("useActions", [])
            if (source_id in row_items or source_program != program or actual_program != program
                    or row_hash != item["cardSha256"][source_id]
                    or not actions or len(actions[0].get("subActions", [])) != 1
                    or actions[0]["subActions"][0].get("$type") != "MatchLogic.MoveCards, MatchLogic"):
                raise ValueError("movement printing differs from its full reviewed source program and raw row")
            row_items[source_id] = item
        items[program] = item
    commands = [command for item in items.values() for command in movement_commands(item)]
    if (set(items) != set(MOVEMENT_PROGRAMS) or len(row_items) != 29 or len(commands) != 55
            or manifest["commands"] != commands):
        raise ValueError("movement family inventory differs from the complete reviewed matrix")
    return items, row_items


def _known_cards(value):
    return (isinstance(value, list)
            and all(isinstance(card, dict) and type(card.get("id")) is int and card["id"] > 0
                    for card in value))


def movement_response(row):
    fields = set(RESPONSE_FIELDS) | set(ZONE_FIELDS) | {"allowedCounts", "prompt", "ownerView", "opponentView"}
    if (not isinstance(row, dict) or set(row) != fields
            or any(type(row[key]) is not kind for key, kind in RESPONSE_FIELDS.items())
            or any(not isinstance(row[key], list) or any(type(value) is not str for value in row[key])
                   for key in ZONE_FIELDS)
            or not isinstance(row["allowedCounts"], list)
            or any(type(value) is not bool for value in row["allowedCounts"])):
        raise ValueError("movement fixture response differs from the reviewed typed observation shape")
    if (any(row[key] != sorted(row[key]) for key in ("hand", "deck", "discard", "optionLabels"))
            or len(set(row["optionLabels"])) != len(row["optionLabels"])):
        raise ValueError("movement unordered zones or prompt identities are not canonical")
    prompt = row["prompt"]
    if row["prompted"]:
        if (not isinstance(prompt, dict) or not isinstance(prompt.get("option"), list)
                or len(prompt["option"]) != len(row["optionLabels"])
                or len(row["allowedCounts"]) != 5
                or type(prompt.get("minCount")) is not int or type(prompt.get("maxCount")) is not int
                or any(not isinstance(option, dict)
                       or set(option) != {"area", "index", "playerIndex", "type"}
                       or any(type(value) is not int for value in option.values())
                       for option in prompt["option"])):
            raise ValueError("movement fixture omitted the actual typed selection prompt")
    elif prompt is not None or row["allowedCounts"] or row["optionLabels"]:
        raise ValueError("movement fixture returned selection evidence without an actual prompt")
    owner, opponent = row["ownerView"], row["opponentView"]
    view_fields = {"active", "bench", "discard", "hand", "deckCount", "handCount", "prize"}
    if (not isinstance(owner, dict) or not isinstance(opponent, dict)
            or not view_fields <= owner.keys() or not view_fields <= opponent.keys()
            or any(not isinstance(view[key], list) for view in (owner, opponent)
                   for key in ("active", "bench", "discard", "prize"))
            or not isinstance(owner["hand"], list)
            or any(type(view[key]) is not int for view in (owner, opponent) for key in ("deckCount", "handCount"))):
        raise ValueError("movement fixture omitted actual private player views")
    if (owner["deckCount"] != len(row["deck"]) or opponent["deckCount"] != len(row["deck"])
            or owner["handCount"] != len(row["hand"]) or opponent["handCount"] != len(row["hand"])
            or any(len(view[key]) != len(row[key]) for view in (owner, opponent) for key in ("bench", "discard"))):
        raise ValueError("movement private views differ from the measured zone inventories")
    return {**{key: row[key] for key in (*RESPONSE_FIELDS, *ZONE_FIELDS, "allowedCounts")}, "privacy": {
        "ownerHandKnown": _known_cards(owner["hand"]),
        "opponentHandHidden": opponent["hand"] is None,
        "deckHiddenFromBoth": "deck" not in owner and "deck" not in opponent,
        "discardPublic": _known_cards(opponent["discard"]),
        "benchPublic": _known_cards(opponent["bench"])}}


def movement_metadata(item, source_id, card, attacks):
    return {"programSha256": item["programSha256"], "cardSourceID": source_id,
            "name": card["name"], "format": NATIVE_FORMATS[card["cardType"]],
            "stage": card["evolutionType"], "hp": card["hp"], "retreat": card["retreatCost"],
            "typing": [NATIVE_TYPES[card["energyType"]]] if card["cardType"] == 0 else [],
            "properties": [0], "prizeValue": 1 if card["pokemonType"] == 1 else 0,
            "weakness": [NATIVE_TYPES[card["weakness"]]] if card["weakness"] is not None else [],
            "resistance": [NATIVE_TYPES[card["resistance"]]] if card["resistance"] is not None else [],
            "weaknessAmount": 2 if card["weakness"] is not None else 0,
            "resistanceAmount": 30 if card["resistance"] is not None else 0,
            "attacks": [{"name": attacks[index]["name"],
                         "energy": "".join("{" + "CGRWLPFDMN"[energy] + "}"
                                           for energy in attacks[index]["energies"])}
                        for index in card["attacks"]] if card["cardType"] == 0
                       else [{"name": skill["name"], "energy": ""} for skill in card["skills"]]}


def run_movement_family(cases, artifacts, candidate, cards, source_rows):
    if not {"movementManifest", "movementFixture", "library"} <= artifacts.keys():
        raise ValueError("movement cases require a pinned manifest, native fixture, and candidate DLL")
    items, row_items = _source_programs(artifacts, candidate, cards, source_rows)
    selected_cases, seen = [], set()
    for case in cases:
        case_id = case["caseId"]
        parts = case_id.split("/")
        if len(parts) == 3 and parts[0] == "row" and parts[2] == "metadata_identity" and parts[1] in row_items:
            item = row_items[parts[1]]
            scenario = movement_metadata_scenario(item, parts[1])
        elif (len(parts) == 5 and parts[0] == "program" and parts[1] in items
              and parts[2:4] == ["useActions", "0"] and parts[4] in MOVEMENT_FACETS):
            item = items[parts[1]]
            scenario = movement_scenario(item, parts[4])
        else:
            raise ValueError("movement fixture cannot admit rules, second actions, or unsupported cases")
        if case_id in seen or case["scenario"] != scenario or case["cabtCardId"] != item["cabtCardId"]:
            raise ValueError("movement case differs from its reviewed source identity and command matrix")
        selected_cases.append((case, item, scenario, parts))
        seen.add(case_id)
    fixture = artifacts["movementFixture"]
    registry = json.loads(subprocess.run([str(fixture), "registry"], capture_output=True, text=True,
                                         encoding="utf-8", timeout=60, check=True).stdout)
    library = ctypes.CDLL(str(artifacts["library"]))
    library.AllCard.argtypes = []
    library.AllCard.restype = ctypes.c_char_p
    library.AllAttack.argtypes = []
    library.AllAttack.restype = ctypes.c_char_p
    cards_raw, attacks_raw = library.AllCard(), library.AllAttack()
    if (not cards_raw or not attacks_raw or json.loads(cards_raw) != list(cards.values())
            or registry != {"entries": json.loads(cards_raw), "attacks": json.loads(attacks_raw)}):
        raise ValueError("movement fixture catalogs differ from the actual candidate DLL")
    attacks = {attack["attackId"]: attack for attack in registry["attacks"]}
    metadata = {source_id: movement_metadata(item, source_id, cards[item["cabtCardId"]], attacks)
                for source_id, item in row_items.items()}
    raw = json.loads(subprocess.run([str(fixture)], capture_output=True, text=True,
                                    encoding="utf-8", timeout=60, check=True).stdout)
    if not isinstance(raw, list):
        raise ValueError("movement fixture must execute and emit actual native observation rows")
    expected_keys = {(program, command["mode"]) for program, item in items.items()
                     for command in movement_commands(item)}
    responses = {}
    for row in raw:
        response = movement_response(row)
        key = (response["programSha256"], response["mode"])
        if key not in expected_keys or key in responses:
            raise ValueError("movement fixture contains unknown or duplicate scenario identities")
        responses[key] = response
    if responses.keys() != expected_keys or len(metadata) != 29:
        raise ValueError("movement observation inventory differs from all 55 commands and 29 printings")
    observations = []
    for case, item, scenario, parts in selected_cases:
        selected = ([metadata[parts[1]]] if parts[0] == "row" else
                    [responses[(item["programSha256"], command["mode"])] for command in scenario["commands"]])
        observations.append({"caseId": case["caseId"], "cabtCardId": item["cabtCardId"],
                             "scenarioSha256": digest(scenario),
                             "observation": {"commands": scenario["commands"], "responses": selected},
                             "scenarioComponents": [{"suite": scenario["suite"], "scenarioSha256": digest(scenario)}]})
    return observations
