import ctypes
from dataclasses import dataclass
from hashlib import sha256
import json
import subprocess

from tcg.live_parity import LIVE_PINS, digest


DRAGAPULT_FACETS = ("effect", "legal_prompts", "information_boundaries", "interactions")
DRAGAPULT_MANIFEST_SHA256 = "dfa95d41da869e8946e8423e1db3f7bc8880f7342fcd330b077180a73057a030"
DRAGAPULT_WORKER_PROVENANCE = {
    "receiptSha256": "8c6ef4b202f445e4041002d8b861bf1f8121c9caa3149bacd8f2e78def53c6e6",
    "oracleSha256": "5e54486d2bdb990dc290644b5aa452b7318d8651fbf9e26e40cbb19e7f3c2c2e",
    "workerBoundarySha256": "e2c7cdcb70636a9cc475582ea1a4a28d935a30232520311460a295c40f9ebbb6",
    "workerSha256": "d471d0a1d7c05d23f34c555499ea21f299e306e19a8c3536222bae0af34c25c4",
    "librarySha256": "01e936b0ed9d886e7d3f9b8405653be81ba2371a0f33b5638fe134c8fc7154f5",
    "prompts": 245, "allocationPrompts": 60, "minimumCounters": 6, "maximumCounters": 6,
    "libraryValidTotals": [0, 5, 6], "workerValidTotals": [6], "workerRejectedTotals": [0, 5, 7],
    "nativeProtocol": "six ordered single-target counter placements",
    "sourceProtocol": "one global allocation validated by the pinned worker facade",
}
REVIEWED_SOURCES = {
    "CardImpl.h": "286a51820d36f9b60b5b13c58bc2edff352eb4050581ddadf1960f13fd6f21a9",
    "TargetList.h": "c93bea62ea3de7b733e3cbbc8d75cde9a612c3dd956516597a8fe8858dc8eac7",
    "experiments/ranked_dragapult/cabt.cpp": "23c64af0948f056eb2028a0aed715ca707ef51da6ff6211b1ff2028997f1c1ea",
    "experiments/ranked_dragapult/Commands.h": "221bb2a67ec899e3dbe40b93f3cd5535357745b0ccc80a4e43f4253c43159040",
}
EXPANDED_REVIEWED_SOURCES = {
    "TargetList.h": "7e48c15e8568fbdd32d8d32257b00e16c57709832bddf8761b0e7321c2965198",
    "experiments/ranked_dragapult/cabt.cpp": "11063e3fc273aa562f806509f8852f8186c3c45b4c93d9967a44fd4ee2416666",
}
ABILITY_MODES = ("regular-top", "regular-second", "one", "empty", "two-second", "bench",
                 "first-turn", "repeat", "second-copy", "reset-next", "three-top", "three-second")
ATTACK_MODES = ("sufficient", "none", "short", "wrong", "first-turn", "bench-actor",
                "weakness", "resistance", "active-ko", "last-prize")
COUNTER_MODES = ("all-one", "no-bench", "split-three", "alternating", "reverse-split",
                 "bench-ko", "tera-bench", "pre-damage")
SCALAR_FIELDS = {
    "programSha256": str, "cardSourceID": str, "actionIndex": int, "mode": str,
    "legal": bool, "completed": bool, "currentPlayer": int, "gameOver": bool,
    "actorDamage": int, "defenderDamage": int, "actorDiscarded": bool, "defenderDiscarded": bool,
    "prizes": int, "enemyPrizes": int, "secondLegal": bool, "otherCopyLegal": bool, "nextTurnLegal": bool,
}
LABEL_FIELDS = ("hand", "deck", "enemyDeck", "discard", "enemyDiscard", "decisionOptions")


@dataclass(frozen=True)
class DragapultProgram:
    source_id: str
    native_id: int
    name: str
    rows: int
    native_actions: tuple


DRAGAPULT_PROGRAMS = {
    "f75b9a3dbc8d0b72081ba6428028c17e4cd04efc796e61bd84c7bbebffd08e43":
        DragapultProgram("sv6_129", 120, "Drakloak", 10, (46, 152)),
    "a6f0f8032c2f3e386befbed4b79aa167493199a2c9c52128d30f7464fe324cef":
        DragapultProgram("sv6_130", 121, "Dragapult ex", 5, (153, 154)),
}


def dragapult_commands(item, action_index=None):
    program = item["programSha256"]
    domain = DRAGAPULT_PROGRAMS[program]
    if action_index is not None and (type(action_index) is not int or action_index not in (0, 1)):
        raise ValueError("Dragapult admits only the two printed useActions indices")
    commands = []
    for source_id in item["cardSourceIDs"]:
        for index in range(2):
            if action_index is not None and index != action_index:
                continue
            ability = domain.native_id == 120 and index == 0
            counter = domain.native_id == 121 and index == 1
            modes = ABILITY_MODES if ability else ATTACK_MODES + (COUNTER_MODES if counter else ())
            for mode in modes:
                deck = {"empty": 0, "one": 1, "two-second": 2, "reset-next": 8}.get(mode, 3 if mode.startswith("three-") else 4)
                assignments = [] if not counter or mode == "no-bench" else [0, 0, 0, 1, 1, 1]
                if mode in ("all-one", "bench-ko", "tera-bench", "pre-damage"):
                    assignments = [0] * 6
                elif mode == "alternating":
                    assignments = [0, 1] * 3
                elif mode == "reverse-split":
                    assignments = [1, 1, 1, 0, 0, 0]
                commands.append({"programSha256": program, "cardSourceID": source_id, "cabtCardId": domain.native_id,
                                 "actionIndex": index, "nativeActionId": domain.native_actions[index], "ability": ability,
                                 "mode": mode, "deck": deck, "prizes": 1 if mode == "last-prize" else 3,
                                 "choice": 1 if mode in ("regular-second", "two-second", "three-second", "bench") else 0,
                                 "assignments": assignments, "seed": 17})
    return commands


def dragapult_metadata_scenario(item, source_id):
    return {"suite": "ranked-dragapult-metadata/1", "commands": [
        {"operation": "nativeDragapultMetadata", "programSha256": item["programSha256"],
         "cardSourceID": source_id, "cardSha256": item["cardSha256"][source_id]}]}


def dragapult_scenario(item, channel, action_index, facet):
    if channel != "useActions" or facet not in DRAGAPULT_FACETS:
        raise ValueError("Dragapult admits only measured printed useActions facets")
    return {"suite": f"ranked-dragapult-{channel}-{action_index}-{facet}/1",
            "commands": dragapult_commands(item, action_index)}


def _source_programs(artifacts, candidate, cards, source_rows):
    raw = artifacts["dragapultManifest"].read_bytes()
    if sha256(raw).hexdigest() != DRAGAPULT_MANIFEST_SHA256:
        raise ValueError("Dragapult manifest differs from the frozen two-program matrix")
    if any(candidate["sourceSha256"].get(path) not in {expected, EXPANDED_REVIEWED_SOURCES.get(path, expected)}
           for path, expected in REVIEWED_SOURCES.items()):
        raise ValueError("Dragapult candidate differs from the reviewed native mechanics and fixture")
    manifest = json.loads(raw)
    if (manifest["schema"] != "ranked-dragapult-inputs/1" or manifest["corpusSha256"] != LIVE_PINS.corpus_sha256
            or manifest["denominator"] != {"rows": 22945, "programs": 9001, "requiredCases": 93514}
            or len(manifest["selected"]) != 2):
        raise ValueError("Dragapult manifest differs from the pinned source domain")
    by_source, program_sources = {}, {}
    for card, _, row_hash, program in source_rows:
        source_id = card["cardSourceID"]
        if source_id in by_source:
            raise ValueError("Dragapult corpus contains duplicate source identities")
        by_source[source_id] = (card, row_hash, program)
        program_sources.setdefault(program, set()).add(source_id)
    items, row_items = {}, {}
    for item in manifest["selected"]:
        program = item["programSha256"]
        domain = DRAGAPULT_PROGRAMS.get(program)
        source_ids = item["cardSourceIDs"]
        if (domain is None or program in items or type(item["cabtCardId"]) is not int
                or item["cabtCardId"] != domain.native_id or domain.native_id not in cards
                or item["sourceID"] != domain.source_id or len(source_ids) != domain.rows
                or len(set(source_ids)) != len(source_ids) or set(source_ids) != program_sources.get(program)
                or set(item["cardSha256"]) != set(source_ids)):
            raise ValueError("Dragapult manifest contains unsupported programs, printings, or native identities")
        for source_id in source_ids:
            card, row_hash, source_program = by_source[source_id]
            if (source_id in row_items or source_program != program or row_hash != item["cardSha256"][source_id]
                    or digest({k: v for k, v in card.items() if k not in {"cardSourceID", "releaseDate"}}) != program
                    or card.get("cardName") != domain.name or card.get("cardFormat", 3) != 3
                    or card.get("useActions") != item["actions"] or len(item["actions"]) != 2
                    or card.get("passiveActions", []) or card.get("triggeredActions", [])
                    or [action.get("actionType") for action in item["actions"]] != ([2, 1] if domain.native_id == 120 else [1, 1])):
                raise ValueError("Dragapult printing differs from its complete program, raw row, or printed channels")
            row_items[source_id] = item
        items[program] = item
    if (set(items) != set(DRAGAPULT_PROGRAMS) or len(row_items) != 15
            or manifest["commands"] != [command for item in items.values() for command in dragapult_commands(item)]):
        raise ValueError("Dragapult inventory differs from all 360 commands and 15 printings")
    return items, row_items


def _known_cards(value):
    return isinstance(value, list) and all(isinstance(card, dict) and type(card.get("id")) is int and card["id"] > 0 for card in value)


def _views(owner, opponent):
    if any(not isinstance(view, dict) or not isinstance(view.get("players"), list) or len(view["players"]) != 2 for view in (owner, opponent)):
        raise ValueError("Dragapult fixture omitted actual private seat views")
    for seat in range(2):
        visible, hidden = (owner, opponent) if seat == 0 else (opponent, owner)
        own, other = visible["players"][seat], hidden["players"][seat]
        if (not _known_cards(own.get("hand")) or other.get("hand") is not None
                or type(own.get("handCount")) is not int or len(own["hand"]) != own["handCount"]
                or own["handCount"] != other.get("handCount") or type(own.get("deckCount")) is not int
                or own["deckCount"] != other.get("deckCount") or "deck" in own or "deck" in other
                or any(not _known_cards(own.get(zone)) or own[zone] != other.get(zone) for zone in ("active", "bench", "discard"))):
            raise ValueError("Dragapult native views violate actual hidden hands, decks, or public board identities")
    if opponent.get("looking") is not None:
        raise ValueError("Dragapult native view exposed privately inspected deck cards")


def dragapult_response(row, command):
    fields = set(SCALAR_FIELDS) | set(LABEL_FIELDS) | {"bench", "allowedCounts", "trace", "ownerView", "opponentView"}
    if (not isinstance(row, dict) or set(row) != fields
            or any(type(row[key]) is not kind for key, kind in SCALAR_FIELDS.items())
            or any(not isinstance(row[key], list) or any(type(label) is not str for label in row[key]) for key in LABEL_FIELDS)
            or not isinstance(row["trace"], list) or not isinstance(row["bench"], list)
            or not isinstance(row["allowedCounts"], list) or any(type(v) is not bool for v in row["allowedCounts"])):
        raise ValueError("Dragapult fixture changed its measured typed response shape")
    for key in ("programSha256", "cardSourceID", "actionIndex", "mode"):
        if row[key] != command[key]:
            raise ValueError("Dragapult response changed its command source identity")
    _views(row["ownerView"], row["opponentView"])
    counter_steps = []
    for step in row["trace"]:
        if (not isinstance(step, dict) or set(step) != {"phase", "prompt", "options", "selected", "validCounts", "valid", "ownerView", "opponentView"}
                or step["valid"] is not True or not isinstance(step["options"], list)
                or any(type(label) is not str for label in step["options"])
                or not isinstance(step["selected"], list) or any(label not in step["options"] for label in step["selected"])
                or not isinstance(step["validCounts"], list) or any(type(v) is not bool for v in step["validCounts"])):
            raise ValueError("Dragapult fixture omitted actual typed native prompts and selections")
        prompt = step["prompt"]
        if (not isinstance(prompt, dict) or type(prompt.get("minCount")) is not int or type(prompt.get("maxCount")) is not int
                or not prompt["minCount"] <= len(step["selected"]) <= prompt["maxCount"]
                or len(prompt.get("option", [])) != len(step["options"])):
            raise ValueError("Dragapult native selection violates its declared prompt constraints")
        _views(step["ownerView"], step["opponentView"])
        if prompt["context"] == 14:
            counter_steps.append(step)
            if (prompt["type"] != 1 or prompt["minCount"] != 1 or prompt["maxCount"] != 1
                    or step["options"] != ["bench0", "bench1"] or step["validCounts"] != [False, True, False]
                    or any(option.get("area") != 5 or option.get("playerIndex") != 1 for option in prompt["option"])):
                raise ValueError("Phantom Dive native counter decisions changed their exact bench-only single-target protocol")
    ability, mode = command["ability"], command["mode"]
    legal = mode != "empty" if ability else mode not in ("none", "short", "first-turn", "bench-actor") and not (mode == "wrong" and command["nativeActionId"] != 153)
    expected_steps = command["assignments"] if legal else []
    if ([step["prompt"]["remainDamageCounter"] for step in counter_steps] != list(range(len(expected_steps), 0, -1))
            or [step["selected"] for step in counter_steps] != [[f"bench{target}"] for target in expected_steps]
            or row["legal"] != legal or row["completed"] != legal or row["secondLegal"]
            or row["otherCopyLegal"] != (mode == "second-copy") or row["nextTurnLegal"] != (mode == "reset-next")):
        raise ValueError("Dragapult native legality, cooldown, or ordered six-counter placement changed")
    for target in row["bench"]:
        if (not isinstance(target, dict) or set(target) != {"label", "damage", "discarded"}
                or type(target["label"]) is not str or type(target["damage"]) is not int or type(target["discarded"]) is not bool):
            raise ValueError("Dragapult native target projection changed")
    if ability:
        deck, hand = [f"deck{i}" for i in range(command["deck"])], ["guard"]
        if legal:
            hand.append(deck.pop(command["choice"]))
            if deck and command["deck"] > 1:
                deck.append(deck.pop(0))
        if mode == "second-copy":
            hand.append(deck.pop(0))
            deck.append(deck.pop(0))
        elif mode == "reset-next":
            hand.append(deck.pop(0))
            hand.append(deck.pop(0))
            deck.append(deck.pop(0))
        if (row["deck"] != deck or row["hand"] != sorted(hand)
                or row["decisionOptions"] != (["deck0", "deck1"] if command["deck"] > 1 else [])
                or row["allowedCounts"] != ([False, True, False] if command["deck"] > 1 else [])):
            raise ValueError("Recon Directive changed inspected choices, bottom-return order, or drawn identities")
    owner, opponent = row["ownerView"], row["opponentView"]
    one, one_hidden = owner["players"][0], opponent["players"][0]
    if len(one["hand"]) != len(row["hand"]) or one["deckCount"] != len(row["deck"]) or len(one["discard"]) != len(row["discard"]):
        raise ValueError("Dragapult actual final views disagree with the observed card movement")
    privacy = {"ownerHandKnown": _known_cards(one["hand"]), "opponentHandHidden": one_hidden["hand"] is None,
               "deckHiddenFromBoth": all("deck" not in view for view in owner["players"] + opponent["players"]),
               "activePublic": bool(one["active"]) and bool(one_hidden["active"]) and _known_cards(one_hidden["active"]),
               "benchPublic": _known_cards(opponent["players"][1]["bench"])}
    if not all(privacy.values()):
        raise ValueError("Dragapult native final privacy boundary changed")
    return {**{key: row[key] for key in (*SCALAR_FIELDS, *LABEL_FIELDS, "bench", "allowedCounts")}, "privacy": privacy}


def dragapult_metadata(item, source_id, card, attacks):
    domain = DRAGAPULT_PROGRAMS[item["programSha256"]]
    if (card["cardId"] != domain.native_id or card["name"] != domain.name or card["cardType"] != 0
            or card["attacks"] != list(domain.native_actions[1:] if domain.native_id == 120 else domain.native_actions)
            or [skill["name"] for skill in card["skills"]] != (["Recon Directive"] if domain.native_id == 120 else [])):
        raise ValueError("Dragapult native metadata differs from its reviewed Card and Attack identities")
    types = {0: 1, 1: 3, 2: 2, 3: 4, 4: 8, 5: 11, 6: 5, 7: 7, 8: 10, 9: 9}
    actions = [{"name": "[Ability] " + s["name"], "energy": "", "type": 2} for s in card["skills"]]
    actions += [{"name": attacks[a]["name"], "energy": "".join("{" + "CGRWLPFDMN"[e] + "}" for e in attacks[a]["energies"]), "type": 1} for a in card["attacks"]]
    return {"programSha256": item["programSha256"], "cardSourceID": source_id, "name": card["name"], "format": 3,
            "stage": {1: 0, 2: 2, 3: 3}[card["evolutionType"]], "hp": card["hp"], "retreat": card["retreatCost"],
            "typing": [types[card["energyType"]]], "properties": ([19, 20] if card["tera"] else [19]) if card["ex"] else [0],
            "prizeValue": 2 if card["ex"] else 1, "weakness": [types[card["weakness"]]] if card["weakness"] is not None else [],
            "resistance": [types[card["resistance"]]] if card["resistance"] is not None else [], "actions": actions}


def run_dragapult_family(cases, artifacts, candidate, cards, source_rows):
    if not {"dragapultManifest", "dragapultFixture", "library"} <= artifacts.keys():
        raise ValueError("Dragapult cases require a pinned manifest, native fixture, and candidate DLL")
    items, row_items = _source_programs(artifacts, candidate, cards, source_rows)
    selected_cases, seen = [], set()
    for case in cases:
        parts = case["caseId"].split("/")
        if len(parts) == 3 and parts[0] == "row" and parts[2] == "metadata_identity" and parts[1] in row_items:
            item = row_items[parts[1]]
            scenario = dragapult_metadata_scenario(item, parts[1])
        elif (len(parts) == 5 and parts[0] == "program" and parts[1] in items and parts[2] == "useActions"
              and parts[3] in ("0", "1") and parts[4] in DRAGAPULT_FACETS):
            item = items[parts[1]]
            scenario = dragapult_scenario(item, parts[2], int(parts[3]), parts[4])
        else:
            raise ValueError("Dragapult fixture cannot admit rules, passive channels, or unsupported cases")
        if (case["caseId"] in seen or case["scenario"] != scenario
                or type(case["cabtCardId"]) is not int or case["cabtCardId"] != item["cabtCardId"]):
            raise ValueError("Dragapult case differs from its exact source identity and command matrix")
        selected_cases.append((case, item, scenario, parts))
        seen.add(case["caseId"])

    def execute(*arguments):
        return json.loads(subprocess.run([str(artifacts["dragapultFixture"]), *arguments], capture_output=True,
                                         text=True, encoding="utf-8", timeout=60, check=True).stdout)

    registry = execute("registry")
    library = ctypes.CDLL(str(artifacts["library"]))
    library.AllCard.argtypes, library.AllCard.restype = [], ctypes.c_char_p
    library.AllAttack.argtypes, library.AllAttack.restype = [], ctypes.c_char_p
    cards_raw, attacks_raw = library.AllCard(), library.AllAttack()
    if (not cards_raw or not attacks_raw or json.loads(cards_raw) != list(cards.values())
            or registry != {"entries": json.loads(cards_raw), "attacks": json.loads(attacks_raw)}):
        raise ValueError("Dragapult fixture catalog differs from the actual initialized candidate DLL")
    attacks = {row["attackId"]: row for row in registry["attacks"]}
    metadata = {source_id: dragapult_metadata(item, source_id, cards[item["cabtCardId"]], attacks) for source_id, item in row_items.items()}
    commands = {(c["programSha256"], c["cardSourceID"], c["actionIndex"], c["mode"]): c for item in items.values() for c in dragapult_commands(item)}
    raw = execute()
    if not isinstance(raw, list):
        raise ValueError("Dragapult fixture must execute actual native observation rows")
    responses = {}
    for row in raw:
        key = (row["programSha256"], row["cardSourceID"], row["actionIndex"], row["mode"])
        if key not in commands or key in responses:
            raise ValueError("Dragapult fixture contains unknown or duplicate command identities")
        responses[key] = dragapult_response(row, commands[key])
    if responses.keys() != commands.keys() or len(responses) != 360 or len(metadata) != 15:
        raise ValueError("Dragapult native inventory differs from all 360 commands and 15 printings")
    observations = []
    for case, item, scenario, parts in selected_cases:
        selected = ([metadata[parts[1]]] if parts[0] == "row" else
                    [responses[(c["programSha256"], c["cardSourceID"], c["actionIndex"], c["mode"])] for c in scenario["commands"]])
        observations.append({"caseId": case["caseId"], "cabtCardId": item["cabtCardId"], "scenarioSha256": digest(scenario),
                             "observation": {"commands": scenario["commands"], "responses": selected},
                             "scenarioComponents": [{"suite": scenario["suite"], "scenarioSha256": digest(scenario)}]})
    return observations
