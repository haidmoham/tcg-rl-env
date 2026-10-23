import ctypes
from dataclasses import dataclass
import json
import subprocess

from tcg.live_parity import digest


ENERGIES = {"G": (1, 3, "Grass"), "R": (2, 2, "Fire"), "W": (3, 4, "Water"),
            "L": (4, 8, "Lightning"), "P": (5, 11, "Psychic"), "F": (6, 5, "Fighting"),
            "D": (7, 7, "Dark"), "M": (8, 10, "Metal")}
SOURCE_FIELDS = set("cardArchetypeID cardFormat cardName cardProperties cardSourceID cardStage "
                    "cardTyping energySprite energyValues logicSourceID releaseDate".split())
ENERGY_MODES = ("active-hand", "bench-hand", "first-turn", "opponent-active", "opponent-hand",
                "not-in-hand", "second-same-turn", "reset-next-own-turn", "retreat", "retreat-short",
                "retreat-once", "cost-C", "cost-G", "cost-R", "cost-W", "cost-L", "cost-P",
                "cost-F", "cost-D", "cost-M")
RESPONSE_FIELDS = {"programSha256": str, "symbol": str, "mode": str, "legal": bool,
                   "attachmentLegal": bool, "handEnergy": int, "activeEnergy": int, "benchEnergy": int,
                   "discardEnergy": int, "activeValue": str, "benchValue": str, "activeIsOriginal": bool,
                   "damage": int, "currentPlayer": int, "gameOver": bool}
METADATA_FIELDS = {"programSha256", "cardSourceID", "name", "cardFormat", "stage", "properties", "typing",
                   "energySprite", "energyValues", "providedCount", "useActionCount", "passiveActionCount",
                   "triggeredActionCount"}


@dataclass(frozen=True)
class EnergyDomain:
    schema: str
    energies: dict
    rows: int
    modes: tuple
    artifact_prefix: str
    suite_prefix: str


BASIC_ENERGY_DOMAIN = EnergyDomain("basic-energy-family/1", ENERGIES, 278, ENERGY_MODES,
                                   "basicEnergy", "basic-energy")
FAIRY_ENERGY_DOMAIN = EnergyDomain("fairy-basic-energy-family/1", {**ENERGIES, "Y": (2004224, 6, "Fairy")},
                                   294, (*ENERGY_MODES, "cost-Y"), "fairyEnergy", "fairy-basic-energy")


def energy_symbol(card, domain=BASIC_ENERGY_DOMAIN):
    if (set(card) != SOURCE_FIELDS or card["cardFormat"] != 1 or card["cardStage"] != 0
            or card["cardProperties"] != [0]):
        raise ValueError("source card is outside the reviewed Basic Energy domain")
    for symbol, (_, typing, sprite) in domain.energies.items():
        if (card["cardName"] == f"Basic {{{symbol}}} Energy" and card["cardTyping"] == [typing]
                and card["energySprite"] == sprite
                and card["energyValues"] == [{"value": "{" + symbol + "}"}]):
            return symbol
    raise ValueError("source energy type is unsupported by the reviewed family")


def energy_metadata_scenario(item, source_id, domain=BASIC_ENERGY_DOMAIN):
    return {"suite": f"{domain.suite_prefix}-metadata/1", "commands": [
        {"operation": "nativeEnergyMetadata", "programSha256": item["programSha256"],
         "cardSourceID": source_id, "cardSha256": item["cardSha256"][source_id]}]}


def energy_rules_scenario(item, domain=BASIC_ENERGY_DOMAIN):
    return {"suite": f"{domain.suite_prefix}-rules/1", "commands": [
        {"operation": "basicEnergyRules", "programSha256": item["programSha256"],
         "symbol": item["symbol"], "mode": mode,
         "fixture": "synthetic-plain-basic-1000hp-retreat1-single10damage-typed-cost"}
        for mode in domain.modes]}


def _response(row, item):
    if (not isinstance(row, dict) or set(row) != set(RESPONSE_FIELDS) | {"ownerView", "opponentView"}
            or any(type(row[key]) is not kind for key, kind in RESPONSE_FIELDS.items())):
        raise ValueError("energy fixture response differs from the reviewed observation shape")
    owner, opponent = row["ownerView"], row["opponentView"]
    if (not isinstance(owner, dict) or not isinstance(opponent, dict)
            or not isinstance(owner.get("hand"), list) or not owner.get("active")
            or not opponent.get("active") or not isinstance(owner.get("prize"), list)
            or not isinstance(opponent.get("prize"), list)):
        raise ValueError("energy fixture omitted actual private player views")
    visible_cards = owner["active"] + owner["bench"] + opponent["active"] + opponent["bench"]
    attached = [energy for card in visible_cards for energy in card["energyCards"]]
    return {**{key: row[key] for key in RESPONSE_FIELDS}, "privacy": {
        "ownerHandKnown": all(card.get("id", 0) > 0 for card in owner["hand"]),
        "opponentHandHidden": opponent.get("hand", object()) is None,
        "deckHiddenFromBoth": "deck" not in owner and "deck" not in opponent,
        "prizesHiddenFromBoth": all(card is None for card in owner["prize"] + opponent["prize"]),
        "attachedEnergyPublic": all(energy["id"] == item["cabtCardId"] for energy in attached),
        "discardedEnergyPublic": sum(card["id"] == item["cabtCardId"] for card in opponent["discard"])
                                  == row["discardEnergy"]}}


def run_energy_family(cases, artifacts, cards, source_rows, domain=BASIC_ENERGY_DOMAIN):
    if f"{domain.artifact_prefix}Manifest" not in artifacts:
        raise ValueError("Basic Energy cases require a pinned native family fixture")
    manifest = json.loads(artifacts[f"{domain.artifact_prefix}Manifest"].read_bytes())
    if (manifest.get("schema") != domain.schema or manifest.get("programs") != len(domain.energies)
            or manifest.get("rows") != domain.rows):
        raise ValueError("Basic Energy family manifest differs from the frozen domain")
    by_source = {card["cardSourceID"]: (card, row_hash, program) for card, _, row_hash, program in source_rows}
    items, row_items = {}, {}
    for item in manifest.get("selected", []):
        program, source_ids, native_id = item["programSha256"], item["cardSourceIDs"], item["cabtCardId"]
        symbol = item["symbol"]
        if (program in items or symbol not in domain.energies or type(native_id) is not int
                or native_id != domain.energies[symbol][0] or native_id not in cards
                or not isinstance(source_ids, list) or not source_ids or len(set(source_ids)) != len(source_ids)
                or set(item["cardSha256"]) != set(source_ids)):
            raise ValueError("Basic Energy manifest contains invalid program or native identities")
        for source_id in source_ids:
            if source_id in row_items or source_id not in by_source:
                raise ValueError("Basic Energy source row is duplicate or absent from pinned corpus")
            card, row_hash, source_program = by_source[source_id]
            if (source_program != program or row_hash != item["cardSha256"][source_id]
                    or energy_symbol(card, domain) != symbol):
                raise ValueError("Basic Energy manifest differs from the exact source program")
            row_items[source_id] = item
        items[program] = item
    if (len(items) != len(domain.energies) or len(row_items) != domain.rows
            or {item["symbol"] for item in items.values()} != set(domain.energies)):
        raise ValueError("Basic Energy family inventory differs from its frozen recognition")
    fixture = artifacts[f"{domain.artifact_prefix}Fixture"]
    registry = json.loads(subprocess.run([str(fixture), "registry"], capture_output=True, text=True,
                                         encoding="utf-8", timeout=60, check=True).stdout)
    library = ctypes.CDLL(str(artifacts["library"]))
    library.AllAttack.argtypes = []
    library.AllAttack.restype = ctypes.c_char_p
    attacks = library.AllAttack()
    if not attacks or registry != {"entries": list(cards.values()), "attacks": json.loads(attacks)}:
        raise ValueError("Basic Energy fixture catalogs differ from the actual candidate DLL")
    actual = json.loads(subprocess.run([str(fixture)], capture_output=True, text=True,
                                       encoding="utf-8", timeout=60, check=True).stdout)
    if not isinstance(actual, dict) or set(actual) != {"metadata", "responses"}:
        raise ValueError("Basic Energy fixture omitted metadata or rule observations")
    metadata, responses = {}, {}
    for row in actual["metadata"]:
        source_id = row.get("cardSourceID")
        item = row_items.get(source_id)
        if (item is None or source_id in metadata or set(row) != METADATA_FIELDS
                or row["programSha256"] != item["programSha256"]):
            raise ValueError("Basic Energy metadata observations contain invalid source bindings")
        metadata[source_id] = row
    for row in actual["responses"]:
        item = items.get(row.get("programSha256"))
        if item is None or row.get("symbol") != item["symbol"]:
            raise ValueError("Basic Energy rule observations contain unsupported programs")
        observation = _response(row, item)
        key = (row["programSha256"], row["mode"])
        if key in responses:
            raise ValueError("Basic Energy rule observations contain duplicate scenarios")
        responses[key] = observation
    if metadata.keys() != row_items.keys() or responses.keys() != {(program, mode) for program in items for mode in domain.modes}:
        raise ValueError("Basic Energy native observation inventory differs from the reviewed matrix")
    observed = []
    for case in cases:
        case_id = case["caseId"]
        parts = case_id.split("/")
        if len(parts) == 3 and parts[0] == "row" and parts[2] == "metadata_identity" and parts[1] in row_items:
            source_id = parts[1]
            item = row_items[source_id]
            scenario = energy_metadata_scenario(item, source_id, domain)
            selected = [metadata[source_id]]
        elif len(parts) == 3 and parts[0] == "program" and parts[2] == "rules" and parts[1] in items:
            item = items[parts[1]]
            scenario = energy_rules_scenario(item, domain)
            selected = [responses[(item["programSha256"], mode)] for mode in domain.modes]
        else:
            raise ValueError("Basic Energy fixture cannot admit unsupported source cases")
        if case["scenario"] != scenario or case["cabtCardId"] != item["cabtCardId"]:
            raise ValueError("Basic Energy scenario differs from the reviewed native identity and commands")
        observed.append({"caseId": case_id, "cabtCardId": item["cabtCardId"], "scenarioSha256": digest(scenario),
                         "observation": {"commands": scenario["commands"], "responses": selected},
                         "scenarioComponents": [{"suite": scenario["suite"], "scenarioSha256": digest(scenario)}]})
    return observed
