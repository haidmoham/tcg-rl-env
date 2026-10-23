"""Finite source bindings and actual native observations for plain evolved cards."""

import ctypes
from dataclasses import dataclass
from hashlib import sha256
import json
import subprocess

from tcg.flat_rules_parity import RULE_MODES, RULE_RESPONSE_FIELDS
from tcg.live_parity import digest


@dataclass(frozen=True)
class EvolvedDomain:
    manifest_schema: str
    header: str
    artifact_prefix: str
    suite_prefix: str
    programs: int
    rows: int
    attacks: int
    new_programs: int
    stage2_programs: int
    names_artifact: str | None = None


EVOLVED_DOMAIN = EvolvedDomain("evolved-flat-attack-family/1", "GeneratedEvolvedFlatCards.h",
                               "evolved", "evolved-flat", 113, 267, 169, 96, 3)
EXTRA_EVOLVED_DOMAIN = EvolvedDomain("extra-evolved-flat-attack-family/1", "GeneratedExtraEvolvedFlatCards.h",
                                     "extraEvolved", "extra-evolved-flat", 48, 110, 78, 48, 1,
                                     "canonical-names.json")


EVOLUTION_MODES = ("eligible-active", "eligible-bench", "wrong-family", "wrong-stage-synthetic",
                   "wrong-stage-source", "devolve-primitive", "first-turn", "same-turn-placement",
                   "inherit", "asleep", "paralyzed", "confused", "poisoned", "burned")
EVOLUTION_FIELDS = {"programSha256": str, "mode": str, "legal": bool, "evolved": bool,
                    "restoredPredecessor": bool, "damage": int, "energy": int, "tools": int,
                    "predecessors": int, "hand": int, "statuses": bool, "currentPlayer": int,
                    "prompts": int, "internalPredecessor": str, "internalBase": str}
EVOLUTION_VIEWS = {"beforeOwnerView", "beforeOpponentView", "ownerView", "opponentView"}
RETREAT_MODES = ("retreat-none", "retreat-short", "retreat-exact", "retreat-surplus", "retreat-no-bench",
                 "retreat-used", "retreat-asleep", "retreat-paralyzed", "retreat-confused", "retreat-poisoned",
                 "retreat-burned")
RETREAT_FIELDS = {"programSha256": str, "mode": str, "evolutionLegal": bool,
                  "inheritedDamage": int, "inheritedEnergy": int, "inheritedTools": int,
                  "inheritedPredecessors": int, "legal": bool, "subjectOnBench": bool,
                  "subjectIsActive": bool, "subjectDamage": int, "subjectAttachedEnergy": int,
                  "subjectTools": int, "subjectPredecessors": int, "discardedEnergy": int,
                  "ownerBench": int, "opponentBench": int, "conditionAfter": int,
                  "retreatRecorded": bool, "secondRetreatLegal": bool, "currentPlayer": int,
                  "gameOver": bool}


def evolved_attack_scenario(item, attack_index, domain=EVOLVED_DOMAIN):
    from tcg.parity_runner import flat_attack_scenario

    return {**flat_attack_scenario(item, attack_index), "suite": f"{domain.suite_prefix}-attack-matrix/1"}


def evolved_metadata_scenario(item, source_id, domain=EVOLVED_DOMAIN):
    from tcg.parity_runner import flat_metadata_scenario

    return {**flat_metadata_scenario(item, source_id), "suite": f"{domain.suite_prefix}-card-metadata/1"}


def evolution_modes(item):
    modes = list(EVOLUTION_MODES)
    if item["card"]["stage"] == 3:
        modes.remove("same-turn-placement")
        modes.append("same-turn-evolution")
    return modes


def _rules_components(item, domain=EVOLVED_DOMAIN):
    program = item["programSha256"]
    return [
        {"suite": f"{domain.suite_prefix}-attack-matrix/1", "commands": [
            command for index in range(len(item["card"]["attacks"]))
            for command in evolved_attack_scenario(item, index, domain)["commands"]]},
        {"suite": f"{domain.suite_prefix}-attack-rules/1", "commands": [
            {"operation": "flatAttackRules", "programSha256": program, "attackIndex": index, "mode": mode}
            for index in range(len(item["card"]["attacks"])) for mode in RULE_MODES]},
        {"suite": f"{domain.suite_prefix}-evolution/1", "commands": [
            {"operation": "evolvePlainSource", "programSha256": program, "mode": mode,
             "predecessor": "synthetic-exact-source-name-stage-base-chain"} for mode in evolution_modes(item)]},
        {"suite": f"{domain.suite_prefix}-retreat/1", "commands": [
            {"operation": "retreatAfterEvolution", "programSha256": program, "mode": mode,
             "fixture": "legal-hand-evolution-inherit30damage-energy-tool-predecessors"} for mode in RETREAT_MODES]},
    ]


def evolved_rules_scenario(item, domain=EVOLVED_DOMAIN):
    return {"suite": f"{domain.suite_prefix}-rules/1", "commands": [
        command for component in _rules_components(item, domain) for command in component["commands"]]}


def evolved_rules_components(item, domain=EVOLVED_DOMAIN):
    return [{"suite": component["suite"], "scenarioSha256": digest(component)}
            for component in _rules_components(item, domain)]


def _source_card(card, names):
    from tcg.parity_runner import _flat_source_card

    stage, parent, bases = card.get("cardStage"), card.get("evolvesFrom"), card.get("evolvesFromBase")
    if (type(stage) is not int or stage not in (2, 3) or not isinstance(parent, str) or not parent
            or len(names.get(parent, set())) != 1 or not isinstance(bases, list) or len(bases) != 1
            or not isinstance(bases[0], str) or len(names.get(bases[0], set())) != 1
            or (stage == 2 and bases != [parent]) or len(names.get(card.get("cardName"), set())) > 1):
        raise ValueError("evolved source has unsupported stage or ambiguous predecessor identities")
    normalized = {key: value for key, value in card.items() if key != "evolvesFrom"}
    normalized.update(cardStage=1, evolvesFromBase=[card.get("cardName")])
    return {**_flat_source_card(normalized), "stage": stage, "predecessor": parent,
            "predecessor_internal": next(iter(names[parent])), "base_names": bases,
            "base_internal": [next(iter(names[bases[0]]))],
            "internal_name": next(iter(names.get(card["cardName"], {card["cardName"]})))}


def _source_chain(card, ordinary):
    parent, base = card["evolvesFrom"], card["evolvesFromBase"][0]
    parents, bases = ordinary.get(parent, []), ordinary.get(base, [])
    expected = card["cardStage"] - 1
    if (not parents or not bases or {c.get("cardStage", 1) for c in parents} != {expected}
            or {c.get("cardStage", 1) for c in bases} != {1}
            or any(c.get("evolvesFromBase") != [base] for c in parents + bases)
            or (expected == 2 and any(c.get("evolvesFrom") != base for c in parents))):
        raise ValueError("extra evolved predecessor differs from the complete ordinary source chain")


def _source_programs(artifacts, candidate, cards, source_rows, domain=EVOLVED_DOMAIN):
    if domain not in (EVOLVED_DOMAIN, EXTRA_EVOLVED_DOMAIN):
        raise ValueError("evolved admission requires an explicitly reviewed frozen domain")
    manifest = json.loads(artifacts[f"{domain.artifact_prefix}AttackManifest"].read_bytes())
    selected = manifest.get("selected")
    names_raw = artifacts[f"{domain.artifact_prefix}Names"].read_bytes()
    names_pin = (manifest.get("artifacts", {}).get(domain.names_artifact) if domain.names_artifact
                 else manifest.get("inputs", {}).get("names", {}).get("sha256"))
    if (manifest.get("schema") != domain.manifest_schema
            or manifest.get("headerSha256") != candidate["sourceSha256"].get(domain.header)
            or manifest.get("denominator") != {"rows": 22945, "programs": 9001}
            or manifest.get("supportedPrograms") != domain.programs or manifest.get("supportedRows") != domain.rows
            or not isinstance(selected, list) or len(selected) != domain.programs
            or sha256(names_raw).hexdigest() != names_pin):
        raise ValueError("evolved manifest differs from the frozen source and name inventory")
    names = {}
    for entry in json.loads(names_raw):
        if (not isinstance(entry, dict) or not isinstance(entry.get("name"), str)
                or not isinstance(entry.get("internalName"), str) or not entry["internalName"]):
            raise ValueError("evolved name map contains an invalid native identity")
        names.setdefault(entry["name"], set()).add(entry["internalName"])
    by_source, program_sources, ordinary = {}, {}, {}
    for card, _, row_hash, program in source_rows:
        source_id = card["cardSourceID"]
        if source_id in by_source:
            raise ValueError("evolved corpus contains duplicate printing identities")
        by_source[source_id] = (card, row_hash, program)
        program_sources.setdefault(program, set()).add(source_id)
        if (card.get("cardProperties") == [0] and card.get("prizeValue") == 1
                and card.get("cardStage", 1) in (1, 2, 3)):
            ordinary.setdefault(card["cardName"], []).append(card)
    if len(by_source) != 22945 or len(program_sources) != 9001:
        raise ValueError("evolved source bindings require the complete frozen corpus denominator")
    items, row_items, native_ids = {}, {}, set()
    for item in selected:
        program, source_ids, native_id = item["programSha256"], item["cardSourceIDs"], item["cabtCardId"]
        if (program in items or type(native_id) is not int or native_id not in cards or native_id in native_ids
                or cards[native_id]["attacks"] != item["attackIds"] or cards[native_id]["skills"]
                or not isinstance(source_ids, list) or not source_ids
                or any(not isinstance(source_id, str) for source_id in source_ids)
                or len(set(source_ids)) != len(source_ids) or set(item["cardSha256"]) != set(source_ids)
                or set(source_ids) != program_sources.get(program)):
            raise ValueError("evolved manifest has invalid program, printing or native card identities")
        for source_id in source_ids:
            if source_id in row_items or source_id not in by_source:
                raise ValueError("evolved printing is duplicate or absent from the pinned corpus")
            card, row_hash, source_program = by_source[source_id]
            if (source_program != program or row_hash != item["cardSha256"][source_id]
                    or _source_card(card, names) != item["card"]):
                raise ValueError("evolved declaration differs from the complete pinned source program")
            if domain == EXTRA_EVOLVED_DOMAIN:
                _source_chain(card, ordinary)
                if cards[native_id]["evolvesFrom"] != card["evolvesFrom"]:
                    raise ValueError("extra evolved native predecessor differs from the source parent name")
            row_items[source_id] = item
        items[program] = item
        native_ids.add(native_id)
    if (len(row_items) != domain.rows or sum(len(item["card"]["attacks"]) for item in selected) != domain.attacks
            or sum(not item["existing"] for item in selected) != domain.new_programs
            or sum(item["card"]["stage"] == 3 for item in selected) != domain.stage2_programs):
        raise ValueError("evolved family differs from the reviewed finite inventory")
    return items, row_items


def _execute(fixture, *arguments):
    return json.loads(subprocess.run([str(fixture), *arguments], capture_output=True, text=True,
                                    encoding="utf-8", timeout=120, check=True).stdout)


def _rows(raw, fields, expected_keys, transform=None):
    if not isinstance(raw, list):
        raise ValueError("evolved fixture must emit actual native observation rows")
    observations = {}
    for row in raw:
        if (not isinstance(row, dict) or set(row) != set(fields)
                or any(type(row[key]) is not kind for key, kind in fields.items())):
            raise ValueError("evolved fixture response differs from the reviewed observation shape")
        key = ((row["programSha256"], row["attackIndex"], row["mode"]) if "attackIndex" in fields
               else (row["programSha256"], row["mode"]))
        if key not in expected_keys or key in observations:
            raise ValueError("evolved fixture contains unknown or duplicate native observations")
        observations[key] = transform(row) if transform else row
    if observations.keys() != expected_keys:
        raise ValueError("evolved fixture inventory differs from the complete reviewed matrix")
    return observations


def _evolution_observation(row, items):
    item = items[row["programSha256"]]
    if (row["internalPredecessor"] != item["card"]["predecessor_internal"]
            or row["internalBase"] != item["card"]["base_internal"][0]):
        raise ValueError("evolved native predecessor chain differs from the source identities")
    owner, opponent = row["ownerView"], row["opponentView"]
    key = "bench" if row["mode"] in ("eligible-bench", "same-turn-placement") else "active"
    if (not isinstance(owner, dict) or not isinstance(opponent, dict)
            or not isinstance(owner.get("hand"), list) or not owner.get(key) or not opponent.get(key)):
        raise ValueError("evolved fixture omitted actual player-specific evolution views")
    target = opponent[key][0]
    if row["evolved"] and target["id"] != item["cabtCardId"]:
        raise ValueError("evolved native operation produced the wrong source card identity")
    attachments = target["energyCards"] + target["tools"] + target["preEvolution"]
    return {**{field: row[field] for field in EVOLUTION_FIELDS}, "privacy": {
        "ownerHandKnown": all(card.get("id", 0) > 0 for card in owner["hand"]),
        "opponentHandHidden": opponent.get("hand", object()) is None,
        "deckHiddenFromBoth": "deck" not in owner and "deck" not in opponent,
        "targetPublic": owner[key][0]["id"] == target["id"] and target["id"] > 0,
        "attachmentsPublic": all(card.get("id", 0) > 0 for card in attachments),
        "publicDamage": target["maxHp"] - target["hp"], "publicEnergy": len(target["energyCards"]),
        "publicTools": len(target["tools"]), "publicPredecessors": len(target["preEvolution"])}}


def _attack_observation(row, items):
    from tcg.parity_runner import _flat_observation

    observed = _flat_observation(row)
    item = items[row["programSha256"]]
    if (row["ownerView"]["active"][0].get("id") != item["cabtCardId"]
            or row["opponentView"]["active"][0].get("id") != item["cabtCardId"]
            or (row["mode"] == "bench"
                and (not row["ownerView"].get("bench")
                     or row["ownerView"]["bench"][0]["id"] != item["cabtCardId"]))):
        raise ValueError("evolved attack views do not bind the requested native source card")
    return observed


def _retreat_observation(row, items):
    item = items[row["programSha256"]]
    if not row["evolutionLegal"]:
        raise ValueError("evolved retreat fixture omitted the actual legal evolution prerequisite")
    owner, opponent = row["ownerView"], row["opponentView"]
    key = "bench" if row["legal"] else "active"
    if (not isinstance(owner.get("hand"), list) or not owner.get(key) or not opponent.get(key)
            or owner[key][0].get("id") != item["cabtCardId"]
            or opponent[key][0].get("id") != item["cabtCardId"]):
        raise ValueError("evolved retreat views do not bind the legally evolved source identity")
    subject = opponent[key][0]
    attachments = subject["energyCards"] + subject["tools"] + subject["preEvolution"]
    return {**{field: row[field] for field in RETREAT_FIELDS}, "privacy": {
        "ownerHandKnown": all(card.get("id", 0) > 0 for card in owner["hand"]),
        "opponentHandHidden": opponent.get("hand", object()) is None,
        "deckHiddenFromBoth": "deck" not in owner and "deck" not in opponent,
        "subjectPublic": owner[key][0]["id"] == subject["id"] and subject["id"] > 0,
        "attachmentsPublic": all(card.get("id", 0) > 0 for card in attachments),
        "publicDamage": subject["maxHp"] - subject["hp"], "publicEnergy": len(subject["energyCards"]),
        "publicTools": len(subject["tools"]), "publicPredecessors": len(subject["preEvolution"])}}


def run_evolved_family(cases, artifacts, candidate, cards, source_rows, domain=EVOLVED_DOMAIN):
    from tcg.parity_runner import FLAT_RESPONSE_SHAPE, TRANSITION_FACETS, _flat_metadata

    prefix = domain.artifact_prefix
    required = {f"{prefix}{suffix}" for suffix in
                ("AttackManifest", "Names", "AttackFixture", "RulesFixture", "EvolutionFixture")}
    if domain == EXTRA_EVOLVED_DOMAIN:
        required.add(f"{prefix}RetreatFixture")
    if not required <= artifacts.keys():
        raise ValueError("evolved cases require pinned source, names and all native companion fixtures")
    items, row_items = _source_programs(artifacts, candidate, cards, source_rows, domain)
    registry = _execute(artifacts[f"{prefix}AttackFixture"], "registry")
    library = ctypes.CDLL(str(artifacts["library"]))
    library.AllAttack.argtypes = []
    library.AllAttack.restype = ctypes.c_char_p
    raw_attacks = library.AllAttack()
    if not raw_attacks or registry != {"entries": list(cards.values()), "attacks": json.loads(raw_attacks)}:
        raise ValueError("evolved fixture catalogs differ from the actual candidate DLL")
    attacks = {row["attackId"]: row for row in json.loads(raw_attacks)}
    attack_keys = {(program, index, command["mode"]) for program, item in items.items()
                   for index in range(len(item["card"]["attacks"]))
                   for command in evolved_attack_scenario(item, index, domain)["commands"]}
    actual = _rows(_execute(artifacts[f"{prefix}AttackFixture"]),
                   {**FLAT_RESPONSE_SHAPE, "ownerView": dict, "opponentView": dict}, attack_keys,
                   lambda row: _attack_observation(row, items))
    rule_keys = {(program, index, mode) for program, item in items.items()
                 for index in range(len(item["card"]["attacks"])) for mode in RULE_MODES}
    rules = _rows(_execute(artifacts[f"{prefix}RulesFixture"]), RULE_RESPONSE_FIELDS, rule_keys)
    evolution_keys = {(program, mode) for program, item in items.items() for mode in evolution_modes(item)}
    evolution = _rows(_execute(artifacts[f"{prefix}EvolutionFixture"]),
                      {**EVOLUTION_FIELDS, **dict.fromkeys(EVOLUTION_VIEWS, dict)}, evolution_keys,
                      lambda row: _evolution_observation(row, items))
    retreat = None
    if f"{prefix}RetreatFixture" in artifacts:
        retreat_keys = {(program, mode) for program in items for mode in RETREAT_MODES}
        retreat = _rows(_execute(artifacts[f"{prefix}RetreatFixture"]),
                        {**RETREAT_FIELDS, "ownerView": dict, "opponentView": dict}, retreat_keys,
                        lambda row: _retreat_observation(row, items))
    observations, seen = [], set()
    for case in cases:
        parts = case["caseId"].split("/")
        if case["caseId"] in seen:
            raise ValueError("evolved fixture cannot admit duplicate source cases")
        seen.add(case["caseId"])
        if len(parts) == 3 and parts[0] == "row" and parts[2] == "metadata_identity" and parts[1] in row_items:
            source_id, item = parts[1], row_items[parts[1]]
            scenario = evolved_metadata_scenario(item, source_id, domain)
            responses = [_flat_metadata(item, source_id, cards, attacks)]
            components = [{"suite": scenario["suite"], "scenarioSha256": digest(scenario)}]
        elif (len(parts) == 5 and parts[0] == "program" and parts[1] in items
              and parts[2] == "useActions" and parts[4] in TRANSITION_FACETS):
            item, index = items[parts[1]], int(parts[3])
            if str(index) != parts[3] or not 0 <= index < len(item["card"]["attacks"]):
                raise ValueError("evolved attack index differs from the complete source program")
            scenario = evolved_attack_scenario(item, index, domain)
            responses = [actual[(item["programSha256"], index, command["mode"])]
                         for command in scenario["commands"]]
            components = [{"suite": scenario["suite"], "scenarioSha256": digest(scenario)}]
        elif (len(parts) == 3 and parts[0] == "program" and parts[1] in items and parts[2] == "rules"
              and retreat is not None):
            program, item = parts[1], items[parts[1]]
            scenario = evolved_rules_scenario(item, domain)
            responses = [actual[(program, index, command["mode"])]
                         for index in range(len(item["card"]["attacks"]))
                         for command in evolved_attack_scenario(item, index, domain)["commands"]]
            responses.extend(rules[(program, index, mode)]
                             for index in range(len(item["card"]["attacks"])) for mode in RULE_MODES)
            responses.extend(evolution[(program, mode)] for mode in evolution_modes(item))
            responses.extend(retreat[(program, mode)] for mode in RETREAT_MODES)
            components = evolved_rules_components(item, domain)
        else:
            raise ValueError("evolved rule admission requires a complete measured retreat campaign")
        if case["scenario"] != scenario or case["cabtCardId"] != item["cabtCardId"]:
            raise ValueError("evolved case differs from the reviewed native identity and commands")
        observations.append({"caseId": case["caseId"], "cabtCardId": item["cabtCardId"],
                             "scenarioSha256": digest(scenario),
                             "observation": {"commands": scenario["commands"], "responses": responses},
                             "scenarioComponents": components})
    return observations
