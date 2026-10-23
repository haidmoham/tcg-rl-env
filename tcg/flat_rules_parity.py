"""Finite native rule campaigns for reviewed plain Basic flat-attack programs."""

import json
import subprocess

from tcg.live_parity import digest


RULE_MODES = ("first-turn", "asleep", "paralyzed", "confused-heads", "confused-tails",
              "knockout", "last-prize", "no-bench")
MOVEMENT_MODES = ("place-empty-bench", "place-four-bench", "place-full-bench", "place-first-turn",
                  "place-off-turn", "retreat-none", "retreat-short", "retreat-exact", "retreat-surplus",
                  "retreat-no-bench", "retreat-used", "retreat-asleep", "retreat-paralyzed",
                  "retreat-confused", "retreat-poisoned", "retreat-burned")
RULE_RESPONSE_FIELDS = {"programSha256": str, "attackIndex": int, "mode": str, "legal": bool,
                        "actorDamage": int, "defenderDamage": int, "actorDiscarded": bool,
                        "defenderDiscarded": bool, "actorPrizes": int, "opponentPrizes": int,
                        "actorBench": int, "opponentBench": int, "currentPlayer": int,
                        "gameOver": bool, "actorWon": bool, "coins": int}
MOVEMENT_RESPONSE_FIELDS = {"programSha256": str, "mode": str, "legal": bool,
                            "subjectInHand": bool, "subjectOnBench": bool, "subjectIsActive": bool,
                            "subjectDamage": int, "subjectAttachedEnergy": int, "discardedEnergy": int,
                            "ownerBench": int, "opponentBench": int, "conditionAfter": int,
                            "retreatRecorded": bool, "secondRetreatLegal": bool,
                            "currentPlayer": int, "gameOver": bool}


def flat_rules_scenario(item):
    program = item["programSha256"]
    return {"suite": "flat-basic-rules/1", "commands": [
        {"operation": "flatAttackRules", "programSha256": program, "attackIndex": attack_index,
         "mode": mode}
        for attack_index in range(len(item["card"]["attacks"])) for mode in RULE_MODES] + [
        {"operation": "flatBasicMovement", "programSha256": program, "mode": mode}
        for mode in MOVEMENT_MODES]}


def flat_rules_components(item):
    commands = flat_rules_scenario(item)["commands"]
    attack_count = len(item["card"]["attacks"]) * len(RULE_MODES)
    return [{"suite": suite, "scenarioSha256": digest({"suite": suite, "commands": component})}
            for suite, component in (("flat-attack-rules/1", commands[:attack_count]),
                                     ("flat-basic-movement/1", commands[attack_count:]))]


def _source_programs(artifacts, candidate, cards, source_rows):
    from tcg.parity_runner import _flat_source_card

    manifest = json.loads(artifacts["flatAttackManifest"].read_bytes())
    selected = manifest.get("selected")
    if (manifest.get("schema") != "flat-attack-family/1"
            or manifest.get("headerSha256") != candidate["sourceSha256"]["GeneratedFlatCards.h"]
            or not isinstance(selected, list) or len(selected) != 733):
        raise ValueError("flat rules manifest differs from the reviewed generated family")
    by_source, program_sources = {}, {}
    for card, _, row_hash, program in source_rows:
        source_id = card["cardSourceID"]
        if source_id in by_source:
            raise ValueError("flat rules corpus contains duplicate source identities")
        by_source[source_id] = (card, row_hash, program)
        program_sources.setdefault(program, set()).add(source_id)
    items, source_ids_seen, native_ids = {}, set(), set()
    for item in selected:
        program, source_ids, native_id = item["programSha256"], item["cardSourceIDs"], item["cabtCardId"]
        if (program in items or type(native_id) is not int or native_id not in cards
                or native_id in native_ids or cards[native_id]["attacks"] != item["attackIds"]):
            raise ValueError("flat rules manifest contains invalid source or native program identities")
        if (not isinstance(source_ids, list) or not source_ids
                or any(not isinstance(source_id, str) for source_id in source_ids)
                or len(set(source_ids)) != len(source_ids)
                or set(item["cardSha256"]) != set(source_ids)
                or set(source_ids) != program_sources.get(program)):
            raise ValueError("flat rules manifest must bind every printing of each full source program")
        declared = {key: value for key, value in item["card"].items() if key != "internal_name"}
        for source_id in source_ids:
            if source_id in source_ids_seen or source_id not in by_source:
                raise ValueError("flat rules printing is duplicate or absent from the pinned corpus")
            card, row_hash, source_program = by_source[source_id]
            if (source_program != program or row_hash != item["cardSha256"][source_id]
                    or _flat_source_card(card) != declared):
                raise ValueError("flat rules manifest differs from the complete pinned source program")
            source_ids_seen.add(source_id)
        items[program] = item
        native_ids.add(native_id)
    if (len(source_ids_seen) != 1692
            or sum(len(item["card"]["attacks"]) for item in selected) != 1018):
        raise ValueError("flat rules family inventory differs from the reviewed finite domain")
    return items


def _campaign_observations(fixture, fields, expected_keys, include_attack):
    raw = json.loads(subprocess.run([str(fixture)], capture_output=True, text=True, encoding="utf-8",
                                    timeout=120, check=True).stdout)
    if not isinstance(raw, list):
        raise ValueError("flat rules fixture must emit actual native observation rows")
    observations = {}
    for row in raw:
        if (not isinstance(row, dict) or set(row) != set(fields)
                or any(type(row[key]) is not kind for key, kind in fields.items())):
            raise ValueError("flat rules fixture response differs from the reviewed observation shape")
        key = ((row["programSha256"], row["attackIndex"], row["mode"]) if include_attack
               else (row["programSha256"], row["mode"]))
        if key not in expected_keys or key in observations:
            raise ValueError("flat rules fixture contains unknown or duplicate program and mode identities")
        observations[key] = row
    if observations.keys() != expected_keys:
        raise ValueError("flat rules native observation inventory differs from the complete reviewed matrix")
    return observations


def run_flat_rules_family(cases, artifacts, candidate, cards, source_rows):
    required = {"flatAttackManifest", "flatRulesRulesFixture", "flatRulesMovementFixture"}
    if not required <= artifacts.keys():
        raise ValueError("flat Basic rules require both pinned native campaign fixtures and source manifest")
    items = _source_programs(artifacts, candidate, cards, source_rows)
    selected_cases, case_ids = [], set()
    for case in cases:
        parts = case["caseId"].split("/")
        if (len(parts) != 3 or parts[0] != "program" or parts[2] != "rules"
                or parts[1] not in items or case["caseId"] in case_ids):
            raise ValueError("flat Basic rules fixture cannot admit duplicate or unsupported source cases")
        item = items[parts[1]]
        scenario = flat_rules_scenario(item)
        if case["scenario"] != scenario or case["cabtCardId"] != item["cabtCardId"]:
            raise ValueError("flat Basic rules scenario differs from the reviewed native identity and commands")
        selected_cases.append((case, item, scenario))
        case_ids.add(case["caseId"])
    rule_keys = {(program, index, mode) for program, item in items.items()
                 for index in range(len(item["card"]["attacks"])) for mode in RULE_MODES}
    movement_keys = {(program, mode) for program in items for mode in MOVEMENT_MODES}
    rules = _campaign_observations(artifacts["flatRulesRulesFixture"], RULE_RESPONSE_FIELDS, rule_keys, True)
    movement = _campaign_observations(artifacts["flatRulesMovementFixture"], MOVEMENT_RESPONSE_FIELDS,
                                      movement_keys, False)
    observations = []
    for case, item, scenario in selected_cases:
        program = item["programSha256"]
        responses = [rules[(program, index, mode)]
                     for index in range(len(item["card"]["attacks"])) for mode in RULE_MODES]
        responses.extend(movement[(program, mode)] for mode in MOVEMENT_MODES)
        observations.append({"caseId": case["caseId"], "cabtCardId": item["cabtCardId"],
                             "scenarioSha256": digest(scenario),
                             "observation": {"commands": scenario["commands"], "responses": responses},
                             "scenarioComponents": flat_rules_components(item)})
    return observations
