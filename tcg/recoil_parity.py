"""Finite recoil observations from separate ordinary-game and full-deck session fixtures."""

import ctypes
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from tcg.live_parity import digest, read_pinned
from tcg.flat_rules_parity import (MOVEMENT_MODES, MOVEMENT_RESPONSE_FIELDS, RULE_MODES,
                                   RULE_RESPONSE_FIELDS, _campaign_observations)


RECOIL_FACETS = ("effect", "legal_prompts", "information_boundaries", "interactions")
RECOIL_MANIFEST_SHA256 = "8abf48e849a52766f9e454a9ca98cc22554e480390ae9bcbee49aaa8aab11ca1"
SOURCE_RECOIL_MANIFEST_SHA256 = "d7448b4fc1e80a05d03217b809f4e392eb03aaa3d0a4ecb5c6e7173f4f18bbf3"
RECOIL_MODES = (
    "sufficient", "none", "short", "wrong", "weakness", "resistance", "self-weakness", "self-resistance",
    "bench", "first-turn", "self-ko", "opponent-ko", "both-ko", "actor-last", "opponent-last", "both-last",
    "self-no-bench", "opponent-no-bench", "both-no-bench",
)
TIE_MODES = ("both-last", "both-no-bench")
RECOIL_FIELDS = (
    "programSha256", "attackIndex", "mode", "legal", "actorDamage", "defenderDamage", "actorDiscarded",
    "defenderDiscarded", "actorPrizes", "opponentPrizes", "actorBench", "opponentBench", "currentPlayer",
    "gameOver", "actorWon", "opponentWon",
)
REVIEWED_SOURCES = {
    "experiments/recoil_family/cabt.cpp": "c40dd40c25a58e8acdda04ff8f04073572edd00e1ee1549b41ed73017562c2eb",
    "experiments/recoil_family/FixturePrograms.h": "f285b6ace365267abd8ffdb8068d4b62399a210032f946bced16f2b390b86f37",
    "experiments/recoil_session/native.cpp": "a55bf198c3f01cecc0d6c1385f50d30a7659346059f60aee86085dbb64def0ed",
    "MatchSession.h": "6d63d1449141b85afd27e3de67a4b6e38ab1ee1e23424eb79904a120b2aa3876",
}
EXPANDED_REVIEWED_SOURCES = {
    "experiments/recoil_family/cabt.cpp": "16aafad5e1c4baa240073ce17785ec3897f76ae123bffd2ad4d4604175272da0",
    "MatchSession.h": "aa7655317d1e03e9fda0663173f31a617613b280f7aaa21ffe5e3c8b5b881883",
}
REVIEWED_RULE_SOURCES = {
    "experiments/recoil_rules/rules.cpp": "892ab1203dc5403cf6efb0a222428ed6e1410e4ede0e65160049ab38e6d8a315",
    "experiments/recoil_rules/movement.cpp": "10dd0582d65357658aad17c2bc4ec1c9e6dd326ea2a4a22cb37cf77aa80a9811",
}
SESSION_SUPPLEMENT_SOURCES = {
    "session.cpp": "a9f75f535ba0698b3abb6e04141fc2a25407853c92bc4974cff3e2aef483363a",
    "session.vcxproj": "aff38d1ed8d0d65d082491daea2608b2788721513b7c7bc6ca2ddd6b20fb7cd8",
}
PAIRED_SESSION_SOURCES = {
    "session.cpp": "b3eb1677fc6cc0f39533f0990280a76df23cfecc3ce1dfc24c35d507a9dff83d",
    "fixture.vcxproj": "706cba255de0d6224ed13896345160e74febc752a7a9e58b59f950d32f6dcaba",
}
ENERGIES = {"Colorless": (1, "bw1_105"), "Grass": (1, "bw1_105"), "Fire": (2, "bw1_106"),
            "Water": (3, "bw1_107"), "Lightning": (4, "bw1_108"), "Psychic": (5, "bw1_109"),
            "Fighting": (6, "bw1_110"), "Darkness": (7, "bw1_111"), "Metal": (8, "bw1_112")}
STAGES = ("initial-playable", "ready-first", "ready-second", "reset-coin", "choose-first",
          "restarted-playable", "terminal")


def recoil_historical_rules_scenario():
    return {"operation": "recoil_knockout_prizes_legality", "omissions": [
        "special conditions", "external modifiers and knockout triggers", "sudden death next-game setup"]}


def recoil_rules_scenario(item):
    program = item["programSha256"]
    ordinary = [command for index in range(len(item["attackIds"]))
                for command in recoil_components(item, index, "effect")[0]["commands"]]
    session = [command for index in range(len(item["attackIds"]))
               for component in recoil_components(item, index, "effect")[1:] for command in component["commands"]]
    components = [{"suite": "recoil-ordinary-native/1", "commands": ordinary},
                  {"suite": "recoil-session-lifecycle/1", "commands": session},
                  {"suite": "recoil-attack-rules/1", "commands": [
                      {"operation": "recoilAttackRules", "programSha256": program, "attackIndex": index, "mode": mode}
                      for index in range(len(item["attackIds"])) for mode in RULE_MODES]},
                  {"suite": "recoil-basic-movement/1", "commands": [
                      {"operation": "recoilBasicMovement", "programSha256": program, "mode": mode}
                      for mode in MOVEMENT_MODES]}]
    return {"suite": "recoil-rules-composite/1", "historicalScenario": recoil_historical_rules_scenario(),
            "criterion": "all-source-attacks-eight-rule-modes-and-basic-sixteen-movement-modes-plus-ordinary-and-session",
            "components": components, "commands": [c for component in components for c in component["commands"]]}


def recoil_modes(item, index):
    return [mode for mode in RECOIL_MODES if item["card"]["typing"] != "Colorless"
            or mode not in {"weakness", "resistance", "self-weakness", "self-resistance"}]


def recoil_records(items):
    records = []
    for ordinal, item in enumerate(items):
        positive = [(index, attack) for index, attack in enumerate(item["card"]["attacks"]) if attack["recoil"] > 0]
        if len(positive) != 1:
            raise ValueError("recoil session requires the frozen single-positive-attack domain")
        index, attack = positive[0]
        records.append({"programIndex": ordinal, "programSha256": item["programSha256"],
                        "cardId": item["cabtCardId"], "cardSourceID": item["cardSourceIDs"][0],
                        "cardSourceIDs": item["cardSourceIDs"], "cardSha256": item["cardSha256"],
                        "attackIndex": index, "attackId": item["attackIds"][index], "hp": item["card"]["hp"],
                        "recoil": attack["recoil"], "energyNativeIDs": [ENERGIES[c][0] for c in attack["energy"]],
                        "energySourceIDs": [ENERGIES[c][1] for c in attack["energy"]]})
    return records


def recoil_legacy_scenario(index=None, facet=None):
    if index is None:
        return {"operation": "native_card_metadata"}
    if facet not in RECOIL_FACETS:
        raise ValueError("recoil rules remain held for unmeasured conditions and external triggers")
    return {"operation": "recoil_native_matrix", "attackIndex": index, "facet": facet,
            "rawPromptTrace": "oracle-raw.json and cabt-raw.json",
            "syntheticControls": "Plain Basic 10000 HP reserve and 10 or 10000 HP defender. "
                                 "Own-type weakness and resistance controls are artificial metadata overrides."}


def recoil_components(item, index=None, facet=None, source_id=None):
    program = item["programSha256"]
    if index is None:
        commands = [{"operation": "nativeCardMetadata", "programSha256": program,
                     "cardSourceID": source_id, "cardSha256": item["cardSha256"][source_id]}]
    else:
        commands = [{"operation": "attack", "programSha256": program, "attackIndex": index, "mode": mode}
                    for mode in recoil_modes(item, index)
                    if not (item["card"]["attacks"][index]["recoil"] > 0 and mode in TIE_MODES)]
    components = [{"suite": "recoil-ordinary-native/1", "legacyScenario": recoil_legacy_scenario(index, facet),
                   "commands": commands}]
    if index is not None and item["card"]["attacks"][index]["recoil"] > 0:
        components.append({"suite": "recoil-session-lifecycle/1", "commands": [
            {"operation": "recoilSession", "programSha256": program, "attackIndex": index, "mode": mode,
             "fixture": "original60-four-source-pokemon-56-real-basic-energy", "stages": list(STAGES),
             "terminal": "actual-winning-attack-with-surviving-actor-bench"} for mode in TIE_MODES]})
    return components


def recoil_scenario(item, index=None, facet=None, source_id=None):
    components = recoil_components(item, index, facet, source_id)
    return {"suite": "recoil-composite/1", "legacyScenario": components[0]["legacyScenario"],
            "components": components, "commands": [c for component in components for c in component["commands"]]}


def recoil_ordinary_observation(row):
    def privacy(views):
        owner, opponent = views["owner"], views["opponent"]
        return {"ownerHandKnown": all(c.get("id", 0) > 0 for c in owner["hand"]),
                "opponentHandHidden": opponent["hand"] is None,
                "deckHiddenFromBoth": "deck" not in owner and "deck" not in opponent,
                "prizesHiddenFromBoth": all(c is None or c.get("id", 0) == 0
                                            for view in (owner, opponent) for c in view["prize"]),
                "discardPublic": all(c.get("id", 0) > 0 for view in (owner, opponent) for c in view["discard"]),
                "activePublic": all(c.get("id", 0) > 0 for view in (owner, opponent) for c in view["active"]),
                "publicActiveDamage": [c["maxHp"] - c["hp"] for c in opponent["active"]],
                "attachedEnergyPublic": all(c.get("id", 0) > 0 for view in (owner, opponent)
                                            for c in view["active"] for c in c["energyCards"])}

    kinds = {8: "prize", 5: "active-replacement"}
    return {**{field: row[field] for field in RECOIL_FIELDS},
            "beforePrivacy": privacy(row["beforeViews"]), "afterPrivacy": privacy(row["afterViews"]),
            "prompts": [{"kind": kinds[p["context"]], "player": p["player"] + 1,
                         "minimum": p["minimum"], "maximum": p["maximum"], "validOptions": p["validOptions"]}
                        for p in row["prompts"]]}


def recoil_session_observation(row):
    stages = {}
    for event in row["events"]:
        name = event["eventName"]
        if name == "playable":
            name = "initial-playable" if "initial-playable" not in stages else "restarted-playable"
        stages[name] = event
    initial = stages["initial-playable"]
    identities = [card[:3] for card in initial["physical"]]
    if len(identities) != 120 or len({c[0] for c in identities}) != 120:
        raise ValueError("recoil session requires 120 distinct original physical cards")
    result = {}
    phases = ("game", "sudden-death-ready", "sudden-death-ready", "coin-call", "choose-first", "game", "finished")
    for name, phase in zip(STAGES, phases):
        event = stages[name]
        a, b = event["views"]
        life, current = a["lifecycle"], a["native"]["current"]
        if [card[:3] for card in event["physical"]] != identities or life["phase"] != phase:
            raise ValueError("recoil session changed physical identity or lifecycle stage")
        for seat, view in enumerate((a, b)):
            if view["prompt"] is not None and life["selectingPlayer"] != seat:
                raise ValueError("recoil session exposed the other player's lifecycle prompt")
            if any(key in view["lifecycle"] for key in ("seed", "seed64", "rng", "rngState")):
                raise ValueError("recoil session exposed hidden chance state")
        prompt = a["prompt"] if a["prompt"] is not None else b["prompt"]
        players = current["players"]
        value = {"matchOver": life["matchOver"], "winners": [life["winner"] == 0, life["winner"] == 1],
                 "inventories": [sum(c[2] == p for c in event["physical"]) for p in (0, 1)],
                 "originalIdentitiesConserved": True, "startingPrizes": life["startingPrizeCount"]}
        if name in ("ready-first", "ready-second", "reset-coin"):
            value.update({"selectingPlayer": life["selectingPlayer"], "promptOptions": prompt["options"],
                          "deckCounts": [p["deckCount"] for p in players], "hands": [p["handCount"] for p in players],
                          "prizes": [len(p["prize"]) for p in players], "active": [len(p["active"]) for p in players],
                          "bench": [len(p["bench"]) for p in players], "discard": [len(p["discard"]) for p in players],
                          "damage": sum(c[4] for c in event["physical"]),
                          "allPhysicalCardsInDeck": all(c[3] == 1 for c in event["physical"])})
        if name in ("ready-first", "ready-second"):
            value["completedGameDraw"] = life["previousGameResult"] == 3
        if name == "reset-coin":
            value.update(turnReset=current["turn"] == 0, firstPlayerUnset=current["firstPlayer"] == -1)
        if name in ("initial-playable", "restarted-playable"):
            value.update(prizes=[len(p["prize"]) for p in players],
                         bothActive=all(len(p["active"]) == 1 for p in players), firstPlayer=current["firstPlayer"])
        result[name] = value
    return result


def _source_programs(artifacts, candidate, cards, source_rows):
    raw = artifacts["recoilManifest"].read_bytes()
    if sha256(raw).hexdigest() not in {RECOIL_MANIFEST_SHA256, SOURCE_RECOIL_MANIFEST_SHA256}:
        raise ValueError("recoil manifest differs from the frozen 76-program source inventory")
    manifest = json.loads(raw)
    if (manifest["supportedPrograms"] != 76 or manifest["supportedRows"] != 179
            or manifest["denominator"] != {"rows": 22945, "programs": 9001}
            or candidate["sourceSha256"].get("GeneratedRecoilCards.h") != manifest["headerSha256"]
            or any(candidate["sourceSha256"].get(path) not in {pin, EXPANDED_REVIEWED_SOURCES.get(path, pin)}
                   for path, pin in REVIEWED_SOURCES.items())):
        raise ValueError("recoil candidate differs from reviewed source and complete denominator")
    by_source = {card["cardSourceID"]: (row_hash, program) for card, _, row_hash, program in source_rows}
    programs = {}
    for card, _, _, program in source_rows:
        programs.setdefault(program, set()).add(card["cardSourceID"])
    if len(by_source) != 22945 or len(programs) != 9001:
        raise ValueError("recoil admission requires the complete pinned corpus")
    items, row_items, native_ids = {}, {}, set()
    for item in manifest["selected"]:
        program, native_id, source_ids = item["programSha256"], item["cabtCardId"], item["cardSourceIDs"]
        if (program in items or type(native_id) is not int or native_id in native_ids or native_id not in cards
                or cards[native_id]["attacks"] != item["attackIds"] or cards[native_id]["skills"]
                or set(source_ids) != programs.get(program) or len(source_ids) != len(set(source_ids))
                or set(item["cardSha256"]) != set(source_ids)):
            raise ValueError("recoil program, printing or native identity changed")
        for source_id in source_ids:
            if source_id in row_items or by_source[source_id] != (item["cardSha256"][source_id], program):
                raise ValueError("recoil printing differs from its complete raw source program")
            row_items[source_id] = item
        items[program] = item
        native_ids.add(native_id)
    if len(items) != 76 or len(row_items) != 179 or sum(len(i["attackIds"]) for i in items.values()) != 96:
        raise ValueError("recoil finite inventory changed")
    expected = recoil_records(list(items.values()))
    records = artifacts["recoilSessionRecords"]
    text = "".join("\t".join([r["programSha256"], str(r["cardId"]), str(r["attackIndex"]), str(r["attackId"]),
                                str(r["recoil"]), str(r["hp"]), r["cardSourceID"],
                                ",".join(map(str, r["energyNativeIDs"]))]) + "\n" for r in expected)
    if records.read_text(encoding="utf-8") != text:
        raise ValueError("recoil session records differ from exact source actors and real Energy costs")
    return items, row_items


def verify_recoil_supplement(artifacts, candidate):
    receipt = json.loads(artifacts["recoilRulesBuildReceipt"].read_bytes())
    if (receipt.get("schema") != "recoil-rules-supplement-build/1"
            or receipt.get("engineSourceCommit") != candidate["sourceCommit"]
            or receipt.get("engineLibrarySha256") != candidate["librarySha256"]
            or receipt.get("engineSourceSha256") != candidate["sourceSha256"]):
        raise ValueError("recoil rule supplement does not bind the exact unchanged engine")
    root = Path(receipt["sourceRoot"]).resolve(strict=True)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"], capture_output=True)
    tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"], text=True).rstrip("\0").split("\0")
    pins = receipt["sourceSha256"]
    if (commit != receipt["sourceCommit"] or dirty.returncode or set(tracked) != set(pins)
            or any(pins.get(name) != pin for name, pin in candidate["sourceSha256"].items())
            or any(pins.get(name) != pin for name, pin in REVIEWED_RULE_SOURCES.items())):
        raise ValueError("recoil rule supplement changed a tracked engine file or reviewed experiment")
    for name, pin in pins.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("recoil supplementary source escapes its frozen root")
        read_pinned(path, pin)
    for kind, artifact in (("rules", "recoilRulesFixture"), ("movement", "recoilMovementFixture")):
        fixture = receipt["fixtures"][kind]
        if artifacts[artifact].resolve(strict=True) != Path(fixture["path"]).resolve(strict=True):
            raise ValueError("recoil supplementary native fixture differs from its build receipt")
        read_pinned(artifacts[artifact], fixture["sha256"])
    return receipt


def verify_recoil_session_supplement(artifacts, candidate):
    if "recoilSessionSupplement" not in artifacts:
        return
    path = artifacts["recoilSessionSupplement"]
    receipt = json.loads(path.read_bytes())
    schemas = {"cabt-recoil-session-supplement/1": SESSION_SUPPLEMENT_SOURCES,
               "cabt-recoil-session-supplement/2": PAIRED_SESSION_SOURCES,
               "source-recoil-paired-native-acceptance/1": PAIRED_SESSION_SOURCES}
    sources = schemas.get(receipt.get("schema"))
    if (sources is None
            or receipt.get("sourceCommit") != candidate["sourceCommit"]
            or receipt.get("sourceSha256") != candidate["sourceSha256"]
            or receipt.get("sourceAndObjectRetained") is not True
            or receipt.get("successful") is not True
            or any(receipt.get("supplementalSource", {}).get(name) != pin
                   for name, pin in (sources or {}).items())):
        raise ValueError("recoil session adapter differs from reviewed source or the exact unchanged engine")
    for name, pin in sources.items():
        read_pinned(path.parent / name, pin)
    registry = receipt["registryReceipt"]
    closure = json.loads(read_pinned(registry["path"], registry["sha256"]))
    if (closure.get("sourceCommit") != candidate["sourceCommit"]
            or closure.get("sourceSha256") != candidate["sourceSha256"]
            or closure.get("objectSha256") != receipt["registryObject"]["sha256"]):
        raise ValueError("recoil session adapter registry object does not bind the exact candidate source")
    read_pinned(receipt["registryObject"]["path"], receipt["registryObject"]["sha256"])
    executable = (receipt["sessionFixture"]["executable"] if receipt["schema"] == "source-recoil-paired-native-acceptance/1"
                  else receipt["nativeExecutable"])
    if artifacts["recoilSessionFixture"].resolve() != Path(executable["path"]).resolve():
        raise ValueError("recoil session adapter executable differs from its source receipt")
    read_pinned(artifacts["recoilSessionFixture"], executable["sha256"])


def run_recoil_family(cases, artifacts, candidate, cards, source_rows):
    from tcg.parity_runner import _flat_metadata

    items, row_items = _source_programs(artifacts, candidate, cards, source_rows)
    verify_recoil_session_supplement(artifacts, candidate)

    def execute(fixture, *arguments):
        return subprocess.run([str(fixture), *map(str, arguments)], capture_output=True, text=True,
                              encoding="utf-8", timeout=180, check=True).stdout

    legacy = artifacts["recoilLegacyFixture"]
    registry = json.loads(execute(legacy, "registry"))
    library = ctypes.CDLL(str(artifacts["library"]))
    for name in ("AllCard", "AllAttack"):
        getattr(library, name).argtypes = []
        getattr(library, name).restype = ctypes.c_char_p
    if registry != {"entries": json.loads(library.AllCard()), "attacks": json.loads(library.AllAttack())}:
        raise ValueError("recoil native fixture catalog differs from the actual candidate DLL")
    if registry["entries"] != list(cards.values()):
        raise ValueError("recoil catalog differs from the checked candidate registry")
    attacks = {a["attackId"]: a for a in registry["attacks"]}
    raw = json.loads(execute(legacy))
    expected = {(p, index, mode) for p, item in items.items() for index in range(len(item["attackIds"]))
                for mode in recoil_modes(item, index)}
    actual = {}
    for row in raw:
        key = (row["programSha256"], row["attackIndex"], row["mode"])
        if key not in expected or key in actual:
            raise ValueError("recoil ordinary fixture emitted duplicate or unknown commands")
        item = items[key[0]]
        if item["card"]["attacks"][key[1]]["recoil"] == 0 or key[2] not in TIE_MODES:
            actual[key] = recoil_ordinary_observation(row)
        else:
            actual[key] = None
    if actual.keys() != expected or len(actual) != 1776 or sum(v is not None for v in actual.values()) != 1624:
        raise ValueError("recoil ordinary matrix must retain all 1624 measured non-tie observations")
    session = {}
    for line in execute(artifacts["recoilSessionFixture"], artifacts["recoilSessionRecords"]).splitlines():
        row = json.loads(line)
        key = (row["programSha256"], row["mode"])
        if key in session or key[0] not in items or key[1] not in TIE_MODES:
            raise ValueError("recoil session fixture emitted duplicate or unknown commands")
        session[key] = recoil_session_observation(row)
    if session.keys() != {(program, mode) for program in items for mode in TIE_MODES}:
        raise ValueError("recoil session fixture must cover all 152 simultaneous-loss continuations")
    rules, movement = {}, {}
    if any(case["caseId"].endswith("/rules") for case in cases):
        if not {"recoilRulesFixture", "recoilMovementFixture", "recoilRulesBuildReceipt"} <= artifacts.keys():
            raise ValueError("recoil rules require both actual native fixtures and a complete source closure receipt")
        verify_recoil_supplement(artifacts, candidate)
        for kind in ("recoilRulesFixture", "recoilMovementFixture"):
            if json.loads(execute(artifacts[kind], "registry")) != registry:
                raise ValueError("recoil supplementary fixture catalog differs from the exact engine DLL")
        rules = _campaign_observations(
            artifacts["recoilRulesFixture"], RULE_RESPONSE_FIELDS,
            {(program, index, mode) for program, item in items.items()
             for index in range(len(item["attackIds"])) for mode in RULE_MODES}, True)
        movement = _campaign_observations(
            artifacts["recoilMovementFixture"], MOVEMENT_RESPONSE_FIELDS,
            {(program, mode) for program in items for mode in MOVEMENT_MODES}, False)

    def response(item, command):
        program, mode = item["programSha256"], command["mode"]
        if command["operation"] == "attack":
            return actual[(program, command["attackIndex"], mode)]
        if command["operation"] == "recoilSession":
            return {"programSha256": program, "attackIndex": command["attackIndex"],
                    "mode": mode, "stages": session[(program, mode)]}
        if command["operation"] == "recoilAttackRules":
            return rules[(program, command["attackIndex"], mode)]
        return movement[(program, mode)]

    observations, seen = [], set()
    for case in cases:
        parts = case["caseId"].split("/")
        if case["caseId"] in seen:
            raise ValueError("recoil cases contain a duplicate requirement")
        seen.add(case["caseId"])
        if len(parts) == 3 and parts[0] == "row" and parts[1] in row_items and parts[2] == "metadata_identity":
            source_id, item = parts[1], row_items[parts[1]]
            scenario = recoil_scenario(item, source_id=source_id)
            responses = [_flat_metadata(item, source_id, cards, attacks)]
        elif (len(parts) == 5 and parts[0] == "program" and parts[1] in items and parts[2] == "useActions"
              and parts[3] in ("0", "1") and parts[4] in RECOIL_FACETS):
            item, index, facet = items[parts[1]], int(parts[3]), parts[4]
            if index >= len(item["attackIds"]):
                raise ValueError("recoil action index is outside the full source program")
            scenario = recoil_scenario(item, index, facet)
            responses = [response(item, command) for command in scenario["commands"]]
        elif len(parts) == 3 and parts[0] == "program" and parts[1] in items and parts[2] == "rules":
            item = items[parts[1]]
            scenario = recoil_rules_scenario(item)
            responses = [response(item, command) for command in scenario["commands"]]
        else:
            raise ValueError("recoil rules and unmeasured facets remain held")
        if case["scenario"] != scenario or case["cabtCardId"] != item["cabtCardId"]:
            raise ValueError("recoil case differs from the original scenario and distinct measured components")
        observations.append({"caseId": case["caseId"], "cabtCardId": item["cabtCardId"],
                             "scenarioSha256": digest(scenario),
                             "observation": {"commands": scenario["commands"], "responses": responses},
                             "scenarioComponents": [{"suite": c["suite"], "scenarioSha256": digest(c)}
                                                    for c in scenario["components"]]})
    return observations
