"""Finite attack legality observations with independently replayed native requests."""

import json
import os
from pathlib import Path
import subprocess
import tempfile

from tcg.live_parity import digest, read_pinned


PACKET_SHA256 = "5e0a770a0519cc36b91728c5c85101510b370222cfb482f74b080f8cff5e4dd1"
MEASUREMENT_SHA256 = "201dc69b2f7e531419bd76f91f9d913a61c568de906148003247510df03c875d"
SUITE = "source-verified-no-effect-legal-prompts/1"
MODES = ("heads", "none", "first-turn", "asleep", "paralyzed")
NATIVE_ZONES = {0: "none", 1: "deck", 2: "hand", 3: "discard", 4: "active", 5: "bench",
                6: "prize", 7: "stadium", 8: "energy", 9: "tool", 10: "preEvolution",
                12: "pending", 13: "playing", 14: "deck"}
SOURCE_ZONES = {2: "stadium", 7: "deck", 8: "deck", 9: "discard", 10: "discard",
                11: "hand", 12: "hand", 13: "bench", 14: "bench", 15: "active",
                16: "active", 17: "lostZone", 18: "lostZone", 19: "prize", 20: "prize",
                5: "pending", 6: "pending", 21: "pending", 22: "pending"}


def _read(path, pin):
    return json.loads(read_pinned(path, pin))


def _packet(packet_path):
    return _read(packet_path, PACKET_SHA256)


def standard_legal_prompts_configuration(packet_path):
    packet_path = Path(packet_path).resolve(strict=True)
    packet = _packet(packet_path)
    return {"packetPath": str(packet_path), "packetSha256": PACKET_SHA256,
            "fixturePath": packet["fixturePath"], "fixtureSha256": packet["fixtureSha256"]}


def standard_legal_prompts_cases(packet_path):
    packet_path = Path(packet_path)
    packet = _packet(packet_path)
    return _read(packet_path.parent / "cases.json", packet["outputs"]["cases.json"])


def standard_legal_prompts_expected(packet_path, case_id):
    packet_path = Path(packet_path)
    packet = _packet(packet_path)
    return _read(packet_path.parent / "oracles.json", packet["outputs"]["oracles.json"])[case_id]


def _records(data, wanted):
    rows = [json.loads(line) for line in data.splitlines()]
    by_key = {row["key"]: row for row in rows}
    if len(by_key) != len(rows) or set(by_key) != wanted:
        raise ValueError("Attack legality replay omitted or duplicated an actual request")
    return by_key


def _state(snapshot, side, formats, actor_id):
    physical = []
    for card in snapshot["physical"]:
        if side == "source":
            zone = SOURCE_ZONES[card["position"]]
            if card.get("parent"):
                form = formats.get(card["sourceID"])
                zone = "energy" if form in (1, 2) else "tool" if form == 6 else "preEvolution"
        else:
            zone = NATIVE_ZONES[card["area"]]
        physical.append({"label": card["label"], "zone": zone, "damage": card["damage"]})
    actors = [card for card in snapshot["physical"] if card["label"] == "actor"]
    if len(actors) != 1:
        raise ValueError("Attack legality snapshot must contain exactly one physical actor")
    actor = actors[0]
    if side == "source":
        if actor["sourceID"] != actor_id:
            raise ValueError("Captured attack actor printing changed")
        board = snapshot["board"]
        turns = board["boardEntity"]["entityMetaData"]["TotalTurnCount"]
        if len(turns) != 1:
            raise ValueError("Captured attack requires one observed total turn count")
        conditions = []
        for name in ("p1Active", "p2Active"):
            active = board[name]
            found = [card for card in snapshot["physical"] if active
                     and card["sourceID"] == active["cardSourceID"]
                     and card["position"] == active["currentPos"] and not card.get("parent")]
            if len(found) != 1:
                raise ValueError("Captured active condition identity is ambiguous")
            conditions.append([bool(value) for value in found[0]["status"]])
        player, terminal, turn = snapshot["currentPlayer"] - 1, snapshot["gameOver"], turns[0]["value"] + 1
        actor_player, actor_zone = 0, "active" if actor["position"] == 15 else str(actor["position"])
    else:
        if actor["cardId"] != actor_id:
            raise ValueError("Native attack actor definition changed")
        view = snapshot["views"]["owner"]
        conditions = [[bool(player.get(key, False)) for key in
                       ("asleep", "paralyzed", "confused", "poisoned", "burned")]
                      for player in view["players"]]
        player, terminal, turn = snapshot["currentPlayer"], snapshot["gameResult"] != 0, view["turn"]
        actor_player, actor_zone = actor["owner"], "active" if actor["area"] == 4 else str(actor["area"])
    return {"currentPlayer": player, "terminal": terminal, "turnNumber": turn,
            "physical": sorted(physical, key=lambda card: (card["label"], card["zone"], card["damage"])),
            "activeConditions": conditions, "actor": {"label": "actor", "player": actor_player, "zone": actor_zone}}


def _normalize(row, side, command, formats):
    if "error" in row or row["mode"] != command["mode"] or row["choices"]:
        raise ValueError("Attack legality requires successful replay and exact empty follow-up prompts")
    if side == "source":
        if (row["sourceID"] != command["cardSourceID"] or row["channel"] != "useActions"
                or row["actionIndex"] != command["actionIndex"]
                or row["requestedActionGuid"] != command["actionGuid"]
                or row["requestedActor"] != row["before"]["board"]["p1Active"]["entityID"]
                or row["status"] != ("executed" if row["legal"] else "blocked")):
            raise ValueError("Captured requested attack or actor identity changed")
        actor_id = command["cardSourceID"]
    else:
        if (row["requestedNativeCardId"] != command["cabtCardId"]
                or row["requestedNativeAttackId"] != command["attackId"]
                or row["requestedActorSerial"] != row["before"]["views"]["owner"]["players"][0]["active"][0]["serial"]):
            raise ValueError("Native requested attack or actor identity changed")
        prompt = row["initialPrompt"]
        available = any(option.get("type") == 13 and option.get("attackId") == command["attackId"]
                        for option in prompt["option"])
        if prompt["type"] != 0 or prompt["context"] != 0 or available != row["legal"]:
            raise ValueError("Native legality differs from its actual main prompt")
        actor_id = command["cabtCardId"]
    if row["legal"] != (command["mode"] == "heads") or row["completed"] != row["legal"]:
        raise ValueError("Attack legality restriction or positive continuation disagrees")
    return {"mode": row["mode"], "legal": row["legal"], "completed": row["completed"],
            "coins": row["coins"], "postSelectionPromptCount": 0,
            **{stage: _state(row[stage], side, formats, actor_id) for stage in ("before", "after")}}


def _execute(command):
    flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    return subprocess.run(command, capture_output=True, check=True, timeout=60, creationflags=flags).stdout


def run_standard_legal_prompts(cases, artifacts, candidate, cards, source_rows):
    from tcg.candidate_replay import candidate_replay, replay_fixture
    replay = candidate_replay(artifacts, candidate)
    current_artifacts = artifacts
    if replay:
        artifacts, candidate = replay["parentArtifacts"], replay["parentCandidate"]
    packet_path = artifacts["standardLegalPromptsPacket"]
    packet = _packet(packet_path)
    if (packet["schema"] != "source-verified-no-effect-legal-prompts-packet/1"
            or packet["sourceCommit"] != candidate["sourceCommit"]
            or packet["librarySha256"] != candidate["librarySha256"]
            or packet["engineSourceSha256"] != candidate["sourceSha256"]
            or packet["caseCount"] != 23 or packet["heldCases"] != 1
            or packet["facet"] != "legal_prompts" or packet["otherFacetsAdded"] != 0
            or packet["actualDllCatalogEqual"] is not True):
        raise ValueError("Attack legality packet does not bind this exact candidate and finite domain")
    for mapping in (packet["capturedRuntime"], packet["capturedBundleClosure"], packet["fixtureSources"]):
        for path, pin in mapping.items():
            read_pinned(path, pin)
    for relative, pin in packet["engineSourceSha256"].items():
        read_pinned(Path(candidate["sourceRoot"]) / relative, pin)
    read_pinned(candidate["libraryPath"], packet["librarySha256"])
    fixture = Path(artifacts["standardLegalPromptsFixture"])
    if fixture.resolve() != Path(packet["fixturePath"]).resolve():
        raise ValueError("Attack legality fixture differs from its measured native build")
    read_pinned(fixture, packet["fixtureSha256"])
    measurement_path = Path(packet["measurementReceiptPath"])
    measurement = _read(measurement_path, MEASUREMENT_SHA256)
    if packet["measurementReceiptSha256"] != MEASUREMENT_SHA256:
        raise ValueError("Attack legality measurement is not reviewed")
    measured = measurement_path.parent
    registry = _read(packet["registryReceiptPath"], packet["registryReceiptSha256"])
    execution = _read(measured / "registry-execution.json", measurement["outputs"]["registry-execution.json"])
    if (registry["sourceCommit"] != candidate["sourceCommit"]
            or registry["sourceSha256"] != candidate["sourceSha256"]
            or execution["sourceCommit"] != candidate["sourceCommit"]
            or execution["sourceAndObjectRetained"] is not True or execution["exitCode"] != 0
            or execution["objectSha256"] != registry["objectSha256"]
            or execution["registryReceiptSha256"] != packet["registryReceiptSha256"]):
        raise ValueError("Attack legality fixture registry object lacks exact candidate closure")
    read_pinned(registry["objectPath"], registry["objectSha256"])
    for name in ("source-commands.json", "source.stdout", "native-commands.tsv", "Oracle.cs",
                 "legal_prompts.cpp", "legal_prompts.vcxproj"):
        read_pinned(measured / name, measurement["outputs"][name])
    for name, pin in packet["outputs"].items():
        read_pinned(Path(packet_path).parent / name, pin)
    if replay:
        fixture = current_artifacts["standardLegalPromptsFixture"]
        replay_fixture(replay, "standardLegalPrompts.fixture", fixture)
    catalog = json.loads(_execute([str(fixture), "registry"]))
    measured_catalog = _read(measured / "native-registry.stdout", measurement["outputs"]["native-registry.stdout"])
    retained = (catalog == measured_catalog if not replay else all(
        {row[key]: row for row in catalog[name]}.get(row[key]) == row
        for name, key in (("entries", "cardId"), ("attacks", "attackId")) for row in measured_catalog[name]))
    if not retained or {row["cardId"]: row for row in catalog["entries"]} != cards:
        raise ValueError("Fresh attack legality registry differs from the actual candidate DLL")
    source_commands = json.loads((measured / "source-commands.json").read_bytes())
    by_command = {command["key"]: command for command in source_commands}
    if len(by_command) != 120 or len(source_commands) != 120:
        raise ValueError("Captured attack legality requires all 120 unique requests")
    source = _records((measured / "source.stdout").read_bytes(), set(by_command))
    templates = {case["caseId"]: case for case in standard_legal_prompts_cases(packet_path)}
    if len(templates) != 23 or len({case["caseId"] for case in cases}) != len(cases):
        raise ValueError("Attack legality case domain is duplicated or changed")
    rows = {card["cardSourceID"]: (card, row_hash, program) for card, _, row_hash, program in source_rows}
    formats = {key: card.get("cardFormat", 0) for key, (card, _, _) in rows.items()}
    requests, expected = {}, {}
    for case in cases:
        template = templates.get(case["caseId"])
        if template is None or any(case.get(key) != template[key] for key in ("caseId", "cabtCardId", "scenario")):
            raise ValueError("Attack legality case differs from its measured finite scenario")
        commands = case["scenario"]["commands"]
        if case["scenario"]["suite"] != SUITE or tuple(command["mode"] for command in commands) != MODES:
            raise ValueError("Attack legality scenario must preserve all five actual modes")
        responses = []
        for command in commands:
            key = f'source/{command["programSha256"]}/useActions/{command["actionIndex"]}/{command["mode"]}'
            captured = by_command[key]
            card, row_hash, program = rows[command["cardSourceID"]]
            action = card["useActions"][command["actionIndex"]]
            attack_index = sum(action["actionType"] == 1 for action in card["useActions"][:command["actionIndex"]])
            if (program != command["programSha256"] or row_hash != captured["sourceRowSha256"]
                    or digest(action) != captured["actionSha256"] or action["actionGuid"] != command["actionGuid"]
                    or command["cardSourceID"] != captured["sourceID"]
                    or action["actionType"] != 1
                    or cards[command["cabtCardId"]]["attacks"][attack_index] != command["attackId"]):
                raise ValueError("Attack legality raw printing, program or ordered native attack changed")
            requests[key] = command
            responses.append(_normalize(source[key], "source", command, formats))
        expected[case["caseId"]] = {"commands": commands, "responses": responses}
        if expected[case["caseId"]] != standard_legal_prompts_expected(packet_path, case["caseId"]):
            raise ValueError("Attack legality oracle no longer derives from its actual captured observations")
    with tempfile.TemporaryDirectory(prefix="cabt-legal-prompts-") as temporary:
        request_path = Path(temporary) / "requests.tsv"
        request_path.write_text("".join(f'{key}\t{command["cabtCardId"]}\tattack\t{command["attackId"]}\tattack\t{command["mode"]}\n'
                                        for key, command in requests.items()), encoding="utf-8")
        native = _records(_execute([str(fixture), str(request_path)]), set(requests))
    observations = []
    for case in cases:
        scenario = case["scenario"]
        responses = [_normalize(native[f'source/{command["programSha256"]}/useActions/{command["actionIndex"]}/{command["mode"]}'],
                                "native", command, formats) for command in scenario["commands"]]
        observation = {"commands": scenario["commands"], "responses": responses}
        if observation != expected[case["caseId"]]:
            raise ValueError("Fresh native attack legality differs from the actual captured observation")
        observations.append({"caseId": case["caseId"], "cabtCardId": case["cabtCardId"],
                             "scenarioSha256": digest(scenario), "observation": observation,
                             "scenarioComponents": [{"suite": SUITE, "scenarioSha256": digest(scenario)}]})
    return observations
