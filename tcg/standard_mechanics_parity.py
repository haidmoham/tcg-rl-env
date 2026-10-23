import ctypes
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from tcg.live_parity import LIVE_PINS, digest, read_pinned


STANDARD_MECHANICS_FACETS = ("effect", "legal_prompts", "information_boundaries", "interactions", "rules")
STANDARD_MECHANICS_RUNTIME_SHA256 = "3d073b129c387f2785c217091b31d0bdd893e0c0aa0430538444896b7c6863a5"
FAMILIES = {
    "friends": {
        "program": "fec7f67826cb1276bcc8d6c7acbf3791487839e016e01e14a2eb49b936dd5fe9", "nativeId": 4517624,
        "receiptSha256": "ee57c482285405613bccd51dbaf452d3cb3f402655c1f43bafa99342740f8b2f",
        "commandsSha256": "10eb1f4f12e3ebdfc5d0970f7b34ad581118c065db3ee3d002c086714f267a9f",
        "sources": {
            "sv8-5_109": "bc47c682ad263cf4bb1877c1847f3c5805f8ccf08f8410cdf17dac097e960f82",
            "sv8-5_109_ph": "c585b446a58f3784fadfe6936caa17c3fc6f6c6f5409ebdc5de7921eabd064b7",
            "sv8-5_109_sph": "7a9ec4dda52db136119fc59c3211951413289393184b8cb51b998cd3e5f8df50",
            "sv8-5_137": "6fcd9311d2f205876b1f03ac45ffb9284a21a88c8f17e8a728fbac02b42d9024"},
        "fixtureSources": {
            "Commands.h": "115e6803dd0a44fb8a5e3fae464eb061f5ac1614e3af3fbe10d67715ec5e5752",
            "native.cpp": "3dfe7309bb573e794cfe396192ff820dfab81b26f613536d3dd3ebf1380f61c8",
            "native.vcxproj": "f467a494e0158fe12d46cc230331b63a07de0f5873da47d307f15b4dbb165c7f"},
        "limitations": ["Eight explicit deck/hand/turn controls per printing, not arbitrary states.",
                        "Supporter first-turn, empty-deck and second-Supporter gates only.",
                        "External card modifiers, full matches and official rulings remain unqualified."]},
    "minccino": {
        "program": "e08b792808db10d7802bcc3e82066285b24ccebed254325b6534a38849a75cec", "nativeId": 555,
        "receiptSha256": "3c6aa436253d087bbd19c314ac0356c6ef5cb6fe4fa8dfef975e77bcac0c2470",
        "commandsSha256": "d6ae0df9c61a66949457be7f49253cdd4e32bf27f977718dc7b564a3b6049ddf",
        "sources": {
            "zsv10-5_152": "92188643910c9b814382380c77f117c71ef45c00bec4c467abea7835bb27a719",
            "zsv10-5_75": "40a45f644598d02d55a1863de8c9d12632567d740058c6b3965820f68caffcef",
            "zsv10-5_75_mph": "f7ee8d00ac6ce9df5bfefe83848993d4a905fa39847cd267effc09b313917c10",
            "zsv10-5_75_ph": "976bf02a9cdc665fb5119c7c0eeea82ca45b527acec51c304609d9d06d56784b",
            "zsv10-5_75_sph": "4e41462e6e7e3c4c2e900fa0590b124a1e68927498ea34ec19fbe3d80c61cdd8"},
        "fixtureSources": {
            "Commands.h": "ad629a4eaee8102ea87f29a72ba7c18f971b8bb353ecbadce507e1d6446ff625",
            "native.cpp": "13a2f9c95d84e8d22d258c93f4d28d596469e9dce442d8b75e720713fc56559a",
            "native.vcxproj": "3a9e70edd77e5678c0f2a231449d845d4ed7b2d8b9f94b0c6359d1d9a33de3d0"},
        "limitations": ["Sixteen preplaced-actor attack controls with explicit coin choices, not arbitrary states.",
                        "Energy, turn, Bench, status, knockout and prize controls only.",
                        "Evolution, retreat, external modifiers, full matches and official rulings remain unqualified."]},
}
FRIENDS_MODES = (("regular", 12, 3, False), ("short", 2, 3, False), ("empty", 0, 0, False),
                 ("hand-only", 0, 3, False), ("one-card", 1, 0, False), ("exact-three", 3, 0, False),
                 ("large-hand", 4, 12, False), ("first-turn", 12, 3, True))
MINCCINO_MODES = ("sufficient", "deck-empty", "none", "short", "wrong", "first-turn", "bench", "asleep",
                  "paralyzed", "confused-heads", "confused-tails", "knockout", "last-prize", "no-bench", "coin-tails", "coin-mixed")


def standard_mechanics_commands(family):
    spec = FAMILIES[family]
    if family == "friends":
        return [{"programSha256": spec["program"], "cardSourceID": sid, "cabtCardId": spec["nativeId"],
                 "mode": mode, "deck": deck, "hand": hand, "enemyDeck": 11, "enemyHand": 2,
                 "prizes": 3, "firstTurn": first, "seed": 17}
                for sid in sorted(spec["sources"]) for mode, deck, hand, first in FRIENDS_MODES]
    return [{"programSha256": spec["program"], "cardSourceID": "zsv10-5_75", "actionIndex": 0, "mode": mode,
             "deck": 0 if mode == "deck-empty" else 4, "prizes": 1 if mode == "last-prize" else 3, "seed": 17}
            for mode in MINCCINO_MODES]


def standard_mechanics_scenario(family, facet):
    if facet not in STANDARD_MECHANICS_FACETS:
        raise ValueError("Unsupported finite Standard mechanics facet")
    spec = FAMILIES[family]
    return {"suite": "standard-mechanics-rules/1" if facet == "rules" else "standard-mechanics-matrix/1",
            "family": family, "programSha256": spec["program"], "actionIndex": 0, "facet": facet,
            "criterion": "Every declared finite source/native control matches; omitted states remain unqualified.",
            "commands": standard_mechanics_commands(family), "limitations": spec["limitations"]}


def standard_mechanics_cases():
    return [{"caseId": f"program/{spec['program']}/rules" if facet == "rules"
                        else f"program/{spec['program']}/useActions/0/{facet}",
             "cabtCardId": spec["nativeId"], "scenario": standard_mechanics_scenario(family, facet)}
            for family, spec in FAMILIES.items() for facet in STANDARD_MECHANICS_FACETS]


def _row_key(row):
    return row["programSha256"], row["cardSourceID"], row.get("actionIndex"), row["mode"]


def standard_mechanics_expected(family, facet, captured_payload):
    scenario = standard_mechanics_scenario(family, facet)
    rows = {_row_key(row): row for row in captured_payload["responses"]}
    keys = [_row_key(command) for command in scenario["commands"]]
    if len(rows) != len(captured_payload["responses"]) or set(rows) != set(keys):
        raise ValueError("Captured Standard mechanics rows differ from declared command identities")
    return {"commands": scenario["commands"], "responses": [rows[key] for key in keys],
            "limitations": scenario["limitations"]}


def standard_mechanics_source_payload(family, receipt_path):
    receipt = json.loads(read_pinned(receipt_path, FAMILIES[family]["receiptSha256"]))
    return json.loads(read_pinned(receipt["cachedSourceObservationsPath"], receipt["cachedSourceObservationsSha256"]))


def standard_mechanics_configuration(replay_root, runtime_closure):
    root, runtime = Path(replay_root).resolve(strict=True), Path(runtime_closure).resolve(strict=True)
    read_pinned(runtime, STANDARD_MECHANICS_RUNTIME_SHA256)
    result = {"runtimeClosurePath": str(runtime), "runtimeClosureSha256": STANDARD_MECHANICS_RUNTIME_SHA256}
    for family, spec in FAMILIES.items():
        receipt_path = root / family / "receipt.json"
        receipt = json.loads(read_pinned(receipt_path, spec["receiptSha256"]))
        for kind, path, pin in (("Receipt", receipt_path, spec["receiptSha256"]),
                                ("Fixture", Path(receipt["fixturePath"]), receipt["fixtureSha256"])):
            read_pinned(path, pin)
            result[family + kind + "Path"], result[family + kind + "Sha256"] = str(path), pin
    return result


def _verify_runtime(artifacts):
    closure = json.loads(read_pinned(artifacts["standardMechanicsRuntimeClosure"], STANDARD_MECHANICS_RUNTIME_SHA256))
    if (closure["schema"] != "standard-mechanics-captured-runtime-closure/1"
            or closure["sourceRecaptures"] != 0 or closure["originalLoadedAssemblies"] != 31):
        raise ValueError("Standard mechanics captured runtime closure changed")
    read_pinned(closure["fullCapturedRuntimeReceiptPath"], closure["fullCapturedRuntimeReceiptSha256"])
    for path, pin in closure["files"].items():
        read_pinned(path, pin)
    return closure


def _verify_family(family, artifacts, candidate, source_rows, closure):
    spec = FAMILIES[family]
    prefix = "standardMechanics" + family.title()
    receipt_path = Path(artifacts[prefix + "Receipt"])
    receipt = json.loads(read_pinned(receipt_path, spec["receiptSha256"]))
    fixture = Path(artifacts[prefix + "Fixture"])
    if (receipt["schema"] != "current-standard-native-candidate-replay/1" or receipt["family"] != family
            or receipt["sourceCommit"] != candidate["sourceCommit"] or receipt["sourceSha256"] != candidate["sourceSha256"]
            or receipt["librarySha256"] != candidate["librarySha256"] or not receipt["successful"]
            or not receipt["actualDllCatalogEqual"] or receipt["sourceRecaptured"]
            or receipt["fixtureSourceSha256"] != spec["fixtureSources"]
            or fixture.resolve() != Path(receipt["fixturePath"]).resolve()
            or receipt["selectedNativeId"] != spec["nativeId"]
            or receipt["scenarioCommandsSha256"] != spec["commandsSha256"]
            or digest(standard_mechanics_commands(family)) != spec["commandsSha256"]):
        raise ValueError("Standard mechanics fixture differs from the reviewed current candidate and command matrix")
    read_pinned(artifacts["library"], candidate["librarySha256"])
    read_pinned(fixture, receipt["fixtureSha256"])
    for name, pin in spec["fixtureSources"].items():
        read_pinned(receipt_path.parent / name, pin)
    root = Path(candidate["sourceRoot"]).resolve(strict=True)
    if (subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() != candidate["sourceCommit"]
            or subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()):
        raise ValueError("Standard mechanics candidate source is not frozen")
    for name, pin in candidate["sourceSha256"].items():
        read_pinned(root / name, pin)
    registry = json.loads(read_pinned(receipt["registryReceiptPath"], receipt["registryReceiptSha256"]))
    if (registry["sourceCommit"] != candidate["sourceCommit"] or registry["sourceSha256"] != candidate["sourceSha256"]
            or registry["objectSha256"] != receipt["registryObjectSha256"]):
        raise ValueError("Standard mechanics registration object changed source closure")
    read_pinned(registry["objectPath"], registry["objectSha256"])
    manifest = json.loads(read_pinned(receipt["inputManifestPath"], receipt["inputManifestSha256"]))
    if manifest["commands"] != standard_mechanics_commands(family):
        raise ValueError("Standard mechanics captured commands changed")
    reference = closure["families"][family]
    if (receipt["referenceReceiptSha256"] != reference["referenceReceiptSha256"]
            or Path(receipt["referenceReceiptPath"]).resolve() != Path(reference["referenceReceiptPath"]).resolve()):
        raise ValueError("Standard mechanics captured observation lineage changed")
    original = json.loads(read_pinned(receipt["referenceReceiptPath"], receipt["referenceReceiptSha256"]))
    read_pinned(receipt["cachedSourceObservationsPath"], receipt["cachedSourceObservationsSha256"])
    if original["outputs"]["oracle.stdout"] != receipt["cachedSourceObservationsSha256"]:
        raise ValueError("Standard mechanics captured observations differ from actual runtime receipt")
    source_ids = {}
    all_program_sources = {}
    for card, _, row_hash, program in source_rows:
        source_ids[card["cardSourceID"]] = (row_hash, program, len(card.get("useActions", [])))
        if program == spec["program"]:
            all_program_sources[card["cardSourceID"]] = row_hash
    if len(source_ids) != LIVE_PINS.rows or all_program_sources != spec["sources"]:
        raise ValueError("Standard mechanics source program/printing inventory changed")
    if any(source_ids[sid] != (pin, spec["program"], 1) for sid, pin in spec["sources"].items()):
        raise ValueError("Standard mechanics raw printing or action ordinal changed")
    if receipt["programs"] != [{"programSha256": spec["program"], "cardSourceIDs": sorted(spec["sources"]), "cardSha256": spec["sources"]}]:
        raise ValueError("Standard mechanics fixture printing bindings changed")
    return receipt, fixture


def _normalize(family, original):
    row = dict(original)
    if family == "friends":
        owner, opponent, enemy_owner, enemy_opponent = [row.pop(name) for name in ("ownerView", "opponentView", "enemyOwnerView", "enemyOpponentView")]
        row.pop("prompt")
        row.pop("rawZones")
        row.pop("traitConsumerLillie")
        row["privacy"] = {"ownerHandKnown": owner["hand"] is not None and all(card.get("id", 0) > 0 for card in owner["hand"]),
                          "opponentHandHidden": opponent["hand"] is None, "enemyHandHidden": enemy_opponent["hand"] is None,
                          "enemyOwnHandKnown": enemy_owner["hand"] is not None and all(card.get("id", 0) > 0 for card in enemy_owner["hand"]),
                          "deckHiddenFromBoth": all("deck" not in view for view in (owner, opponent, enemy_owner, enemy_opponent)),
                          "discardPublic": all(card.get("id", 0) > 0 for card in opponent["discard"] + enemy_opponent["discard"])}
    else:
        owner, opponent = row.pop("ownerView"), row.pop("opponentView")
        row.pop("trace")
        one, hidden = owner["players"][0], opponent["players"][0]
        row["privacy"] = {"ownerHandKnown": isinstance(one["hand"], list) and all(card.get("id", 0) > 0 for card in one["hand"]),
                          "opponentHandHidden": hidden["hand"] is None,
                          "deckHiddenFromBoth": all("deck" not in player for player in owner["players"] + opponent["players"]),
                          "activePublic": bool(one["active"]) and bool(hidden["active"]) and all(card.get("id", 0) > 0 for card in hidden["active"]),
                          "discardPublic": all(card.get("id", 0) > 0 for player in owner["players"] + opponent["players"] for card in player["discard"])}
        row.pop("nativeCardId")
    return row


def run_standard_mechanics(cases, artifacts, candidate, cards, source_rows):
    from tcg.candidate_replay import candidate_replay, replay_fixture
    replay = candidate_replay(artifacts, candidate)
    prepared = {case["caseId"]: case for case in standard_mechanics_cases()}
    seen, grouped = set(), {}
    for case in cases:
        if case["caseId"] in seen or case["caseId"] not in prepared or any(case[key] != prepared[case["caseId"]][key] for key in ("scenario", "cabtCardId")):
            raise ValueError("Standard mechanics case differs from its finite source program/scenario/native identity")
        seen.add(case["caseId"])
        grouped.setdefault(case["scenario"]["family"], []).append(case)
    closure = _verify_runtime(artifacts)
    library = ctypes.CDLL(str(artifacts["library"]))
    catalog = {}
    for export, name in (("AllCard", "entries"), ("AllAttack", "attacks")):
        function = getattr(library, export)
        function.argtypes = []
        function.restype = ctypes.c_char_p
        catalog[name] = json.loads(function())
    if catalog["entries"] != list(cards.values()):
        raise ValueError("Standard mechanics current DLL catalog changed after caller initialization")
    observations = []
    for family, selected_cases in grouped.items():
        receipt, fixture = _verify_family(family, replay["parentArtifacts"] if replay else artifacts,
                                          replay["parentCandidate"] if replay else candidate, source_rows, closure)
        if replay:
            fixture = artifacts["standardMechanics" + family.title() + "Fixture"]
            replay_fixture(replay, "standardMechanics." + family + "Fixture", fixture)

        def execute(*arguments):
            return subprocess.run([str(fixture), *arguments], capture_output=True, timeout=60, check=True).stdout

        if json.loads(execute("registry")) != catalog:
            raise ValueError("Fresh Standard mechanics fixture catalog differs from current DLL")
        raw_bytes = execute()
        if sha256(raw_bytes).hexdigest() != receipt["outputs"]["native-raw.stdout"]:
            raise ValueError("Fresh Standard mechanics raw prompts/views differ from reviewed current-candidate replay")
        raw = json.loads(raw_bytes)
        if family == "minccino":
            if len(raw) != 32 or {row["nativeCardId"] for row in raw} != {555, 18487}:
                raise ValueError("Minccino fixture omitted selected or historical control candidate")
            raw = [row for row in raw if row["nativeCardId"] == FAMILIES[family]["nativeId"]]
        actual = [_normalize(family, row) for row in raw]
        for case in selected_cases:
            scenario = case["scenario"]
            observation = standard_mechanics_expected(family, scenario["facet"], {"responses": actual})
            observations.append({"caseId": case["caseId"], "cabtCardId": case["cabtCardId"], "scenarioSha256": digest(scenario),
                                 "observation": observation,
                                 "scenarioComponents": [{"suite": scenario["suite"], "scenarioSha256": digest(scenario)}]})
    return observations
