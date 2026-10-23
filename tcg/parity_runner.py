"""Execute reviewed native fixtures and validate source-bound observations."""

import ctypes
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

from tcg.live_parity import digest, read_pinned
from tcg.energy_parity import FAIRY_ENERGY_DOMAIN, run_energy_family
from tcg.flat_rules_parity import run_flat_rules_family
from tcg.movement_parity import run_movement_family
from tcg.evolved_parity import EXTRA_EVOLVED_DOMAIN, run_evolved_family
from tcg.draw_parity import run_draw_family
from tcg.dragapult_parity import run_dragapult_family, DRAGAPULT_FACETS
from tcg.recoil_parity import run_recoil_family
from tcg.flat_draw_parity import run_flat_draw_family, FLAT_DRAW_FACETS
from tcg.native_metadata_parity import run_native_metadata
from tcg.standard_mechanics_parity import run_standard_mechanics
from tcg.standard_legal_prompts_parity import run_standard_legal_prompts
from tcg.standard_effect_parity import run_standard_effects
from tcg.standard_missing_pool_parity import run_standard_missing_pool
from tcg.standard_privacy_parity import run_standard_privacy


GWYNN_PROGRAM = "ed5224801534fdd7580ea904c8322b94fff86f7ff8a724e0fc93fef5146b82d3"
GWYNN_CASE = f"program/{GWYNN_PROGRAM}/useActions/0/legal_prompts"
GWYNN_COMMANDS = [{"operation": "canPlay", "cardSourceID": "me5_78", "deck": deck} for deck in (0, 1)]
RED_CARD_PROGRAM = "cca6a448606b908b0d3dac0bcd70876d0b94a7bbc372511889320386b6e61269"
GLADION_PROGRAM = "2bce9505cc36589424658770d8232ea2399fe7172c42b088816478bd54821ef8"
SOURCE_NATIVE_SCOPE = "source_native_fixture_with_verified_candidate_dll_catalog"
TRANSITION_FACETS = ("effect", "legal_prompts", "information_boundaries", "interactions")
PRIVACY_SHAPE = {"ownerHandKnown": bool, "opponentHandHidden": bool,
                 "deckHiddenFromBoth": bool, "discardedTrainerPublic": bool}


@dataclass(frozen=True)
class FixtureComponent:
    suite: str
    commands: list
    response_shape: dict
    echoed_fields: tuple

    @property
    def scenario(self):
        return {"suite": self.suite, "commands": self.commands}

    @property
    def arguments(self):
        return (f"--observe-{self.suite.split('/')[0]}",)


GWYNN_PLAYABILITY = FixtureComponent(
    "gwynn-playability/1", GWYNN_COMMANDS, {"deck": int, "canPerform": bool}, ("deck",))
GWYNN_TRANSITIONS = FixtureComponent(
    "gwynn-transitions/1",
    [{"operation": "playAndSelect", "cardSourceID": "me5_78", "deck": deck,
      "eligible": 3, "selected": selected} for deck, selected in ((9, 1), (9, 2), (1, 2))],
    {"deck": int, "selected": int, "min": int, "max": int, "options": int,
     "zeroValid": bool, "threeValid": bool, "selectedValid": bool, "hand": int,
     "remainingDeck": int, "discardedPokemon": int, "trainerDiscarded": bool,
     "secondSupporter": bool, "gameOver": bool, "status": str, "privacy": PRIVACY_SHAPE},
    ("deck", "selected"))
RED_CARD_TRANSITIONS = FixtureComponent(
    "red-card-transitions/1",
    [{"operation": "play", "cardSourceID": "me4_82", "opponentDeck": deck,
      "opponentHand": hand, "opponentPrizes": prizes,
      "actorProbe": "gwynn-with-dreepy-and-three-card-deck"}
     for deck, hand, prizes in ((6, 4, 3), (0, 1, 3), (0, 2, 3), (0, 4, 3), (6, 4, 4), (6, 0, 3))],
    {"opponentDeck": int, "opponentHand": int, "opponentPrizes": int, "canPerform": bool,
     "hand": int, "remainingDeck": int, "originalHandInDeck": int, "drawnOriginalDeck": int,
     "trainerDiscarded": bool, "opponentHandHidden": bool, "opponentDeckHidden": bool,
     "ownerHandKnown": bool, "gameOver": bool, "promptAfterPlay": bool, "supporterPlayableAfter": bool},
    ("opponentDeck", "opponentHand", "opponentPrizes"))
GLADION_TRANSITIONS = FixtureComponent(
    "gladion-transitions/1",
    [{"operation": "playAndAttack", "cardSourceID": "me5_77", "scenario": scenario,
      "combatFixture": "synthetic-fire50-grass1000-weakfire"}
     for scenario in ("ordinary", "rule-box", "late-entry", "extra-hand")],
    {"scenario": str, "canPerform": bool, "trainerDiscarded": bool,
     "bonusActiveAfterPlay": bool, "damage": int, "bonusActiveAfterAttack": bool,
     "currentPlayer": int, "gameOver": bool, "promptAfterPlay": bool, "privacy": PRIVACY_SHAPE},
    ("scenario",))


@dataclass(frozen=True)
class FixtureHandler:
    program: str
    card_id: int
    facets: tuple
    components: tuple

    @property
    def commands(self):
        return [command for component in self.components for command in component.commands]

    @property
    def case_ids(self):
        return {f"program/{self.program}/useActions/0/{facet}" for facet in self.facets}


HANDLERS = {
    "gwynn-playability/1": FixtureHandler(
        GWYNN_PROGRAM, 1428, ("legal_prompts",), (GWYNN_PLAYABILITY,)),
    "gwynn-playability-and-transitions/1": FixtureHandler(
        GWYNN_PROGRAM, 1428, ("legal_prompts",), (GWYNN_PLAYABILITY, GWYNN_TRANSITIONS)),
    "gwynn-transitions/1": FixtureHandler(
        GWYNN_PROGRAM, 1428, ("effect", "information_boundaries", "interactions"), (GWYNN_TRANSITIONS,)),
    "red-card-transitions/1": FixtureHandler(
        RED_CARD_PROGRAM, 1413, TRANSITION_FACETS, (RED_CARD_TRANSITIONS,)),
    "gladion-transitions/1": FixtureHandler(
        GLADION_PROGRAM, 1426, TRANSITION_FACETS, (GLADION_TRANSITIONS,)),
}


@dataclass(frozen=True)
class NativeFixtureExecution:
    cases: list
    provenance: dict


def execution_manifest(manifest):
    return {"schema": "live-parity-execution/1", "engine": manifest["engine"],
            "sourcePins": manifest["sourcePins"], "requiredCasesSha256": manifest["requiredCasesSha256"],
            "cases": [{key: case[key] for key in ("caseId", "scenario", "cabtCardId")}
                      for case in manifest["cases"]]}


def verify_candidate_build(candidate, manifest_root, engine):
    root = Path(manifest_root)
    receipt = json.loads(read_pinned(root / candidate["receiptPath"], candidate["receiptSha256"]))
    if receipt.get("schema") != "live-parity-candidate-build/1":
        raise ValueError("candidate receipt schema is unsupported")
    for key in ("sourceCommit", "sourceSha256", "librarySha256", "fixtureSha256"):
        if candidate.get(key) != receipt.get(key):
            raise ValueError(f"candidate build differs from its receipt: {key}")
    if (candidate["sourceCommit"] != engine["sourceCommit"]
            or candidate["librarySha256"] != engine["librarySha256"]
            or receipt.get("registrySha256") != engine["registrySha256"]):
        raise ValueError("candidate build does not bind the requested engine")
    source_root = (root / candidate["sourceRoot"]).resolve(strict=True)
    if source_root != (root / receipt["sourceRoot"]).resolve(strict=True):
        raise ValueError("candidate source root differs from its build receipt")
    source_hashes = candidate["sourceSha256"]
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("candidate source files must be pinned")
    commit = subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    if commit != candidate["sourceCommit"]:
        raise ValueError("candidate source HEAD differs from pinned commit")
    unchanged = subprocess.run(["git", "-C", str(source_root), "diff", "--quiet", "HEAD", "--"],
                               capture_output=True, timeout=15)
    if unchanged.returncode != 0:
        raise ValueError("candidate source differs from its pinned commit")
    tracked = subprocess.run(["git", "-C", str(source_root), "ls-files", "-z"],
                             capture_output=True, text=True, timeout=15, check=True).stdout
    if set(tracked.rstrip("\0").split("\0")) != set(source_hashes):
        raise ValueError("candidate receipt must pin every tracked source file")
    for relative, expected in source_hashes.items():
        source = (source_root / relative).resolve(strict=True)
        if not source.is_relative_to(source_root):
            raise ValueError("candidate source file escapes its source root")
        read_pinned(source, expected)
    for relative in ("All.h", "Api.h", "Export.cpp"):
        if relative not in source_hashes:
            raise ValueError(f"candidate build omits required native source: {relative}")
    if not {"experiments/standard_trainers.cpp", "tests/standard_trainers.cpp"} & source_hashes.keys():
        raise ValueError("candidate build omits reviewed native trainer experiment source")
    artifacts = {}
    for kind in ("library", "fixture"):
        path = (root / candidate[f"{kind}Path"]).resolve(strict=True)
        if path != (root / receipt[f"{kind}Path"]).resolve(strict=True):
            raise ValueError(f"candidate {kind} path differs from its build receipt")
        read_pinned(path, candidate[f"{kind}Sha256"])
        artifacts[kind] = path
    families = candidate.get("fixtureFamilies", {})
    if families != receipt.get("fixtureFamilies", {}) or set(families) - {
            "flatAttack", "basicEnergy", "fairyEnergy", "flatRules", "movement", "evolved", "extraEvolved", "draw", "dragapult", "recoil", "flatDraw", "nativeMetadata", "standardMechanics", "standardLegalPrompts", "standardEffects", "standardMissingPool", "standardPrivacy", "candidateReplay"}:
        raise ValueError("candidate fixture families differ from the build receipt")
    if "candidateReplay" in families:
        family = families["candidateReplay"]
        if set(family) != {"receiptPath", "receiptSha256"}:
            raise ValueError("Candidate replay requires its pinned supplemental receipt")
        path = (root / family["receiptPath"]).resolve(strict=True)
        read_pinned(path, family["receiptSha256"])
        artifacts["candidateReplayReceipt"] = path
    if "flatAttack" in families:
        family = families["flatAttack"]
        if set(family) != {"manifestPath", "manifestSha256", "fixturePath", "fixtureSha256"}:
            raise ValueError("flat attack family requires pinned manifest and fixture")
        for kind in ("manifest", "fixture"):
            path = (root / family[f"{kind}Path"]).resolve(strict=True)
            read_pinned(path, family[f"{kind}Sha256"])
            artifacts[f"flatAttack{kind.title()}"] = path
        required = {"GeneratedFlatCards.h", "experiments/flat_attack_family/cabt.cpp",
                    "experiments/flat_attack_family/FixturePrograms.h"}
        if not required <= source_hashes.keys():
            raise ValueError("candidate source omits the reviewed flat attack experiment")
    for family_name, directory in (("basicEnergy", "basic_energy"), ("fairyEnergy", "fairy_energy")):
        if family_name not in families:
            continue
        family = families[family_name]
        if set(family) != {"manifestPath", "manifestSha256", "fixturePath", "fixtureSha256"}:
            raise ValueError("Basic Energy family requires pinned manifest and fixture")
        for kind in ("manifest", "fixture"):
            path = (root / family[f"{kind}Path"]).resolve(strict=True)
            read_pinned(path, family[f"{kind}Sha256"])
            artifacts[f"{family_name}{kind.title()}"] = path
        if not {f"experiments/{directory}/cabt.cpp", f"experiments/{directory}/EnergyPrograms.h"} <= source_hashes.keys():
            raise ValueError("candidate source omits the reviewed Basic Energy experiment")
        if family_name == "fairyEnergy" and "GeneratedFairyEnergy.h" not in source_hashes:
            raise ValueError("candidate source omits the reviewed Fairy Energy registration")
    if "flatRules" in families:
        family = families["flatRules"]
        if ("flatAttack" not in families
                or set(family) != {"rulesFixturePath", "rulesFixtureSha256", "movementFixturePath", "movementFixtureSha256"}):
            raise ValueError("flat Basic rules require both pinned campaigns and the flat attack family")
        for kind in ("rules", "movement"):
            path = (root / family[f"{kind}FixturePath"]).resolve(strict=True)
            read_pinned(path, family[f"{kind}FixtureSha256"])
            artifacts[f"flatRules{kind.title()}Fixture"] = path
        if not {"experiments/flat_attack_family/rules.cpp", "experiments/flat_attack_family/movement.cpp"} <= source_hashes.keys():
            raise ValueError("candidate source omits the reviewed flat Basic rule campaigns")
    if "movement" in families:
        family = families["movement"]
        if set(family) != {"manifestPath", "manifestSha256", "fixturePath", "fixtureSha256"}:
            raise ValueError("movement family requires pinned manifest and fixture")
        for kind in ("manifest", "fixture"):
            path = (root / family[f"{kind}Path"]).resolve(strict=True)
            read_pinned(path, family[f"{kind}Sha256"])
            artifacts[f"movement{kind.title()}"] = path
        if not {"experiments/movement_family/cabt.cpp", "experiments/movement_family/Commands.h"} <= source_hashes.keys():
            raise ValueError("candidate source omits the reviewed movement experiment")
    for family_name, directory, header in (("evolved", "evolved_flat_family", "GeneratedEvolvedFlatCards.h"),
                                            ("extraEvolved", "extra_evolved_flat", "GeneratedExtraEvolvedFlatCards.h")):
        if family_name not in families:
            continue
        family = families[family_name]
        paths = {"manifest": f"{family_name}AttackManifest", "names": f"{family_name}Names",
                 **{f"{kind}Fixture": f"{family_name}{kind.title()}Fixture"
                    for kind in ("attack", "rules", "evolution", "retreat")}}
        required = {f"{name}{suffix}" for name in paths if name != "retreatFixture" for suffix in ("Path", "Sha256")}
        optional = {"retreatFixturePath", "retreatFixtureSha256"}
        if not required <= family.keys() or set(family) not in (required, required | optional):
            raise ValueError("evolved family requires pinned source, names and companion fixtures")
        if family_name == "extraEvolved" and not optional <= family.keys():
            raise ValueError("extra evolved family requires the complete retreat campaign")
        for name, artifact in paths.items():
            if f"{name}Path" in family:
                path = (root / family[f"{name}Path"]).resolve(strict=True)
                read_pinned(path, family[f"{name}Sha256"])
                artifacts[artifact] = path
        required_source = {header, *(f"experiments/{directory}/{name}.cpp"
                          for name in ("cabt", "rules", "evolution"))}
        if "retreatFixturePath" in family:
            required_source.add(f"experiments/{directory}/retreat.cpp")
        if not required_source <= source_hashes.keys():
            raise ValueError("candidate source omits the reviewed evolved experiments")
    if "draw" in families:
        family = families["draw"]
        if set(family) != {"manifestPath", "manifestSha256", "fixturePath", "fixtureSha256"}:
            raise ValueError("draw family requires pinned manifest and fixture")
        for kind in ("manifest", "fixture"):
            path = (root / family[f"{kind}Path"]).resolve(strict=True)
            read_pinned(path, family[f"{kind}Sha256"])
            artifacts[f"draw{kind.title()}"] = path
        if not {"experiments/draw_supporters/cabt.cpp", "experiments/draw_supporters/Commands.h"} <= source_hashes.keys():
            raise ValueError("candidate source omits the reviewed draw experiment")
    if "dragapult" in families:
        family = families["dragapult"]
        if set(family) != {"manifestPath", "manifestSha256", "fixturePath", "fixtureSha256"}:
            raise ValueError("Dragapult family requires pinned manifest and fixture")
        for kind in ("manifest", "fixture"):
            path = (root / family[f"{kind}Path"]).resolve(strict=True)
            read_pinned(path, family[f"{kind}Sha256"])
            artifacts[f"dragapult{kind.title()}"] = path
        if not {"experiments/ranked_dragapult/cabt.cpp", "experiments/ranked_dragapult/Commands.h"} <= source_hashes.keys():
            raise ValueError("candidate source omits the reviewed Dragapult experiment")
    if "recoil" in families:
        family = families["recoil"]
        names = ("manifest", "legacyFixture", "sessionFixture", "sessionRecords", "legacyReceipt", "sessionReceipt", "buildReceipt")
        required = {name + suffix for name in names for suffix in ("Path", "Sha256")}
        supplement = ("rulesFixture", "movementFixture", "rulesBuildReceipt", "rulesReceipt")
        optional = {name + suffix for name in supplement for suffix in ("Path", "Sha256")}
        session_optional = {"sessionSupplementPath", "sessionSupplementSha256"}
        if set(family) not in (required, required | optional, required | session_optional,
                               required | optional | session_optional):
            raise ValueError("recoil family requires pinned ordinary and session fixture provenance")
        for name in (*names, *supplement, "sessionSupplement"):
            if name + "Path" not in family:
                continue
            path = (root / family[name + "Path"]).resolve(strict=True)
            read_pinned(path, family[name + "Sha256"])
            artifacts["recoil" + name[0].upper() + name[1:]] = path
        if not {"GeneratedRecoilCards.h", "MatchSession.h", "experiments/recoil_family/cabt.cpp",
                "experiments/recoil_session/native.cpp"} <= source_hashes.keys():
            raise ValueError("candidate source omits the measured recoil and session fixtures")
    if "flatDraw" in families:
        family = families["flatDraw"]
        if set(family) != {"manifestPath", "manifestSha256", "fixturePath", "fixtureSha256"}:
            raise ValueError("flat draw family requires pinned manifest and fixture")
        for kind in ("manifest", "fixture"):
            path = (root / family[f"{kind}Path"]).resolve(strict=True)
            read_pinned(path, family[f"{kind}Sha256"])
            artifacts[f"flatDraw{kind.title()}"] = path
        if not {"GeneratedFlatDrawCards.h", "experiments/flat_draw_family/cabt.cpp",
                "experiments/flat_draw_family/Commands.h"} <= source_hashes.keys():
            raise ValueError("candidate source omits the measured flat draw fixture")
    if "nativeMetadata" in families:
        family = families["nativeMetadata"]
        names = ("manifest", "fixture", "sourceReceipt")
        if set(family) != {name + suffix for name in names for suffix in ("Path", "Sha256")}:
            raise ValueError("native metadata requires pinned inputs, fixture and supplemental source receipt")
        for name in names:
            path = (root / family[name + "Path"]).resolve(strict=True)
            read_pinned(path, family[name + "Sha256"])
            artifacts["nativeMetadata" + name[0].upper() + name[1:]] = path
    if "standardMechanics" in families:
        family = families["standardMechanics"]
        names = ("runtimeClosure", "friendsReceipt", "friendsFixture", "minccinoReceipt", "minccinoFixture")
        if set(family) != {name + suffix for name in names for suffix in ("Path", "Sha256")}:
            raise ValueError("Standard mechanics requires captured runtime and both measured native fixtures")
        for name in names:
            path = (root / family[name + "Path"]).resolve(strict=True)
            read_pinned(path, family[name + "Sha256"])
            artifacts["standardMechanics" + name[0].upper() + name[1:]] = path
    for name in ("standardLegalPrompts", "standardEffects"):
        if name not in families:
            continue
        family = families[name]
        if name == "standardEffects" and set(family) == {"batches"}:
            if not isinstance(family["batches"], list) or not family["batches"]:
                raise ValueError("Standard effect batches require a nonempty pinned packet inventory")
            batches = []
            seen = set()
            for batch in family["batches"]:
                names = ("packet", "fixture", "replayReceipt") if isinstance(batch, dict) and "replayReceiptPath" in batch else ("packet", "fixture")
                if (not isinstance(batch, dict)
                        or set(batch) != {kind + suffix for kind in names for suffix in ("Path", "Sha256")}
                        or batch["packetSha256"] in seen):
                    raise ValueError("Standard effect batch changed or duplicated its pinned artifacts")
                seen.add(batch["packetSha256"])
                entry = {}
                for kind in names:
                    path = (root / batch[kind + "Path"]).resolve(strict=True)
                    read_pinned(path, batch[kind + "Sha256"])
                    entry[name + kind[0].upper() + kind[1:]] = path
                batches.append(entry)
            artifacts["standardEffectsBatches"] = batches
            continue
        names = ("packet", "fixture", "replayReceipt") if name == "standardEffects" and "replayReceiptPath" in family else ("packet", "fixture")
        if set(family) != {kind + suffix for kind in names for suffix in ("Path", "Sha256")}:
            raise ValueError("Standard action observations require a measured packet and native fixture")
        for kind in names:
            path = (root / family[kind + "Path"]).resolve(strict=True)
            read_pinned(path, family[kind + "Sha256"])
            artifacts[name + kind[0].upper() + kind[1:]] = path
    if "standardMissingPool" in families:
        family = families["standardMissingPool"]
        names = ("packet", "replayReceipt")
        if set(family) != {kind + suffix for kind in names for suffix in ("Path", "Sha256")}:
            raise ValueError("Standard missing-pool cases require their captured packet and current candidate replay receipt")
        for kind in names:
            path = (root / family[kind + "Path"]).resolve(strict=True)
            read_pinned(path, family[kind + "Sha256"])
            artifacts["standardMissingPool" + kind[0].upper() + kind[1:]] = path
    if "standardPrivacy" in families:
        family = families["standardPrivacy"]
        names = ("prepared", "measurementReceipt")
        if set(family) != {kind + suffix for kind in names for suffix in ("Path", "Sha256")}:
            raise ValueError("Standard privacy requires its captured preparation and actual native measurement receipt")
        for kind in names:
            path = (root / family[kind + "Path"]).resolve(strict=True)
            read_pinned(path, family[kind + "Sha256"])
            artifacts["standardPrivacy" + kind[0].upper() + kind[1:]] = path
    return artifacts


def verify_library_catalog(library_path, engine, registry):
    library = ctypes.CDLL(str(library_path))
    library.GameInitialize.argtypes = []
    library.GameInitialize.restype = None
    library.AllCard.argtypes = []
    library.AllCard.restype = ctypes.c_char_p
    library.GameInitialize()
    raw = library.AllCard()
    if not raw or sha256(raw).hexdigest() != engine["registrySha256"]:
        raise ValueError("candidate DLL AllCard export differs from the pinned registry")
    cards = json.loads(raw)
    if cards != registry["entries"]:
        raise ValueError("candidate DLL metadata differs from the registry snapshot")
    return {card["cardId"]: card for card in cards}


def run_native_fixtures(execution, candidate, manifest_root, registry, source_rows=()):
    handlers = []
    flat_cases = []
    energy_cases = []
    fairy_energy_cases = []
    flat_rule_cases = []
    movement_cases = []
    evolved_cases = []
    extra_evolved_cases = []
    draw_cases = []
    dragapult_cases = []
    recoil_cases = []
    flat_draw_cases = []
    native_metadata_cases = []
    standard_mechanics_cases = []
    standard_legal_prompt_cases = []
    standard_effect_cases = []
    standard_missing_pool_cases = []
    standard_privacy_cases = []
    for case in execution["cases"]:
        scenario = case["scenario"]
        if scenario.get("suite") == "standard-information-boundaries/1":
            standard_privacy_cases.append(case)
            continue
        if scenario.get("suite") == "standard-missing-pool/1":
            standard_missing_pool_cases.append(case)
            continue
        if scenario.get("suite") == "standard-physical-effect/1":
            standard_effect_cases.append(case)
            continue
        if scenario.get("suite") == "source-verified-no-effect-legal-prompts/1":
            standard_legal_prompt_cases.append(case)
            continue
        if scenario.get("suite") in {"standard-mechanics-matrix/1", "standard-mechanics-rules/1"}:
            standard_mechanics_cases.append(case)
            continue
        if scenario.get("suite") in {"flat-attack-matrix/1", "flat-card-metadata/1"}:
            flat_cases.append(case)
            continue
        if scenario.get("suite") in {"basic-energy-metadata/1", "basic-energy-rules/1"}:
            energy_cases.append(case)
            continue
        if scenario.get("suite") in {"fairy-basic-energy-metadata/1", "fairy-basic-energy-rules/1"}:
            fairy_energy_cases.append(case)
            continue
        if scenario.get("suite") == "flat-basic-rules/1":
            flat_rule_cases.append(case)
            continue
        if scenario.get("suite") in {"movement-metadata/1", *(f"movement-family-{facet}/1" for facet in TRANSITION_FACETS)}:
            movement_cases.append(case)
            continue
        if scenario.get("suite") in {"evolved-flat-card-metadata/1", "evolved-flat-attack-matrix/1", "evolved-flat-rules/1"}:
            evolved_cases.append(case)
            continue
        if scenario.get("suite") in {"extra-evolved-flat-card-metadata/1", "extra-evolved-flat-attack-matrix/1", "extra-evolved-flat-rules/1"}:
            extra_evolved_cases.append(case)
            continue
        if scenario.get("suite") in {"draw-supporters-metadata/1", *(f"draw-supporters-{facet}/1" for facet in TRANSITION_FACETS)}:
            draw_cases.append(case)
            continue
        if scenario.get("suite") in {"ranked-dragapult-metadata/1", *(f"ranked-dragapult-useActions-{index}-{facet}/1" for index in (0, 1) for facet in DRAGAPULT_FACETS)}:
            dragapult_cases.append(case)
            continue
        if scenario.get("suite") in {"recoil-composite/1", "recoil-rules-composite/1"}:
            recoil_cases.append(case)
            continue
        if scenario.get("suite") in {"flat-draw-metadata/1", *(f"flat-draw-useActions-{index}-{facet}/1" for index in (0, 1) for facet in FLAT_DRAW_FACETS)}:
            flat_draw_cases.append(case)
            continue
        if scenario.get("suite") == "native-full-metadata/1":
            native_metadata_cases.append(case)
            continue
        handler = HANDLERS.get(scenario.get("suite"))
        if (handler is None or case["caseId"] not in handler.case_ids or case["cabtCardId"] != handler.card_id
                or digest(scenario) != digest({"suite": scenario.get("suite"), "commands": handler.commands})):
            raise ValueError(f"no reviewed native fixture handler for case and scenario: {case['caseId']}")
        handlers.append((case, handler))
    artifacts = verify_candidate_build(candidate, manifest_root, execution["engine"])
    cards = verify_library_catalog(artifacts["library"], execution["engine"], registry)
    observations = []
    executed = {}
    for case, handler in handlers:
        if handler.card_id not in cards:
            raise ValueError("reviewed fixture native card is absent from candidate DLL")
        observation = {"commands": [], "responses": []}
        components = []
        for component in handler.components:
            if component.suite not in executed:
                completed = subprocess.run([str(artifacts["fixture"]), *component.arguments],
                                           capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
                actual = json.loads(completed.stdout)
                if (not isinstance(actual, dict) or set(actual) != {"commands", "responses"}
                        or digest(actual["commands"]) != digest(component.commands)
                        or not isinstance(actual["responses"], list)
                        or len(actual["responses"]) != len(component.commands)
                        or any(not _response_shape(response, component.response_shape)
                               or any(response[field] != command[field] for field in component.echoed_fields)
                               for response, command in zip(actual["responses"], component.commands))):
                    raise ValueError(f"native fixture did not return the requested observations: {component.suite}")
                executed[component.suite] = actual
            actual = executed[component.suite]
            observation["commands"].extend(actual["commands"])
            observation["responses"].extend(actual["responses"])
            components.append({"suite": component.suite, "scenarioSha256": digest(component.scenario)})
        observations.append({"caseId": case["caseId"], "scenarioSha256": digest(case["scenario"]),
                             "cabtCardId": handler.card_id, "observation": observation,
                             "scenarioComponents": components})
    if flat_cases or flat_rule_cases:
        observations.extend(_run_flat_family(flat_cases, artifacts, candidate, cards, source_rows))
    if energy_cases:
        observations.extend(run_energy_family(energy_cases, artifacts, cards, source_rows))
    if fairy_energy_cases:
        observations.extend(run_energy_family(fairy_energy_cases, artifacts, cards, source_rows, FAIRY_ENERGY_DOMAIN))
    if flat_rule_cases:
        observations.extend(run_flat_rules_family(flat_rule_cases, artifacts, candidate, cards, source_rows))
    if movement_cases:
        observations.extend(run_movement_family(movement_cases, artifacts, candidate, cards, source_rows))
    if evolved_cases:
        observations.extend(run_evolved_family(evolved_cases, artifacts, candidate, cards, source_rows))
    if extra_evolved_cases:
        observations.extend(run_evolved_family(extra_evolved_cases, artifacts, candidate, cards, source_rows,
                                                domain=EXTRA_EVOLVED_DOMAIN))
    if draw_cases:
        observations.extend(run_draw_family(draw_cases, artifacts, candidate, cards, source_rows))
    if dragapult_cases:
        observations.extend(run_dragapult_family(dragapult_cases, artifacts, candidate, cards, source_rows))
    if recoil_cases:
        observations.extend(run_recoil_family(recoil_cases, artifacts, candidate, cards, source_rows))
    if flat_draw_cases:
        observations.extend(run_flat_draw_family(flat_draw_cases, artifacts, candidate, cards, source_rows))
    if native_metadata_cases:
        observations.extend(run_native_metadata(native_metadata_cases, artifacts, candidate, cards, source_rows))
    if standard_mechanics_cases:
        observations.extend(run_standard_mechanics(standard_mechanics_cases, artifacts, candidate, cards, source_rows))
    if standard_legal_prompt_cases:
        observations.extend(run_standard_legal_prompts(standard_legal_prompt_cases, artifacts, candidate, cards, source_rows))
    if standard_effect_cases:
        observations.extend(run_standard_effects(standard_effect_cases, artifacts, candidate, cards, source_rows))
    if standard_missing_pool_cases:
        observations.extend(run_standard_missing_pool(standard_missing_pool_cases, artifacts, candidate, cards, source_rows))
    if standard_privacy_cases:
        observations.extend(run_standard_privacy(standard_privacy_cases, artifacts, candidate, cards, source_rows))
    return NativeFixtureExecution(observations, {
        "kind": "reviewed_builtin_source_native", "qualificationScope": SOURCE_NATIVE_SCOPE,
        "publicDllArbitraryStateExecution": False, "receiptSha256": candidate["receiptSha256"],
        "sourceCommit": candidate["sourceCommit"], "sourceFiles": len(candidate["sourceSha256"]),
        "librarySha256": candidate["librarySha256"], "registrySha256": execution["engine"]["registrySha256"],
        "fixtureSha256": candidate["fixtureSha256"], "fixtureFamilies": candidate.get("fixtureFamilies", {}),
        "executionManifestSha256": digest(execution)})


def _response_shape(response, shape):
    return (isinstance(response, dict) and response.keys() == shape.keys()
            and all(_response_shape(response[field], expected) if isinstance(expected, dict)
                    else type(response[field]) is expected for field, expected in shape.items()))


FLAT_TYPES = {1: "Colorless", 2: "Fire", 3: "Grass", 4: "Water", 5: "Fighting", 7: "Darkness",
              8: "Lightning", 9: "Dragon", 10: "Metal", 11: "Psychic"}
FLAT_ENERGIES = dict(zip("CRGWFDLNMP", ("Colorless", "Fire", "Grass", "Water", "Fighting", "Darkness",
                                           "Lightning", "Dragon", "Metal", "Psychic")))
NATIVE_TYPES = {0: 1, 1: 3, 2: 2, 3: 4, 4: 8, 5: 11, 6: 5, 7: 7, 8: 10, 9: 9}
NATIVE_ENERGY_SYMBOLS = dict(enumerate("CGRWLPFDMN"))
FLAT_MODES = ("sufficient", "none", "short", "wrong", "weakness", "resistance", "bench")
FLAT_RESPONSE_SHAPE = {"programSha256": str, "attackIndex": int, "mode": str, "legal": bool,
                       "damage": int, "turnEnded": bool, "prompted": bool, "gameOver": bool}


def flat_attack_scenario(item, attack_index):
    modes = [mode for mode in FLAT_MODES
             if item["card"]["typing"] != "Colorless" or mode not in {"weakness", "resistance"}]
    return {"suite": "flat-attack-matrix/1", "commands": [
        {"operation": "attack", "programSha256": item["programSha256"], "attackIndex": attack_index,
         "mode": mode, "defender": "synthetic-basic-10000hp"} for mode in modes]}


def flat_metadata_scenario(item, source_id):
    return {"suite": "flat-card-metadata/1", "commands": [
        {"operation": "nativeCardMetadata", "programSha256": item["programSha256"],
         "cardSourceID": source_id, "cardSha256": item["cardSha256"][source_id]}]}


def _flat_fields(value, allowed):
    if not isinstance(value, dict) or set(value) - set(allowed.split()):
        raise ValueError("source program is outside the reviewed flat attack domain")


def _flat_source_card(card):
    _flat_fields(card, "cardArchetypeID cardName cardProperties cardSourceID cardTyping evolvesFromBase "
                 "evolvesInto hp logicSourceID prizeValue releaseDate resistance resistanceAmt retreatCost "
                 "useActions weakness weaknessAmt cardStage")
    if (card.get("cardStage", 1) != 1 or card.get("cardProperties") != [0] or card.get("prizeValue") != 1
            or card.get("evolvesFromBase") != [card.get("cardName")]
            or len(card.get("cardTyping", [])) != 1 or card["cardTyping"][0] not in FLAT_TYPES
            or type(card.get("hp")) is not int or not 0 < card["hp"] <= 1000
            or type(card.get("retreatCost", 0)) is not int or not 0 <= card.get("retreatCost", 0) <= 4
            or not isinstance(card.get("cardName"), str) or not card["cardName"]
            or any(part in card["cardName"] for part in ("'s ", "’s ", "Team ", "[", "]"))):
        raise ValueError("source card is outside the reviewed plain Basic domain")
    parsed = {"name": card["cardName"], "hp": card["hp"], "retreat": card.get("retreatCost", 0),
              "typing": FLAT_TYPES[card["cardTyping"][0]], "attacks": []}
    for field, amount in (("weakness", 2), ("resistance", 30)):
        values = card.get(field, [])
        if (not isinstance(values, list) or len(values) > 1
                or any(value not in FLAT_TYPES or value == 1 for value in values)
                or (values and card.get(field + "Amt") != amount)
                or (not values and card.get(field + "Amt", 0) != 0)):
            raise ValueError("source weakness or resistance is outside the reviewed domain")
        parsed[field] = FLAT_TYPES[values[0]] if values else None
    actions = card.get("useActions", [])
    if not 1 <= len(actions) <= 2:
        raise ValueError("flat source must contain one or two attacks")
    for action in actions:
        _flat_fields(action, "actionGuid actionName actionProperties actionSourceID actionType energyCost subActions")
        cost = action.get("energyCost", "")
        if (action.get("actionType") != 1 or action.get("actionProperties") != [1]
                or not isinstance(action.get("actionName"), str) or not action["actionName"]
                or not isinstance(cost, str) or re.fullmatch(r"(?:\{[CRGWFDLMNP]\}){0,8}", cost) is None
                or len(action.get("subActions", [])) != 2):
            raise ValueError("source action is outside the reviewed flat attack domain")
        damage, resolve = action["subActions"]
        _flat_fields(damage, "$type damageValue editorSettings subActionName target")
        _flat_fields(resolve, "$type subActionName")
        target, value = damage.get("target", {}), damage.get("damageValue", {})
        _flat_fields(target, "entityLocations groupingSize editorSettings")
        _flat_fields(value, "$type value1")
        number = value.get("value1", {})
        _flat_fields(number, "$type explicitValue")
        amount = number.get("explicitValue")
        if (damage.get("$type") != "MatchLogic.ApplyDamage, MatchLogic"
                or resolve.get("$type") != "MatchLogic.ResolveAttack, MatchLogic"
                or damage.get("editorSettings", "DamageOpponentsActive") != "DamageOpponentsActive"
                or target.get("entityLocations") != [{"category": 12, "entityPositions": [7]}]
                or target.get("groupingSize") != {"$type": "MatchLogic.AllNumberVariableTotal, MatchLogic"}
                or target.get("editorSettings", "OpponentActive") != "OpponentActive"
                or value.get("$type") != "MatchLogic.SingleVariableTotal, MatchLogic"
                or number.get("$type") != "MatchLogic.ExplicitNumberVariable, MatchLogic"
                or type(amount) is not int or not 0 < amount <= 1000 or amount % 10):
            raise ValueError("source damage program is outside the reviewed flat attack domain")
        parsed["attacks"].append({"name": action["actionName"], "damage": amount,
                                  "energy": [FLAT_ENERGIES[symbol] for symbol in re.findall(r"\{(.)\}", cost)],
                                  "source_guid": action["actionGuid"]})
    return parsed


def _flat_metadata(item, source_id, cards, attacks):
    card = cards[item["cabtCardId"]]
    return {"programSha256": item["programSha256"], "cardSourceID": source_id, "name": card["name"],
            "hp": card["hp"], "retreat": card["retreatCost"], "typing": [NATIVE_TYPES[card["energyType"]]],
            "stage": 1 if card["basic"] else 2 if card["stage1"] else 3,
            "prizeValue": 1 if card["pokemonType"] == 1 else 2,
            "properties": [0] if card["pokemonType"] == 1 else [-1],
            "weakness": [NATIVE_TYPES[card["weakness"]]] if card["weakness"] is not None else [],
            "resistance": [NATIVE_TYPES[card["resistance"]]] if card["resistance"] is not None else [],
            "weaknessAmount": 2 if card["weakness"] is not None else 0,
            "resistanceAmount": 30 if card["resistance"] is not None else 0,
            "attacks": [{"name": attacks[identifier]["name"],
                         "energy": "".join("{" + NATIVE_ENERGY_SYMBOLS[energy] + "}"
                                           for energy in attacks[identifier]["energies"]),
                         "damage": int(attacks[identifier]["damage"])} for identifier in card["attacks"]]}


def _flat_observation(row):
    if (not isinstance(row, dict) or set(row) != set(FLAT_RESPONSE_SHAPE) | {"ownerView", "opponentView"}
            or not _response_shape({key: row[key] for key in FLAT_RESPONSE_SHAPE}, FLAT_RESPONSE_SHAPE)):
        raise ValueError("flat native fixture response shape differs from reviewed observations")
    owner, opponent = row["ownerView"], row["opponentView"]
    if (not isinstance(owner, dict) or not isinstance(opponent, dict)
            or not isinstance(owner.get("hand"), list) or not owner.get("active") or not opponent.get("active")):
        raise ValueError("flat native fixture omitted actual player views")
    active = opponent["active"][0]
    privacy = {"ownerHandKnown": bool(owner["hand"]) and all(bool(card.get("id")) for card in owner["hand"]),
               "opponentHandHidden": opponent.get("hand", object()) is None,
               "deckHiddenFromBoth": "deck" not in owner and "deck" not in opponent,
               "activePublic": owner["active"][0]["id"] == active["id"] and active["id"] > 0,
               "publicActiveDamage": active["maxHp"] - active["hp"],
               "publicAttachedEnergyCount": len(active["energyCards"]),
               "attachedEnergyPublic": all(card.get("id", 0) > 0 for card in active["energyCards"])}
    return {**{key: row[key] for key in FLAT_RESPONSE_SHAPE}, "privacy": privacy}


def _run_flat_family(cases, artifacts, candidate, cards, source_rows):
    if "flatAttackManifest" not in artifacts:
        raise ValueError("flat attack scenarios require a pinned family fixture")
    manifest = json.loads(artifacts["flatAttackManifest"].read_bytes())
    if (manifest.get("schema") != "flat-attack-family/1"
            or manifest.get("headerSha256") != candidate["sourceSha256"]["GeneratedFlatCards.h"]):
        raise ValueError("flat family manifest does not bind the generated source")
    by_source = {card["cardSourceID"]: (card, row_hash, program) for card, _, row_hash, program in source_rows}
    selected = manifest.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("flat attack family must select actual pinned source programs")
    items, row_items = {}, {}
    for item in selected:
        program, source_ids, native_id = item["programSha256"], item["cardSourceIDs"], item["cabtCardId"]
        if program in items or type(native_id) is not int or native_id not in cards:
            raise ValueError("duplicate flat program or native card absent from candidate DLL")
        if (not isinstance(source_ids, list) or not source_ids or len(set(source_ids)) != len(source_ids)
                or set(item["cardSha256"]) != set(source_ids)):
            raise ValueError("flat family printing identities must be complete and unique")
        if cards[native_id]["attacks"] != item["attackIds"]:
            raise ValueError("flat family attack IDs differ from candidate DLL")
        for source_id in source_ids:
            if source_id in row_items or source_id not in by_source:
                raise ValueError("flat family row is duplicate or absent from the pinned corpus")
            card, row_hash, source_program = by_source[source_id]
            expected = _flat_source_card(card)
            declared = {key: value for key, value in item["card"].items() if key != "internal_name"}
            if source_program != program or row_hash != item["cardSha256"][source_id] or expected != declared:
                raise ValueError("flat family declaration differs from full pinned source program")
            row_items[source_id] = item
        items[program] = item
    fixture = artifacts["flatAttackFixture"]
    registry = json.loads(subprocess.run([str(fixture), "registry"], capture_output=True, text=True,
                                         encoding="utf-8", timeout=60, check=True).stdout)
    library = ctypes.CDLL(str(artifacts["library"]))
    library.AllAttack.argtypes = []
    library.AllAttack.restype = ctypes.c_char_p
    raw_attacks = library.AllAttack()
    if not raw_attacks:
        raise ValueError("candidate DLL omitted AllAttack catalog")
    attacks = json.loads(raw_attacks)
    if (registry.get("entries") != list(cards.values()) or registry.get("attacks") != attacks):
        raise ValueError("flat fixture catalogs differ from actual candidate DLL")
    attacks = {attack["attackId"]: attack for attack in attacks}
    raw = json.loads(subprocess.run([str(fixture)], capture_output=True, text=True, encoding="utf-8",
                                   timeout=120, check=True).stdout)
    if not isinstance(raw, list):
        raise ValueError("flat fixture must emit actual scenario observations")
    actual = {}
    for row in raw:
        observed = _flat_observation(row)
        item = items.get(observed["programSha256"])
        if (item is None or row["ownerView"]["active"][0]["id"] != item["cabtCardId"]
                or row["opponentView"]["active"][0]["id"] != item["cabtCardId"]
                or (observed["mode"] == "bench"
                    and (not row["ownerView"].get("bench")
                         or row["ownerView"]["bench"][0]["id"] != item["cabtCardId"]))):
            raise ValueError("flat fixture player views do not bind the requested native card")
        key = (observed["programSha256"], observed["attackIndex"], observed["mode"])
        if key in actual:
            raise ValueError("flat fixture emitted duplicate scenario observations")
        actual[key] = observed
    expected_keys = {(item["programSha256"], attack_index, command["mode"])
                     for item in selected for attack_index in range(len(item["card"]["attacks"]))
                     for command in flat_attack_scenario(item, attack_index)["commands"]}
    if actual.keys() != expected_keys:
        raise ValueError("flat fixture scenario inventory differs from reviewed family matrix")
    observations = []
    for case in cases:
        case_id, scenario = case["caseId"], case["scenario"]
        parts = case_id.split("/")
        if parts[0] == "row" and len(parts) == 3 and parts[2] == "metadata_identity":
            source_id = parts[1]
            item = row_items.get(source_id)
            if item is None:
                raise ValueError("flat metadata case is outside pinned family printings")
            expected = flat_metadata_scenario(item, source_id)
            responses = [_flat_metadata(item, source_id, cards, attacks)]
        elif (parts[0] == "program" and len(parts) == 5 and parts[2] == "useActions"
              and parts[4] in TRANSITION_FACETS and parts[1] in items):
            item = items[parts[1]]
            attack_index = int(parts[3])
            if str(attack_index) != parts[3] or not 0 <= attack_index < len(item["card"]["attacks"]):
                raise ValueError("flat attack case index differs from pinned source")
            expected = flat_attack_scenario(item, attack_index)
            responses = [actual[(item["programSha256"], attack_index, command["mode"])]
                         for command in expected["commands"]]
        else:
            raise ValueError("flat fixture cannot admit rules or unsupported action cases")
        if case["cabtCardId"] != item["cabtCardId"] or scenario != expected:
            raise ValueError("flat scenario differs from the reviewed full source binding")
        observations.append({"caseId": case_id, "scenarioSha256": digest(scenario),
                             "cabtCardId": item["cabtCardId"],
                             "observation": {"commands": expected["commands"], "responses": responses},
                             "scenarioComponents": [{"suite": scenario["suite"], "scenarioSha256": digest(scenario)}]})
    return observations
