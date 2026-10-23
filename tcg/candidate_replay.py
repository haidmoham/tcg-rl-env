"""Retain captured source contracts while binding fresh candidate native fixtures."""

import json
from pathlib import Path
import subprocess

from tcg.live_parity import read_pinned


BASE_MANIFEST_SHA256 = "7c857ea598ec8e660575fc8d51927e32d3636c06386c8caed0a10b3576294d12"
REGISTRY_SHA256 = "a81584982eb02325cf5d558a1c3fd091c74a980702bc2be6d424708725fbe5e0"
LIBRARY_SHA256 = "eb0ba3176dfcc572da029d5765529f1d032a0d764a3e62aa5e937a20243673a0"
CATALOG_COMPANIONS = {
    "flatRules.rulesFixture": "flatAttack.fixture", "flatRules.movementFixture": "flatAttack.fixture",
    "evolved.rulesFixture": "evolved.attackFixture", "evolved.evolutionFixture": "evolved.attackFixture",
    "extraEvolved.rulesFixture": "extraEvolved.attackFixture",
    "extraEvolved.evolutionFixture": "extraEvolved.attackFixture",
}


def _read(binding):
    return json.loads(read_pinned(binding["path"], binding["sha256"]))


def candidate_replay(artifacts, candidate):
    path = artifacts.get("candidateReplayReceipt")
    if path is None:
        return None
    family = candidate["fixtureFamilies"]["candidateReplay"]
    proof = json.loads(read_pinned(path, family["receiptSha256"]))
    if (Path(path).resolve() != Path(family["receiptPath"]).resolve()
            or proof.get("schema") != "native-candidate-additive-replay/1"
            or proof["baselineManifest"]["sha256"] != BASE_MANIFEST_SHA256
            or proof["registryReceipt"]["sha256"] != REGISTRY_SHA256
            or proof["library"]["sha256"] != candidate["librarySha256"]
            or proof["sourceRecaptures"] != 0 or proof["priorImplementationFilesRetained"] is not True
            or proof["preservedRegistrationOrder"] is not True):
        raise ValueError("Candidate replay changed its measured source or native domain")
    if candidate["librarySha256"] != LIBRARY_SHA256 or proof["library"]["sha256"] != LIBRARY_SHA256:
        raise ValueError("Candidate replay library differs from the measured C14 library")
    registry = _read(proof["registryReceipt"])
    if (registry["sourceCommit"] != candidate["sourceCommit"]
            or registry["sourceSha256"] != candidate["sourceSha256"]
            or proof["candidateSourceCommit"] != candidate["sourceCommit"]
            or proof["candidateSourceSha256"] != candidate["sourceSha256"]
            or Path(proof["candidateSourceRoot"]).resolve() != Path(candidate["sourceRoot"]).resolve()
            or Path(registry["sourceRoot"]).resolve() != Path(candidate["sourceRoot"]).resolve()):
        raise ValueError("Candidate replay registration object changed complete source closure")
    read_pinned(registry["objectPath"], registry["objectSha256"])
    read_pinned(proof["library"]["path"], LIBRARY_SHA256)
    if Path(proof["library"]["path"]).resolve() != Path(artifacts["library"]).resolve():
        raise ValueError("Candidate replay does not bind the initialized library")
    parent = _read(proof["baselineManifest"])
    from tcg.parity_runner import verify_candidate_build
    old_artifacts = verify_candidate_build(parent["candidateBuild"], Path(proof["baselineManifest"]["path"]).parent,
                                          parent["engine"])
    old = parent["candidateBuild"]
    old_root, current_root = Path(old["sourceRoot"]), Path(candidate["sourceRoot"])
    changed = [name for name, pin in old["sourceSha256"].items()
               if not name.startswith("experiments/") and candidate["sourceSha256"].get(name) != pin]
    if changed != ["All.h"]:
        raise ValueError("Candidate replay changed old native implementation files")
    current_all = (current_root / "All.h").read_text(encoding="utf-8")
    for line in ('#include "GeneratedStandardMissingPool.h"\n', '#include "SourceVerifiedResidualCorrections.h"\n',
                 '  RegisterGeneratedStandardMissingPool();\n', '  RegisterSourceVerifiedResidualCorrections();\n'):
        if current_all.count(line) != 1:
            raise ValueError("Candidate replay omitted or duplicated a reviewed registration addition")
        current_all = current_all.replace(line, "")
    if current_all != (old_root / "All.h").read_text(encoding="utf-8"):
        raise ValueError("Candidate replay changed prior registration order")
    proof["parentCandidate"], proof["parentArtifacts"] = old, old_artifacts
    for role in proof["roles"]:
        if role == "fixture":
            fixture = candidate["fixturePath"]
        elif role == "graph":
            fixture = proof["roles"][role]["fixture"]["path"]
        else:
            name, field = role.split(".", 1)
            fixture = candidate["fixtureFamilies"][name][field + "Path"]
        replay_fixture(proof, role, fixture)
    return proof


def replay_fixture(proof, role, fixture):
    entry = proof["roles"][role]
    binding = entry["fixture"]
    if Path(binding["path"]).resolve() != Path(fixture).resolve():
        raise ValueError("Candidate replay fixture path differs from its role")
    read_pinned(fixture, binding["sha256"])
    old = proof["parentCandidate"]
    if role == "fixture":
        old_fixture = old["fixturePath"]
    elif role != "graph":
        family, field = role.split(".", 1)
        old_fixture = old["fixtureFamilies"][family][field + "Path"]
    else:
        old_fixture = entry["oldFixture"]["path"]
    if Path(entry["oldFixture"]["path"]).resolve() != Path(old_fixture).resolve():
        raise ValueError("Candidate replay replaced a different baseline fixture role")
    read_pinned(old_fixture, entry["oldFixture"]["sha256"])
    if role == "nativeMetadata.fixture":
        receipt = _read(entry["probeReceipt"])
        if (receipt["fixtureSha256"] != binding["sha256"]
                or receipt["librarySha256"] != LIBRARY_SHA256 or not receipt["actualDllCatalogEqual"]
                or receipt["engineSourceSha256"] != {
                    name: pin for name, pin in proof["candidateSourceSha256"].items()
                    if not name.startswith("experiments/") and Path(name).suffix in {".h", ".cpp", ".def", ".props", ".vcxproj"}}):
            raise ValueError("Candidate replay metadata probe changed compiled source or actual catalog")
        return
    registry = _read(proof["registryReceipt"])
    command = _read(entry["buildCommand"])
    if (command["exitCode"] or command["registryObjectSha256"] != registry["objectSha256"]
            or entry["registryObjectSha256"] != registry["objectSha256"]
            or Path(command["argv"][0]).resolve() != Path(registry["toolchain"]["msbuildPath"]).resolve()
            or Path(command["argv"][1]).resolve() != Path(entry["project"]["path"]).resolve()):
        raise ValueError("Candidate replay fixture was not built with the current registration object")
    for path, pin in entry["sources"].items():
        read_pinned(path, pin)
        path = Path(path).resolve()
        current_root = Path(proof["candidateSourceRoot"]).resolve()
        if path.is_relative_to(current_root):
            relative = path.relative_to(current_root).as_posix()
            if old["sourceSha256"].get(relative) != pin or proof["candidateSourceSha256"].get(relative) != pin:
                raise ValueError("Candidate replay changed a baseline native fixture source")
        else:
            read_pinned(Path(old_fixture).parent.parent / path.name, pin)
    read_pinned(entry["project"]["path"], entry["project"]["sha256"])
    if "catalog" in entry:
        _read(entry["catalog"])
    if role in CATALOG_COMPANIONS:
        paired = CATALOG_COMPANIONS[role]
        if (entry.get("catalogVerifiedThrough") != paired
                or proof["roles"][paired]["registryObjectSha256"] != registry["objectSha256"]
                or _read(proof["roles"][paired]["catalog"]) != _read(proof["metadata"]["registry.json"])):
            raise ValueError("Numeric-only companion fixture lacks its actual paired candidate catalog")
    if "catalogCapabilityCorrection" in entry:
        correction = _read(entry["catalogCapabilityCorrection"])
        if (correction["role"] != role or correction["sourceOrEngineChanges"] != 0
                or correction["catalogVerifiedThrough"] != CATALOG_COMPANIONS[role]
                or not _read(correction["failedCommand"])["exitCode"]):
            raise ValueError("Legacy fixture catalog capability correction changed its recorded failure")


def retained_metadata(proof, raw, registry, skills):
    old_receipt = json.loads(Path(proof["parentArtifacts"]["nativeMetadataSourceReceipt"]).read_bytes())
    old_root = Path(proof["parentArtifacts"]["nativeMetadataSourceReceipt"]).parent
    old = {name: json.loads(read_pinned(old_root / name, old_receipt["outputs"][name]))
           for name in ("native-raw.json", "native-skills.json", "registry.json")}
    for before, after, key in ((old["native-raw.json"], raw, "cardId"),
                               (old["native-skills.json"], skills, "skillId"),
                               (old["registry.json"]["entries"], registry["entries"], "cardId"),
                               (old["registry.json"]["attacks"], registry["attacks"], "attackId")):
        actual = {row[key]: row for row in after}
        if len(actual) != len(after) or any(actual.get(row[key]) != row for row in before):
            raise ValueError("Candidate replay changed an earlier physical metadata record")
    measured_graph = _read(proof["graph"])
    old_graph_path = Path(proof["parentCandidate"]["fixtureFamilies"]["nativeMetadata"]["manifestPath"])
    metadata_manifest = json.loads(old_graph_path.read_bytes())
    extension = metadata_manifest["graphIdentityExtension"]
    canonical = json.loads(read_pinned(extension["canonicalReceiptPath"], extension["canonicalReceiptSha256"]))
    graph_ref = canonical["measurementReceipt"]
    old_graph_receipt = _read(graph_ref)
    old_graph = json.loads(read_pinned(Path(graph_ref["path"]).parent / "graph.stdout",
                                     old_graph_receipt["outputs"]["graph.stdout"]))
    fixture = proof["roles"]["graph"]["fixture"]["path"]
    old_fixture = Path(graph_ref["path"]).parent / "bin/graph.exe"
    if Path(proof["roles"]["graph"]["oldFixture"]["path"]).resolve() != old_fixture.resolve():
        raise ValueError("Candidate replay changed its baseline Trainer graph exporter")
    replay_fixture(proof, "graph", fixture)
    graph = json.loads(subprocess.run([fixture, *[str(row["cardId"]) for row in old_graph["cards"]]],
                                      capture_output=True, check=True, timeout=60).stdout)
    if graph != measured_graph:
        raise ValueError("Fresh Trainer identity graphs differ from the bound candidate replay")
    if (graph["cards"] != old_graph["cards"] or graph["references"] != old_graph["references"]
            or (graph["cardCount"], graph["attackCount"], graph["skillCount"]) != (10251, 14305, 2648)):
        raise ValueError("Candidate replay changed Trainer identity graphs or concrete identity consumers")
    return old
