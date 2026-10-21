from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

from tcg.standard_coverage import _cohort_registry, _name_key


CHANNELS = ("useActions", "passiveActions", "triggeredActions")
FACETS = ("effect", "legal_prompts", "information_boundaries", "interactions")


@dataclass(frozen=True)
class SourcePins:
    corpus_sha256: str
    index_sha256: str
    rows: int


LIVE_PINS = SourcePins(
    "afe7931aab5a6d4f77cdecbb7ae58502dd17fce4695986a348c2b27ebdd6c4ec",
    "6ceef4c2f6e0fe1cfd47947e8829936528633b67a2ea910a7d7592120fe643ad", 22945)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value):
    return sha256(canonical_bytes(value)).hexdigest()


def read_pinned(path, expected):
    data = Path(path).read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"input hash mismatch: {Path(path).name}")
    return data


def private_output(root, output):
    output = Path(output).resolve()
    if not output.is_relative_to((Path(root).resolve() / "runs").resolve()):
        raise ValueError("output must remain under this checkout's resolved runs/ directory")
    if output.exists():
        raise ValueError("output exists; preserve the frozen report and choose a fresh path")
    return output


def load_corpus(corpus_path, index_path, pins=LIVE_PINS):
    data = read_pinned(corpus_path, pins.corpus_sha256)
    index = json.loads(read_pinned(index_path, pins.index_sha256))
    if index.get("schemaVersion") != 2 or index.get("corpusSha256") != pins.corpus_sha256:
        raise ValueError("index does not bind the pinned corpus")
    indexed = index.get("cards")
    if not isinstance(indexed, list) or len(indexed) != pins.rows:
        raise ValueError("index row denominator differs from source pin")
    index_by_id = {row["cardSourceID"]: row for row in indexed}
    if len(index_by_id) != len(indexed):
        raise ValueError("duplicate cardSourceID in index")
    rows = []
    seen = set()
    offset = 0
    for line in data.splitlines(keepends=True):
        card = json.loads(line)
        source_id = card.get("cardSourceID")
        if not isinstance(source_id, str) or not source_id or source_id in seen:
            raise ValueError("missing or duplicate full cardSourceID in corpus")
        seen.add(source_id)
        entry = index_by_id.get(source_id)
        row_hash = sha256(line).hexdigest()
        if (entry is None or entry.get("cardSha256") != row_hash
                or entry.get("jsonlOffset") != offset or entry.get("jsonlBytes") != len(line)):
            raise ValueError(f"index row binding differs: {source_id}")
        for channel in CHANNELS:
            actions = card.get(channel, [])
            if not isinstance(actions, list) or any(not isinstance(action, dict) for action in actions):
                raise ValueError(f"invalid action channel: {source_id}/{channel}")
            names = [action.get("actionName") for action in actions]
            if entry.get("actions", {}).get(channel) != names:
                raise ValueError(f"index action inventory differs: {source_id}/{channel}")
        # Only printing identity and publication time are omitted. Nested identifiers stay significant.
        program = {key: value for key, value in card.items() if key not in {"cardSourceID", "releaseDate"}}
        rows.append((card, entry, row_hash, digest(program)))
        offset += len(line)
    if len(rows) != pins.rows or seen != set(index_by_id):
        raise ValueError("full raw-row denominator differs from source pin")
    return rows


def _requirements(rows):
    cases = {}
    programs = {}
    for card, entry, row_hash, program_hash in rows:
        source_id = card["cardSourceID"]
        case_id = f"row/{source_id}/metadata_identity"
        cases[case_id] = {"caseId": case_id, "programSha256": program_hash,
                          "cardSourceID": source_id, "cardSha256": row_hash, "facet": "metadata_identity"}
        if program_hash in programs:
            programs[program_hash]["cardSourceIDs"].append(source_id)
            continue
        required = [f"program/{program_hash}/rules"]
        cases[required[0]] = {"caseId": required[0], "programSha256": program_hash, "facet": "rules"}
        for channel in CHANNELS:
            for action_index, action in enumerate(card.get(channel, [])):
                for facet in FACETS:
                    case_id = f"program/{program_hash}/{channel}/{action_index}/{facet}"
                    required.append(case_id)
                    cases[case_id] = {"caseId": case_id, "programSha256": program_hash,
                                      "channel": channel, "actionIndex": action_index, "facet": facet}
        programs[program_hash] = {"programSha256": program_hash, "cardSourceIDs": [source_id],
                                  "logicSourceID": card.get("logicSourceID"), "requiredCaseIds": required}
    return cases, programs


def compare_fixture_observations(manifest, observations, cases, engine, pins, manifest_root):
    if (manifest.get("schema") != "live-parity-fixtures/1" or manifest.get("engine") != engine
            or manifest.get("sourcePins") != asdict(pins)
            or manifest.get("requiredCasesSha256") != digest(sorted(cases))):
        raise ValueError("fixture manifest does not bind the full requirements and engine")
    if (observations.get("manifestSha256") != digest(manifest)
            or observations.get("engine") != engine):
        raise ValueError("runner observations do not bind the manifest and engine")
    fixtures = manifest.get("cases", [])
    observed_rows = observations.get("cases", [])
    if not isinstance(fixtures, list) or not isinstance(observed_rows, list):
        raise ValueError("fixture cases and observations must be arrays")
    by_id = {row["caseId"]: row for row in observed_rows}
    if len(by_id) != len(observed_rows) or len({row["caseId"] for row in fixtures}) != len(fixtures):
        raise ValueError("duplicate fixture or observation caseId")
    fixture_ids = {row["caseId"] for row in fixtures}
    if not fixture_ids <= cases.keys() or not by_id.keys() <= fixture_ids:
        raise ValueError("fixture or observation refers to unknown required case")
    runtime = manifest.get("oracleRuntime", {})
    for kind in ("library", "worker"):
        if not isinstance(runtime.get(f"{kind}Path"), str) or not runtime.get(f"{kind}Sha256"):
            raise ValueError("oracle runtime requires pinned library and worker artifacts")
        read_pinned(Path(manifest_root) / runtime[f"{kind}Path"], runtime[f"{kind}Sha256"])
    statuses = {}
    bindings = []
    oracle_bundles = {}
    for fixture in fixtures:
        case_id = fixture["caseId"]
        case = cases[case_id]
        native_id = fixture.get("cabtCardId")
        if type(native_id) is not int or native_id not in case.get("nativeCandidateIds", []):
            raise ValueError(f"fixture has no exact row-bound native identity candidate: {case_id}")
        if native_id in case.get("metadataConflictingNativeIds", []):
            raise ValueError(f"fixture native metadata contradicts the pinned source row: {case_id}")
        oracle_path = (Path(manifest_root) / fixture["oraclePath"]).resolve()
        if "oracleCaseId" in fixture:
            bundle_key = (oracle_path, fixture["oracleSha256"])
            if bundle_key not in oracle_bundles:
                if len(oracle_bundles) == 2:
                    del oracle_bundles[next(iter(oracle_bundles))]
                oracle_bundles[bundle_key] = json.loads(read_pinned(*bundle_key))
            bundle = oracle_bundles[bundle_key]
            if (fixture["oracleCaseId"] != case_id or not isinstance(bundle, dict)
                    or bundle.get("schema") != "live-parity-oracle-bundle/1"
                    or not isinstance(bundle.get("oracles"), dict)):
                raise ValueError(f"oracle bundle does not bind its selected case: {case_id}")
            oracle = bundle["oracles"].get(case_id)
            if not isinstance(oracle, dict):
                raise ValueError(f"oracle bundle omits its selected case: {case_id}")
        else:
            oracle = json.loads(read_pinned(oracle_path, fixture["oracleSha256"]))
        scenario = fixture.get("scenario")
        provenance = oracle.get("provenance", {})
        observation = oracle.get("observation", {})
        if (not isinstance(scenario, dict) or not scenario or not isinstance(oracle.get("observation"), dict)
                or not oracle["observation"] or oracle.get("caseId") != case_id
                or oracle.get("sourceProgramSha256") != cases[case_id]["programSha256"]
                or oracle.get("scenario") != scenario
                or provenance.get("corpusSha256") != pins.corpus_sha256
                or provenance.get("librarySha256") != runtime["librarySha256"]
                or provenance.get("workerSha256") != runtime["workerSha256"]
                or oracle.get("cabtCardId") != native_id
                or (case["facet"] == "metadata_identity"
                    and (fixture.get("cardSourceID") != case["cardSourceID"]
                         or fixture.get("cardSha256") != case["cardSha256"]
                         or oracle.get("cardSourceID") != case["cardSourceID"]
                         or oracle.get("cardSha256") != case["cardSha256"]))
                or not isinstance(observation.get("commands"), list) or not observation["commands"]
                or not isinstance(observation.get("responses"), list) or not observation["responses"]):
            raise ValueError(f"oracle does not bind scenario, program, and observations: {case_id}")
        observed = by_id.get(case_id)
        status = "unobserved"
        if observed is not None:
            if (observed.get("scenarioSha256") != digest(scenario)
                    or observed.get("cabtCardId") != native_id
                    or not isinstance(observed.get("observation"), dict)):
                raise ValueError(f"runner omitted scenario or actual observations: {case_id}")
            status = "oracle_matched" if observed["observation"] == oracle["observation"] else "divergent"
        statuses[case_id] = status
        bindings.append({"caseId": case_id, "oracleSha256": fixture["oracleSha256"],
                         "scenarioSha256": digest(scenario)})
    return statuses, bindings


def _case_group_status(cases, required, *, candidate_ids=None, fallback="unobserved"):
    counts = Counter(cases[case_id]["status"] for case_id in required)
    native_ids = {cases[case_id]["observedNativeId"] for case_id in required
                  if cases[case_id]["status"] in {"oracle_matched", "provisional_oracle_matched"}}
    if counts["divergent"]:
        status = "divergent"
    elif len(native_ids) > 1 or (candidate_ids is not None and not native_ids <= set(candidate_ids)):
        status = "native_identity_conflict"
    else:
        status = fallback
        for matched, full, partial in (("oracle_matched", "declared_cases_matched", "partially_observed"),
                                       ("provisional_oracle_matched", "provisional_cases_matched",
                                        "partially_observed_provisional")):
            if counts[matched]:
                status = full if counts[matched] == len(required) else partial
                break
    return status, sorted(native_ids)


def inventory(rows, registry, *, pins=LIVE_PINS, engine_commit, library_sha256, registry_sha256,
              aliases=(), manifest=None, observations=None, manifest_root=None, runner_sha256=None,
              native_execution=None):
    if len(rows) != pins.rows:
        raise ValueError("row denominator differs from source pin")
    by_id, by_name = _cohort_registry(registry, library_sha256, registry_sha256)
    if (not isinstance(engine_commit, str) or not engine_commit.strip()
            or type(registry.get("engineVersion")) is not int or registry["engineVersion"] < 1
            or any(type(identifier) is not int or identifier < 0 for identifier in by_id)):
        raise ValueError("native registry needs an engine version, source commit, and integer card IDs")
    engine = {"version": registry.get("engineVersion"), "sourceCommit": engine_commit,
              "librarySha256": library_sha256, "registrySha256": registry_sha256}
    cases, programs = _requirements(rows)
    statuses, bindings = {}, []
    aliases_by_source = defaultdict(set)
    row_hashes = {card["cardSourceID"]: row_hash for card, _, row_hash, _ in rows}
    for alias in aliases:
        source_id, card_id = alias["cardSourceID"], alias["cabtCardId"]
        if (source_id not in row_hashes or alias.get("cardSha256") != row_hashes[source_id]
                or type(card_id) is not int or card_id not in by_id):
            raise ValueError("identity candidate does not bind an exact source row and registered card")
        aliases_by_source[source_id].add(card_id)
    exact_by_source = {}
    metadata_by_source = {}
    native_by_source = defaultdict(set)
    for identifier, native in by_id.items():
        source_id = native.get("cardSourceID")
        if isinstance(source_id, str):
            native_by_source[source_id].add(identifier)
    for card, _, _, program_hash in rows:
        source_id = card["cardSourceID"]
        exact = native_by_source[source_id] | aliases_by_source[source_id]
        exact_by_source[source_id] = sorted(exact)
        metadata = []
        for identifier in sorted(exact):
            native = by_id[identifier]
            checks = [{"field": field, "source": card[field], "native": native[field],
                       "matches": card[field] == native[field]}
                      for field in ("hp", "retreatCost") if field in card and field in native]
            if "attacks" in native:
                count = sum(action.get("actionType") == 1 for action in card.get("useActions", []))
                checks.append({"field": "attackCount", "source": count, "native": len(native["attacks"]),
                               "matches": count == len(native["attacks"])})
            metadata.append({"cabtCardId": identifier, "checks": checks,
                             "unresolvedFields": ["typing", "stage", "weakness", "resistance", "action_names"]})
        metadata_by_source[source_id] = metadata
        row_case = cases[f"row/{source_id}/metadata_identity"]
        row_case["nativeCandidateIds"] = sorted(exact)
        row_case["metadataConflictingNativeIds"] = [row["cabtCardId"] for row in metadata
                                                     if any(not check["matches"] for check in row["checks"])]
        for case_id in programs[program_hash]["requiredCaseIds"]:
            case = cases[case_id]
            case["nativeCandidateIds"] = sorted(set(case.get("nativeCandidateIds", [])) | exact)
    if observations is not None:
        if manifest is None or not runner_sha256:
            raise ValueError("observations require a pinned runnable fixture manifest and runner")
        statuses, bindings = compare_fixture_observations(
            manifest, observations, cases, engine, pins, manifest_root)
    admitted_ids = set()
    admission = {"kind": "custom_provisional" if observations is not None else "unobserved",
                 "qualificationScope": "comparison_only_no_native_execution_admitted"}
    if native_execution is not None:
        from tcg.parity_runner import NativeFixtureExecution, execution_manifest

        if (not isinstance(native_execution, NativeFixtureExecution) or observations is None
                or native_execution.cases != observations["cases"]
                or native_execution.provenance.get("executionManifestSha256") != digest(execution_manifest(manifest))):
            raise ValueError("native admission must bind the reviewed runner's actual observations")
        admitted_ids = {case["caseId"] for case in native_execution.cases}
        admission = native_execution.provenance
    observed_by_id = {case["caseId"]: case for case in observations["cases"]} if observations else {}
    for case_id, case in cases.items():
        case["status"] = statuses.get(case_id, "unobserved")
        if case["status"] == "oracle_matched" and case_id not in admitted_ids:
            case["status"] = "provisional_oracle_matched"
        if case_id in observed_by_id:
            case["observedNativeId"] = observed_by_id[case_id]["cabtCardId"]
            case["qualificationScope"] = (admission["qualificationScope"] if case_id in admitted_ids
                                          else "comparison_only_no_native_execution_admitted")
            if case_id in admitted_ids:
                case["admittedScenarioComponents"] = observed_by_id[case_id]["scenarioComponents"]
    result_rows = []
    conflicts = defaultdict(set)
    family_gaps = defaultdict(set)
    for card, entry, row_hash, program_hash in rows:
        source_id = card["cardSourceID"]
        conflicts[card.get("logicSourceID")].add(program_hash)
        named = sorted(by_name[_name_key(card["cardName"])])
        exact = exact_by_source[source_id]
        required = [f"row/{source_id}/metadata_identity", *programs[program_hash]["requiredCaseIds"]]
        fallback = "identity_candidate_untested" if exact else "identity_unresolved" if named else "absent"
        status, native_ids = _case_group_status(cases, required, candidate_ids=exact, fallback=fallback)
        if status != "declared_cases_matched":
            for family in entry.get("types", {}):
                family_gaps[family].add(program_hash)
        result_rows.append({"cardSourceID": source_id, "cardName": card["cardName"],
                            "cardSha256": row_hash, "logicSourceID": card.get("logicSourceID"),
                            "programSha256": program_hash, "status": status,
                            "exactIdentityCandidateIds": exact, "nameCandidateIds": named,
                            "structuralMetadata": metadata_by_source[source_id],
                            "observedNativeIds": native_ids,
                            "metadataStatus": cases[required[0]]["status"], "requiredCaseIds": required})
    for program in programs.values():
        program["status"], program["observedNativeIds"] = _case_group_status(cases, program["requiredCaseIds"])
    sorted_rows = sorted(result_rows, key=lambda row: row["cardSourceID"])
    requirement_ids = sorted(cases)
    row_ids = [row["cardSourceID"] for row in sorted_rows]
    matched = sum(row["status"] == "declared_cases_matched" for row in sorted_rows)
    return {"schema": "live-parity-inventory/1", "sourcePins": asdict(pins), "engine": engine,
            "denominator": {"rawRows": pins.rows, "uniqueSourceIds": len(row_ids),
                            "sourceIdsSha256": digest(row_ids), "sourcePrograms": len(programs),
                            "logicSourceIds": len(conflicts), "requiredCases": len(cases),
                            "requiredCasesSha256": digest(requirement_ids)},
            "rowStatusCounts": dict(sorted(Counter(row["status"] for row in sorted_rows).items())),
            "programStatusCounts": dict(sorted(Counter(row["status"] for row in programs.values()).items())),
            "caseStatusCounts": dict(sorted(Counter(row["status"] for row in cases.values()).items())),
            "declaredCaseRowCoveragePercent": 100 * matched / pins.rows,
            "fullParityQualified": matched == pins.rows,
            "qualificationScope": "frozen_full_live_dump_source_native_fixture_required_cases",
            "qualificationGate": "all_frozen_cases_require_reviewed_native_execution_and_one_consistent_native_identity",
            "rulingsPassDeferred": True,
            "stateSpaceEquivalenceProven": False,
            "fixtureVerification": {"runnerSha256": runner_sha256,
                                    "manifestSha256": digest(manifest) if manifest else None,
                                    "admission": admission,
                                    "bindings": bindings},
            "logicSourceProgramVariants": [{"logicSourceID": logic, "programSha256s": sorted(hashes)}
                                           for logic, hashes in sorted(conflicts.items()) if len(hashes) > 1],
            "unmatchedEffectFamilies": [{"family": family, "unmatchedPrograms": len(hashes)}
                                        for family, hashes in sorted(family_gaps.items(),
                                                                     key=lambda item: (-len(item[1]), item[0]))],
            "nextBatch": [{"programSha256": program["programSha256"],
                           "cardSourceIDs": program["cardSourceIDs"],
                           "requiredCaseCount": len(program["requiredCaseIds"])}
                          for program in sorted(programs.values(),
                                                key=lambda row: (-len(row["cardSourceIDs"]), row["programSha256"]))
                          if program["status"] != "declared_cases_matched"][:50],
            "rows": sorted_rows, "programs": [programs[key] for key in sorted(programs)],
            "cases": [cases[key] for key in requirement_ids]}


def compare_baseline(report, baseline, *, allow_engine_change=False):
    if (baseline.get("schema") != report["schema"] or baseline.get("sourcePins") != report["sourcePins"]
            or (not allow_engine_change and baseline.get("engine") != report["engine"])):
        raise ValueError("hill-climb comparison requires identical corpus and engine pins")
    if baseline.get("denominator") != report["denominator"]:
        raise ValueError("hill-climb comparison cannot shrink or change frozen row/program/case requirements")
    before_scenarios = _admitted_scenario_hashes(baseline)
    after_scenarios = _admitted_scenario_hashes(report)
    for case_id, hashes in before_scenarios.items():
        if not hashes <= after_scenarios.get(case_id, set()):
            raise ValueError(f"baseline admitted scenarios were removed or no longer match: {case_id}")
    for field, key in (("rows", "cardSourceID"), ("programs", "programSha256"), ("cases", "caseId")):
        before = {row[key]: row["status"] for row in baseline[field]}
        after = {row[key]: row["status"] for row in report[field]}
        if before.keys() != after.keys():
            raise ValueError("baseline identities differ despite denominator declaration")
        if field in {"rows", "programs"}:
            before_requirements = {row[key]: row["requiredCaseIds"] for row in baseline[field]}
            if any(before_requirements[row[key]] != row["requiredCaseIds"] for row in report[field]):
                raise ValueError("baseline required cases were removed or reassigned")
        report.setdefault("changes", {})[field] = [
            {key: identifier, "before": before[identifier], "after": after[identifier]}
            for identifier in sorted(after) if before[identifier] != after[identifier]]
    if baseline["engine"] != report["engine"]:
        report["engineTransition"] = {"baselineEngine": baseline["engine"], "candidateEngine": report["engine"],
                                      "evidencePolicy": "all observations rerun against candidate engine; no inherited statuses"}
    return report


def _admitted_scenario_hashes(report):
    verification = report.get("fixtureVerification", {})
    if verification.get("admission", {}).get("kind") != "reviewed_builtin_source_native":
        return {}
    bindings = {binding["caseId"]: binding["scenarioSha256"] for binding in verification.get("bindings", [])}
    admitted = {}
    for case in report["cases"]:
        if (case["status"] != "oracle_matched"
                or case.get("qualificationScope") != "source_native_fixture_with_verified_candidate_dll_catalog"):
            continue
        components = case.get("admittedScenarioComponents")
        if components is None:
            if case["caseId"] not in bindings:
                raise ValueError("legacy admitted scenario is missing its execution binding")
            admitted[case["caseId"]] = {bindings[case["caseId"]]}
        elif (not isinstance(components, list) or not components
              or any(not isinstance(component, dict) or set(component) != {"suite", "scenarioSha256"}
                     or not isinstance(component["suite"], str) or not component["suite"]
                     or not isinstance(component["scenarioSha256"], str) or len(component["scenarioSha256"]) != 64
                     for component in components)):
            raise ValueError("admitted scenario components must bind executed builtin fixtures")
        else:
            admitted[case["caseId"]] = {component["scenarioSha256"] for component in components}
    return admitted
