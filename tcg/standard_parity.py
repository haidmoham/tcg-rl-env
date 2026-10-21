from collections import Counter, defaultdict
from dataclasses import asdict

from tcg.live_parity import LIVE_PINS, digest


def standard_action_measurements(report, receipt, pairs, source_only):
    if (receipt.get("schema") != "current-standard-paired-first-sweep/1"
            or receipt["actualEngineSourceCommit"] != report["engine"]["sourceCommit"]
            or receipt["newFormalAdmissions"] != 0):
        raise ValueError("Action measurements must bind the current measured engine without adding formal credit")
    programs = {row["programSha256"] for row in report["programs"]}
    identities = defaultdict(set)
    for row in [*report["programs"], *report["rows"]]:
        identities[row["programSha256"]].update(row.get("observedNativeIds", []))
    expected = {"/".join(["source", *case["caseId"].split("/")[1:-1]])
                for case in report["cases"] if case.get("facet") == "effect"}
    statuses, edges, observed = Counter(), set(), defaultdict(set)
    binding_counts, bound = defaultdict(Counter), defaultdict(set)
    allowed = {"normalized-execution-match", "normalized-execution-difference", "readiness-difference",
               "both-blocked-no-effect-claim", "unsupported-action-binding", "unsupported-or-error"}
    for pair in pairs:
        edge = (pair["programSha256"], pair["nativeActionKey"])
        if edge in edges or edge[0] not in programs or pair["status"] not in allowed:
            raise ValueError("Action measurement edge is duplicated, outside Standard or has an unknown status")
        edges.add(edge)
        statuses[pair["status"]] += 1
        native_id = int(pair["nativeActionKey"].split("/")[1])
        measured_ids = identities[edge[0]]
        binding = ("measured_native_identity" if native_id in measured_ids else
                   "alternate_native_identity" if measured_ids else "identity_unresolved")
        binding_counts[binding][pair["status"]] += 1
        for key in pair["sourceActionKeys"]:
            if key not in expected or key.split("/")[1] != edge[0]:
                raise ValueError("Action measurement does not bind a frozen Standard action requirement")
            observed[key].add(pair["status"])
            if binding == "measured_native_identity":
                bound[key].add(pair["status"])
    unbound = {row["key"]: row for row in source_only}
    if (len(unbound) != len(source_only) or set(unbound) & observed.keys()
            or set(observed) | set(unbound) != expected
            or dict(statuses) != receipt["pairStatuses"]
            or len(pairs) != receipt["candidateBindingEdges"]
            or len(expected) != receipt["sourceActions"]
            or len(observed) != receipt["sourceActionsWithBinding"]
            or len(unbound) != receipt["sourceActionsWithNoBinding"]):
        raise ValueError("Action measurements must account for every frozen Standard action without changing scope")
    actions = [{"sourceActionKey": key, "programSha256": key.split("/")[1],
                "candidateMeasurementStatuses": sorted(observed[key]) if key in observed else ["source-only-no-action-binding"],
                "measuredIdentityNativeIds": sorted(identities[key.split("/")[1]]),
                "measuredIdentityPairStatuses": sorted(bound[key]),
                "sourceObservationStatus": unbound[key].get("observation", {}).get("status") if key in unbound else None}
               for key in sorted(expected)]
    return {"schema": "current-standard-action-measurements/1", "sourceActions": len(expected),
            "candidateBindingEdges": len(pairs), "pairStatusCounts": dict(sorted(statuses.items())),
            "pairStatusCountsByMeasuredIdentity": {key: dict(sorted(value.items()))
                                                  for key, value in sorted(binding_counts.items())},
            "sourceActionsWithNoBinding": len(unbound), "sourceAccounting": receipt["sourceAccounting"],
            "newFormalAdmissions": 0, "actions": actions, "scope": receipt["boundary"]}


def standard_coverage(pool, full_report):
    if (pool.get("schema") != "current-live-standard-pool/1"
            or full_report.get("schema") != "live-parity-inventory/1"
            or full_report.get("sourcePins") != asdict(LIVE_PINS)):
        raise ValueError("Standard coverage needs the pinned pool and full Live measurement report")
    historical = pool["historicalCorpus"]
    if (historical["corpusSha256"] != LIVE_PINS.corpus_sha256
            or historical["indexSha256"] != LIVE_PINS.index_sha256
            or historical["rows"] != LIVE_PINS.rows
            or historical["requiredCases"] != full_report["denominator"]["requiredCases"]):
        raise ValueError("Standard pool and measured historical source denominator differ")
    full_rows = {row["cardSourceID"]: row for row in full_report["rows"]}
    full_programs = {row["programSha256"]: row for row in full_report["programs"]}
    full_cases = {row["caseId"]: row for row in full_report["cases"]}
    if (len(full_rows) != LIVE_PINS.rows
            or len(full_cases) != full_report["denominator"]["requiredCases"]
            or len(full_programs) != historical["programs"]):
        raise ValueError("Full measurement report omits or duplicates source rows, programs or cases")
    selected_rows, program_rows, required = [], defaultdict(list), set()
    for identity in pool["rows"]:
        source_id = identity["cardSourceID"]
        row = full_rows[source_id]
        if (row["cardSha256"] != identity["cardSha256"]
                or row["programSha256"] != identity["programSha256"]
                or identity["activeProgramSha256"] != identity["programSha256"]):
            raise ValueError("Standard printing does not bind its measured raw row and active program")
        selected_rows.append(row)
        program_rows[row["programSha256"]].append(source_id)
        required.update(row["requiredCaseIds"])
    source_ids = [row["cardSourceID"] for row in selected_rows]
    if (len(source_ids) != len(set(source_ids))
            or len(source_ids) != pool["counts"]["pinnedEligibleRows"]
            or len(program_rows) != pool["counts"]["uniquePrograms"]):
        raise ValueError("Standard pool row/program denominator is incomplete or duplicated")
    declared_programs = {row["programSha256"]: sorted(row["cardSourceIDs"])
                         for row in pool["uniquePrograms"]}
    if (declared_programs != {key: sorted(value) for key, value in program_rows.items()}
            or len(declared_programs) != len(pool["uniquePrograms"])):
        raise ValueError("Standard program inventory differs from exact eligible printing identities")
    for program in program_rows:
        required.update(full_programs[program]["requiredCaseIds"])
    cases = [full_cases[case_id] for case_id in sorted(required)]
    missing_ids = sorted(row["cardSourceID"] for row in pool["sourceMissing"])
    if (missing_ids != sorted(pool["sourceMissingIDs"]) or len(missing_ids) != len(set(missing_ids))
            or set(missing_ids) & set(source_ids)
            or len(source_ids) + len(missing_ids) != pool["counts"]["eligibleCatalogRows"]):
        raise ValueError("Missing Standard printings must stay in the eligible denominator")
    counts = Counter(row["status"] for row in cases)
    missing_cases = [{"caseId": f"catalog-printing/{source_id}/metadata_identity", "status": "source_missing",
                      "cardSourceID": source_id, "facet": "metadata_identity"} for source_id in missing_ids]
    counts.update(row["status"] for row in missing_cases)
    current_unverified = pool["liveServerUnverifiedSourceIDs"]
    if not set(current_unverified) <= set(source_ids) | set(missing_ids):
        raise ValueError("Live freshness uncertainty names a source outside the Standard pool")
    snapshot_qualified = (counts["oracle_matched"] == len(cases) + len(missing_cases)
                          and not pool["unknowns"] and not pool["activeProgramAmbiguities"])
    programs = [{**full_programs[key], "cardSourceIDs": sorted(program_rows[key])}
                for key in sorted(program_rows)]
    row_counts = Counter(row["status"] for row in selected_rows)
    if missing_ids:
        row_counts["source_missing"] = len(missing_ids)
    return {
        "schema": "current-standard-parity/1", "asOf": pool["asOf"], "engine": full_report["engine"],
        "sourcePins": full_report["sourcePins"], "poolStatus": pool["status"],
        "denominator": {"eligiblePrintings": len(source_ids) + len(missing_ids),
                        "rawSourcePrintings": len(source_ids), "sourceMissingPrintings": len(missing_ids),
                        "programs": len(program_rows), "retainedHistoricalRequirements": len(cases),
                        "additionalMissingPrintingRequirements": len(missing_cases),
                        "requiredCases": len(cases) + len(missing_cases),
                        "sourceIdsSha256": digest(sorted(source_ids + missing_ids)),
                        "caseIdsSha256": digest(sorted(required | {row["caseId"] for row in missing_cases}))},
        "caseStatusCounts": dict(sorted(counts.items())),
        "rowStatusCounts": dict(sorted(row_counts.items())),
        "programStatusCounts": dict(sorted(Counter(row["status"] for row in programs).items())),
        "cachedSnapshotParityQualified": snapshot_qualified,
        "fullCurrentStandardParityQualified": snapshot_qualified and not current_unverified,
        "currentServerUnverifiedPrintings": len(current_unverified),
        "sourceMissing": pool["sourceMissing"], "unknowns": pool["unknowns"],
        "activeProgramAmbiguities": pool["activeProgramAmbiguities"],
        "sourceSnapshots": pool["sourceSnapshots"],
        "measurementScope": "Projection of the pinned full native measurement; no new native execution or inherited engine statuses.",
        "rulingsPassDeferred": True, "stateSpaceEquivalenceProven": False,
        "rows": selected_rows, "programs": programs, "cases": cases + missing_cases,
    }
