import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tcg.live_parity import private_output, read_pinned
from tcg.standard_parity import standard_action_measurements, standard_coverage


def main():
    parser = argparse.ArgumentParser(description="Project measured native parity onto a pinned current Standard pool.")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--pool-sha256", required=True)
    parser.add_argument("--full-report", type=Path, required=True)
    parser.add_argument("--full-report-sha256", required=True)
    parser.add_argument("--action-measurements", type=Path)
    parser.add_argument("--action-measurements-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = private_output(ROOT, args.output)
    pool = json.loads(read_pinned(args.pool, args.pool_sha256))
    for snapshot in pool["sourceSnapshots"]:
        if not isinstance(snapshot.get("snapshotPath"), str):
            raise ValueError("Standard source snapshots require immutable copied paths")
        read_pinned(snapshot["snapshotPath"], snapshot["sha256"])
    full = json.loads(read_pinned(args.full_report, args.full_report_sha256))
    report = standard_coverage(pool, full)
    if bool(args.action_measurements) != bool(args.action_measurements_sha256):
        raise ValueError("Action measurements require their pinned receipt hash")
    if args.action_measurements:
        measured = json.loads(read_pinned(args.action_measurements, args.action_measurements_sha256))
        artifacts = {name: json.loads(read_pinned(args.action_measurements.parent / name, pin))
                     for name, pin in measured["outputs"].items()}
        provenance = artifacts["engine-provenance.json"]
        native_build = json.loads(read_pinned(provenance["nativeBuildReceipt"], provenance["nativeBuildReceiptSha256"]))
        registry = json.loads(read_pinned(provenance["registryReceipt"], provenance["registryReceiptSha256"]))
        if (provenance["actualEngineSourceCommit"] != report["engine"]["sourceCommit"]
                or registry["sourceCommit"] != report["engine"]["sourceCommit"]
                or native_build["registryReceiptSha256"] != provenance["registryReceiptSha256"]):
            raise ValueError("Action measurements do not bind the actual candidate native source/object build")
        report["actionMeasurements"] = standard_action_measurements(
            report, measured, artifacts["pairs.json"], artifacts["source-only-actions.json"])
    report["measurementInputs"] = {
        "poolSha256": args.pool_sha256, "fullReportSha256": args.full_report_sha256,
        "sourceFiles": {name: sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in ("scripts/check_standard_parity.py", "tcg/standard_parity.py", "tcg/live_parity.py")}}
    if args.action_measurements:
        report["measurementInputs"]["actionMeasurementsSha256"] = args.action_measurements_sha256
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("denominator", "caseStatusCounts", "rowStatusCounts",
                       "programStatusCounts", "cachedSnapshotParityQualified", "fullCurrentStandardParityQualified")}, indent=2))


if __name__ == "__main__":
    main()
