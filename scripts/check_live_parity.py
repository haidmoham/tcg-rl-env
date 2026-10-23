"""Check the pinned full private Live dump without importing a native engine by default."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tcg.live_parity import (LIVE_PINS, compare_baseline, digest,
                             inventory, load_corpus, private_output, read_pinned)
from tcg.parity_runner import execution_manifest, run_native_fixtures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--inventory-index", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--library-sha256", required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--aliases", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--allow-engine-change", action="store_true",
                        help="compare a new candidate engine against identical frozen source and case requirements")
    parser.add_argument("--behavior-manifest", type=Path)
    parser.add_argument("--behavior-sha256")
    runners = parser.add_mutually_exclusive_group()
    runners.add_argument("--case-runner", type=Path, help="custom runner comparisons remain provisional")
    runners.add_argument("--builtin-runner", action="store_true", help="run reviewed source-native fixture handlers")
    parser.add_argument("--runner-sha256")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = private_output(ROOT, args.output)
        fixture_args = (args.behavior_manifest, args.behavior_sha256, args.library)
        if any((*fixture_args, args.case_runner, args.builtin_runner, args.runner_sha256)):
            if not all(fixture_args) or not (args.case_runner or args.builtin_runner):
                raise ValueError("fixture verification requires manifest, manifest hash, library, and a runner")
            if bool(args.case_runner) != bool(args.runner_sha256):
                raise ValueError("custom runner requires its hash; builtin runner uses reviewed repository code")
        if args.allow_engine_change and not args.baseline:
            raise ValueError("--allow-engine-change requires an explicit frozen --baseline")
        bundle = json.loads(args.bundle_manifest.read_text(encoding="utf-8"))
        receipt = json.loads(args.inventory_index.read_text(encoding="utf-8"))
        full = bundle.get("fullCorpus", {})
        if (full.get("sha256") != LIVE_PINS.corpus_sha256 or full.get("cards") != LIVE_PINS.rows
                or receipt.get("corpusSha256") != LIVE_PINS.corpus_sha256
                or receipt.get("indexSha256") != LIVE_PINS.index_sha256
                or receipt.get("cards") != LIVE_PINS.rows):
            raise ValueError("bundle manifest and inventory receipt must bind the full pinned corpus/index")
        rows = load_corpus(args.corpus, args.index)
        registry_bytes = args.registry.read_bytes()
        registry = json.loads(registry_bytes)
        aliases_bytes = args.aliases.read_bytes() if args.aliases else None
        aliases = json.loads(aliases_bytes) if aliases_bytes is not None else []
        if isinstance(aliases, dict):
            expected = {"corpusSha256": LIVE_PINS.corpus_sha256, "indexSha256": LIVE_PINS.index_sha256,
                        "registrySnapshotSha256": sha256(registry_bytes).hexdigest()}
            if (aliases.get("schema") != "live-printing-identities/1"
                    or any(aliases.get("inputPins", {}).get(key) != value for key, value in expected.items())
                    or aliases.get("librarySha256") != args.library_sha256
                    or aliases.get("registrySha256") != args.registry_sha256
                    or aliases.get("engineVersion") != registry["engineVersion"]):
                raise ValueError("identity mapping envelope does not bind the current corpus and candidate registry")
            aliases = aliases["aliases"]
        engine = {"version": registry.get("engineVersion"), "sourceCommit": args.engine_commit,
                  "librarySha256": args.library_sha256, "registrySha256": args.registry_sha256}
        manifest, observations = None, None
        native_execution, execution, runner_sha256 = None, None, None
        if args.case_runner or args.builtin_runner:
            manifest = json.loads(read_pinned(args.behavior_manifest, args.behavior_sha256))
            read_pinned(args.library, args.library_sha256)
            if (manifest.get("schema") != "live-parity-fixtures/1" or manifest.get("engine") != engine
                    or manifest.get("sourcePins") != asdict(LIVE_PINS)):
                raise ValueError("fixture manifest does not bind the pinned source and engine")
            execution = execution_manifest(manifest)
            runner_sha256 = args.runner_sha256
            if args.builtin_runner:
                native_execution = run_native_fixtures(
                    execution, manifest["candidateBuild"], args.behavior_manifest.parent, registry, rows)
                observations = {"cases": native_execution.cases}
                runner_sha256 = sha256((ROOT / "tcg" / "parity_runner.py").read_bytes()).hexdigest()
            else:
                read_pinned(args.case_runner, args.runner_sha256)
                output.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="parity-execution-", dir=output.parent) as directory:
                    execution_path = Path(directory) / "execution.json"
                    execution_path.write_text(json.dumps(execution), encoding="utf-8")
                    completed = subprocess.run(
                        [sys.executable, str(args.case_runner.resolve()), "--manifest", str(execution_path),
                         "--library", str(args.library.resolve()), "--registry", str(args.registry.resolve())],
                        capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)
                observations = json.loads(completed.stdout)
                if (observations.get("manifestSha256") != digest(execution)
                        or observations.get("engine") != engine):
                    raise ValueError("custom runner observations must bind the sanitized execution manifest")
            observations = {"manifestSha256": digest(manifest), "engine": engine,
                            "cases": observations["cases"]}
        report = inventory(rows, registry, engine_commit=args.engine_commit,
                           library_sha256=args.library_sha256, registry_sha256=args.registry_sha256,
                           aliases=aliases, manifest=manifest, observations=observations,
                           manifest_root=args.behavior_manifest.parent if manifest is not None else None,
                           runner_sha256=runner_sha256, native_execution=native_execution)
        if execution is not None:
            report["fixtureVerification"]["executionManifestSha256"] = digest(execution)
        report["sourceReceipts"] = {"bundleManifestSha256": sha256(args.bundle_manifest.read_bytes()).hexdigest(),
                                    "inventoryIndexSha256": sha256(args.inventory_index.read_bytes()).hexdigest()}
        if args.baseline:
            baseline_bytes = args.baseline.read_bytes()
            baseline = json.loads(baseline_bytes)
            if baseline.get("sourceReceipts") != report["sourceReceipts"]:
                raise ValueError("source receipt pins differ from frozen baseline")
            compare_baseline(report, baseline, allow_engine_change=args.allow_engine_change)
            report["baselineSha256"] = sha256(baseline_bytes).hexdigest()
        report["measurementInputs"] = {
            "registrySnapshotSha256": sha256(registry_bytes).hexdigest(),
            "aliasesSha256": sha256(aliases_bytes).hexdigest() if aliases_bytes is not None else None,
            "behaviorManifestSha256": args.behavior_sha256,
            "baselineSha256": report.get("baselineSha256"),
            "sourceFiles": {relative: sha256((ROOT / relative).read_bytes()).hexdigest()
                            for relative in ("scripts/check_live_parity.py", "tcg/live_parity.py",
                                             "tcg/parity_runner.py", "tcg/energy_parity.py",
                                             "tcg/flat_rules_parity.py",
                                             "tcg/movement_parity.py",
                                             "tcg/evolved_parity.py",
                                             "tcg/draw_parity.py",
                                             "tcg/dragapult_parity.py",
                                             "tcg/recoil_parity.py",
                                             "tcg/flat_draw_parity.py",
                                             "tcg/native_metadata_parity.py",
                                             "tcg/standard_mechanics_parity.py",
                                             "tcg/standard_legal_prompts_parity.py",
                                             "tcg/standard_effect_parity.py",
                                             "tcg/standard_missing_pool_parity.py",
                                             "tcg/standard_privacy_parity.py",
                                             "tcg/candidate_replay.py",
                                             "tcg/standard_coverage.py")}}
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    except (ValueError, OSError, KeyError, TypeError, AttributeError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps({key: report[key] for key in ("denominator", "rowStatusCounts", "programStatusCounts",
                                                  "caseStatusCounts", "fullParityQualified", "qualificationGate")},
                     indent=2))
    print(f"private report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
