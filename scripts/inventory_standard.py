"""Build a private, dated TCG Live Standard admission report from explicit snapshots."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tcg.standard_coverage import (candidate_search_inventory, cohort_inventory, inventory,
                                   load_json, read_deck_pages, read_reference_csv)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path,
                        help="dated legal-printing catalog with source attestations")
    parser.add_argument("--registry", type=Path, required=True,
                        help="CABT card registry snapshot with the pinned AllCard hash")
    parser.add_argument("--aliases", type=Path,
                        help="explicit printing-to-CABT identity candidates and evidence")
    parser.add_argument("--qualifications", type=Path,
                        help="effect and interaction evidence pinned to the engine")
    parser.add_argument("--cohort", type=Path,
                        help="bounded sampled decklists, kept separate from the Standard denominator")
    parser.add_argument("--candidate-page", type=Path, action="append",
                        help="captured full-display Limitless Standard search page; repeat for every page")
    parser.add_argument("--observed-on", help="ISO date when candidate search pages were captured")
    parser.add_argument("--reference-csv", type=Path,
                        help="CABT reference printing table required with --cohort")
    parser.add_argument("--printing-overrides", type=Path,
                        help="source-backed exact printing IDs for name-only cohort rows")
    parser.add_argument("--deck-page", type=Path, action="append",
                        help="captured Limitless deck HTML; repeat for each --cohort tournament list")
    parser.add_argument("--library-sha256", help="pinned CABT library hash required with --cohort")
    parser.add_argument("--registry-sha256", help="pinned AllCard hash required with --cohort")
    parser.add_argument("--editable-registry", type=Path,
                        help="snapshot built from the official editable source release")
    parser.add_argument("--editable-library-sha256")
    parser.add_argument("--editable-registry-sha256")
    parser.add_argument("--editable-source-commit")
    parser.add_argument("--output", type=Path, required=True,
                        help="new private report path under this checkout's runs/")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if not args.output.is_relative_to((ROOT / "runs").resolve()):
        parser.error("output must remain under this checkout's resolved runs/ directory")
    if args.output.exists():
        parser.error("output exists; preserve the previous report and choose a new path")
    if args.cohort and args.candidate_page:
        parser.error("choose either --cohort or --candidate-page")
    if args.deck_page and not args.cohort:
        parser.error("--deck-page requires --cohort")
    if args.candidate_page and not args.observed_on:
        parser.error("--candidate-page requires --observed-on")
    if (args.cohort or args.candidate_page) and not all((args.reference_csv, args.library_sha256,
                                                        args.registry_sha256)):
        parser.error("sampled inventories require --reference-csv, --library-sha256, and --registry-sha256")
    if args.editable_registry and not all((args.editable_library_sha256, args.editable_registry_sha256,
                                           args.editable_source_commit)):
        parser.error("--editable-registry requires both hashes and --editable-source-commit")
    if not args.cohort and not args.candidate_page and not all((args.catalog, args.aliases, args.qualifications)):
        parser.error("full inventory requires --catalog, --aliases, and --qualifications")
    try:
        if args.candidate_page:
            result = candidate_search_inventory(
                args.candidate_page, load_json(args.registry), read_reference_csv(args.reference_csv),
                as_of=args.observed_on, library_sha256=args.library_sha256,
                registry_sha256=args.registry_sha256,
                editable_registry_snapshot=(load_json(args.editable_registry)
                                            if args.editable_registry else None),
                editable_library_sha256=args.editable_library_sha256,
                editable_registry_sha256=args.editable_registry_sha256,
                editable_source_commit=args.editable_source_commit)
        elif args.cohort:
            overrides = (load_json(args.printing_overrides).get("cohortPrintingOverrides", [])
                         if args.printing_overrides else ())
            result = cohort_inventory(load_json(args.cohort), load_json(args.registry),
                                      read_reference_csv(args.reference_csv),
                                      library_sha256=args.library_sha256,
                                      registry_sha256=args.registry_sha256,
                                      printing_overrides=overrides,
                                      editable_registry_snapshot=(load_json(args.editable_registry)
                                                                  if args.editable_registry else None),
                                      editable_library_sha256=args.editable_library_sha256,
                                      editable_registry_sha256=args.editable_registry_sha256,
                                      editable_source_commit=args.editable_source_commit,
                                      deck_pages=(read_deck_pages(args.deck_page)
                                                  if args.deck_page else None))
        else:
            result = inventory(load_json(args.catalog), load_json(args.registry),
                               load_json(args.aliases), load_json(args.qualifications))
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    if args.candidate_page:
        print(json.dumps({"candidatePrintingCount": result["candidatePrintingCount"],
                          "shippedV2": result["shippedV2"]["statusCounts"],
                          "editableSource": result.get("editableSource", {}).get("statusCounts"),
                          "officialDenominatorVerified": result["officialDenominatorVerified"],
                          "output": str(args.output)}, indent=2))
        return 0
    if args.cohort:
        print(json.dumps({"scope": result["scope"],
                          "decks": [{"name": deck["name"], "sourceKind": deck["sourceKind"],
                                     "statusCopies": deck["statusCopies"],
                                     "editableSourceStatusCopies": deck.get("editableSourceStatusCopies")}
                                    for deck in result["decks"]],
                          "output": str(args.output)}, indent=2))
        return 0
    print(json.dumps({"denominator": result["denominator"],
                      "printingStatusCounts": result["printingStatusCounts"],
                      "fullStandardQualified": result["fullStandardQualified"],
                      "output": str(args.output)}, indent=2))
    return 0 if result["fullStandardQualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
