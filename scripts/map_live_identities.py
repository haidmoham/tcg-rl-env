"""Write private exact printing candidates without starting either game runtime."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tcg.live_identity import (CORPUS_SHA256, INDEX_SHA256, load_corpus, load_public_metadata,
                               map_identities, private_output, read_reference)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("corpus", "index", "reference-csv", "registry", "config-cache",
                     "catalog-receipt", "join-receipt", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = private_output(ROOT, args.output)
        cards = load_corpus(args.corpus, args.index)
        metadata = load_public_metadata(args.config_cache, args.catalog_receipt, args.join_receipt)
        reference_raw = args.reference_csv.read_bytes()
        registry_raw = args.registry.read_bytes()
        report = map_identities(cards, read_reference(reference_raw), json.loads(registry_raw), metadata)
        report["inputPins"] = {"corpusSha256": CORPUS_SHA256, "indexSha256": INDEX_SHA256,
                               "referenceCsvSha256": sha256(reference_raw).hexdigest(),
                               "registrySnapshotSha256": sha256(registry_raw).hexdigest(),
                               "receipts": metadata["receiptSha256"]}
        report["publicMetadataSnapshot"] = metadata
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"counts": report["counts"], "unresolvedReasonCounts": report["unresolvedReasonCounts"],
                      "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
