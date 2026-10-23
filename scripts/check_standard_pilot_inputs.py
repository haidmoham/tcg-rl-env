"""Check pinned private Standard manifests without claiming engine readiness."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def pinned(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("input differs from the declared immutable handoff")
    return json.loads(data)


def check(args):
    pool = pinned(args.pool, args.pool_sha256)
    deck = pinned(args.deck, args.deck_sha256)
    if pool["schema"] != "current-live-standard-pool/1":
        raise ValueError("unsupported legality handoff schema")
    if not isinstance(deck, list) or len(deck) != 60 or any(not isinstance(card, str) for card in deck):
        raise ValueError("deck must contain exactly 60 source IDs")
    rows = {row["cardSourceID"]: row for row in pool["rows"]}
    if len(rows) != len(pool["rows"]):
        raise ValueError("pool contains duplicate printing source IDs")
    missing = sorted(set(deck) - set(rows))
    if missing:
        raise ValueError("exact deck has printing IDs absent from the pinned source pool")
    if any(not rows[card]["programSha256"] or not rows[card]["cardSha256"] for card in deck):
        raise ValueError("exact deck contains a printing with missing captured source")
    return {"schema": "standard-pilot-input-handoff/1", "status": "pinned_metadata_verified_training_held",
            "pool_sha256": args.pool_sha256, "exact_source_deck_sha256": args.deck_sha256,
            "pool_as_of": pool["asOf"], "pool_status": pool["status"],
            "pool_counts": pool["counts"], "source_freshness": pool["sourceFreshness"],
            "deck": {"cards": len(deck), "unique_source_ids": len(Counter(deck)),
                     "all_exact_source_ids_in_captured_eligible_pool": True,
                     "all_exact_source_program_hashes_present": True},
            "training_enabled": False,
            "unfulfilled_gates": [
                "Current server overrides/live availability remain unverified by this installed-cache handoff.",
                "Resolve exact source-to-native identity holds; metadata membership cannot admit an alias.",
                "Qualify actual full 60-card mirror games from both seats with natural outcomes and independent seeds.",
                "Qualify C12 actor-visible fastenc rows, ordered prefixes and non-consuming scoring on the exact native build.",
                "Build and qualify dense column-aware learner vocabulary and reader features; native IDs are not teacher embedding indices.",
                "Full pinned current-Standard coverage remains the ranked-lineage gate."
            ],
            "scope": "Source-only metadata/identity check. No native library, model, game, training, current-server query or rank-readiness verification."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--pool-sha256", required=True)
    parser.add_argument("--deck-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = check(args)
    with args.receipt.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "deck": result["deck"]}))
