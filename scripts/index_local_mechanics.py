"""Index private native card programs into an ignored fixture-planning artifact."""

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]+\Z")

from tcg.local_mechanics import build_index, sha256_bytes


def _load_json(path):
    raw = path.read_bytes()
    return json.loads(raw), sha256_bytes(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mapping-receipt", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "local-mechanics-index.json")
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        parser.error("output must be inside this checkout's ignored runs directory")

    config, _ = _load_json(args.config)
    bundle = Path(config["bundleRoot"])
    cards, card_hash = _load_json(bundle / "cardData.json")
    _, game_hash = _load_json(bundle / "gameData.json")
    joins_document, joins_hash = _load_json(bundle / "printing-joins.json")
    if not isinstance(cards, list) or not all(
        isinstance(card, dict) and isinstance(card.get("cardSourceID"), str)
        and isinstance(card.get("logicSourceID"), str)
        and IDENTIFIER.fullmatch(card["cardSourceID"])
        and IDENTIFIER.fullmatch(card["logicSourceID"]) for card in cards
    ):
        parser.error("cardData.json must contain cardSourceID and logicSourceID records")
    joins = joins_document.get("joins")
    if not isinstance(joins, list) or not all(
        isinstance(join, dict) and isinstance(join.get("printingId"), str)
        and IDENTIFIER.fullmatch(join["printingId"])
        and ((isinstance(join.get("cardSourceID"), str)
              and IDENTIFIER.fullmatch(join["cardSourceID"]))
             or (join.get("cardSourceID") is None and join.get("status") == "unresolved"))
        for join in joins
    ):
        parser.error("printing-joins.json has an invalid printingId or source join")
    mapping = {}
    mapping_hash = None
    if args.mapping_receipt:
        receipt, mapping_hash = _load_json(args.mapping_receipt)
        mapping = receipt["inventory"]["mapping"]
        if not isinstance(mapping, dict) or not all(
            isinstance(source_id, str) and IDENTIFIER.fullmatch(source_id)
            and isinstance(item, dict) and isinstance(item.get("cabtId"), int)
            and isinstance(item.get("method"), str) for source_id, item in mapping.items()
        ):
            parser.error("mapping receipt has no valid inventory mapping")

    index = build_index(cards, joins, mapping, card_data_sha256=card_hash,
                        game_data_sha256=game_hash, joins_sha256=joins_hash,
                        mapping_sha256=mapping_hash)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(index, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"schema": index["schema"], "cardPrograms": len(cards),
                      "printings": len(index["printings"]), "mappedCandidates": len(mapping),
                      "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
