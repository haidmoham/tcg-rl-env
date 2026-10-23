"""Print reproducible port priorities from a dated public-source ledger."""

import argparse
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tcg.port_priority import parse_snapshot, prioritize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path,
                        default=ROOT / "experiments/cabt-standard-priority-sources.json")
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--max-age-days", type=int, default=14)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    if args.top < 0:
        parser.error("--top must be nonnegative")
    snapshot = parse_snapshot(json.loads(args.sources.read_text(encoding="utf-8")))
    report = prioritize(snapshot, as_of=args.as_of, max_age_days=args.max_age_days)
    report["cards"] = report["cards"][:args.top]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
