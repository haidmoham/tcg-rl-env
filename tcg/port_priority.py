"""Pure, source-bound priorities for a full-Standard card port."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Source:
    source_id: str
    role: str
    url: str
    observed_on: date
    format_id: str | None
    metric: str | None
    denominator: int | None


@dataclass(frozen=True)
class Card:
    name: str
    count: int
    category: str


@dataclass(frozen=True)
class Archetype:
    name: str
    source_id: str
    rank: int
    points: int | None
    displayed_share_percent: float | None
    list_url: str | None
    list_event: str | None
    list_event_date: date | None
    cards: tuple[Card, ...]


@dataclass(frozen=True)
class Snapshot:
    as_of: date
    format_id: str
    sources: tuple[Source, ...]
    archetypes: tuple[Archetype, ...]


def parse_snapshot(raw):
    """Validate external JSON once and return normalized evidence."""
    if raw.get("schema") != "cabt-standard-port-priority/1":
        raise ValueError("unsupported priority schema")
    as_of = date.fromisoformat(raw["as_of"])
    format_id = raw["format_id"]
    if not isinstance(format_id, str) or not format_id:
        raise ValueError("format_id is required")
    sources = tuple(
        Source(row["id"], row["role"], row["url"], date.fromisoformat(row["observed_on"]),
               row.get("format_id"), row.get("metric"), row.get("denominator"))
        for row in raw["sources"]
    )
    source_by_id = {source.source_id: source for source in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("duplicate source id")
    if any(source.observed_on > as_of for source in sources):
        raise ValueError("source observation is in the future")
    if any(source.denominator is not None and
           (type(source.denominator) is not int or source.denominator <= 0) for source in sources):
        raise ValueError("denominator must be a positive count or unknown")
    archetypes = []
    for row in raw["archetypes"]:
        source = source_by_id[row["source_id"]]
        if source.role != "tournament_ranking" or source.format_id != format_id:
            raise ValueError("archetype ranking requires a matching tournament source")
        if type(row["rank"]) is not int or row["rank"] <= 0:
            raise ValueError("rank must be positive")
        points = row.get("points")
        if points is not None and (type(points) is not int or points < 0):
            raise ValueError("points must be nonnegative or unknown")
        share = row.get("displayed_share_percent")
        if share is not None and (type(share) not in (int, float) or not 0 <= share <= 100):
            raise ValueError("displayed share must be a percentage or unknown")
        representative = row.get("representative_list")
        cards = ()
        list_url = list_event = list_event_date = None
        if representative is not None:
            list_url = representative["url"]
            list_event = representative["event"]
            list_event_date = date.fromisoformat(representative["event_date"])
            if list_event_date > as_of:
                raise ValueError("representative event is in the future")
            cards = tuple(Card(name, count, category)
                          for category, entries in representative["cards"].items()
                          for count, name in entries)
            if any(card.category not in ("pokemon", "trainer", "energy") or
                   type(card.count) is not int or not 1 <= card.count <= (60 if card.category == "energy" else 4) or
                   not isinstance(card.name, str) or not card.name for card in cards):
                raise ValueError("invalid representative card entry")
            if sum(card.count for card in cards) != 60:
                raise ValueError(f"representative list for {row['name']} is not 60 cards")
        archetypes.append(Archetype(row["name"], row["source_id"], row["rank"], points,
                                    share, list_url, list_event, list_event_date, cards))
    if len({row.name for row in archetypes}) != len(archetypes):
        raise ValueError("duplicate archetype")
    for example in raw.get("simulator_example_lists", []):
        if source_by_id[example["source_id"]].role != "simulator_catalog":
            raise ValueError("simulator example requires a simulator catalog source")
        entries = [entry for rows in example["cards"].values() for entry in rows]
        if any(len(entry) != 4 or type(entry[0]) is not int or entry[0] <= 0 or
               not isinstance(entry[1], str) or not entry[1] or
               not isinstance(entry[2], str) or not entry[2] or
               type(entry[3]) is not int or entry[3] <= 0 for entry in entries):
            raise ValueError("invalid exact-printing simulator entry")
        if sum(entry[0] for entry in entries) != 60:
            raise ValueError("simulator example is not 60 cards")
    return Snapshot(as_of, format_id, sources, tuple(archetypes))


def prioritize(snapshot, *, as_of=None, max_age_days=14):
    """Rank observed lists without merging unlike sources or limiting the legal pool."""
    as_of = as_of or snapshot.as_of
    if as_of < snapshot.as_of or max_age_days < 0:
        raise ValueError("ranking date or freshness limit is invalid")
    ranking_sources = {source.source_id: source for source in snapshot.sources
                       if source.role == "tournament_ranking" and source.format_id == snapshot.format_id
                       and 0 <= (as_of - source.observed_on).days <= max_age_days}
    eligible = sorted((row for row in snapshot.archetypes if row.source_id in ranking_sources),
                      key=lambda row: (row.rank, row.name))
    card_evidence = defaultdict(list)
    for deck in eligible:
        deck_card_counts = defaultdict(int)
        for card in deck.cards:
            if card.category == "energy" and card.name in {
                "Grass Energy", "Fire Energy", "Water Energy", "Lightning Energy",
                "Psychic Energy", "Fighting Energy", "Darkness Energy", "Metal Energy"
            }:
                continue
            deck_card_counts[card.name] += card.count
        for name, count in deck_card_counts.items():
            card_evidence[name].append((deck.rank, deck.name, count, deck.list_url))
    cards = []
    for name, observations in card_evidence.items():
        rows = sorted(observations)
        cards.append({"card": name, "sampled_archetype_count": len(rows),
                      "best_rank": rows[0][0], "sampled_copies": sum(row[2] for row in rows),
                      "lists": [{"archetype": row[1], "rank": row[0], "copies": row[2],
                                 "url": row[3]} for row in rows]})
    cards.sort(key=lambda row: (-row["sampled_archetype_count"], row["best_rank"],
                                -row["sampled_copies"], row["card"]))
    return {
        "as_of": as_of.isoformat(), "format_id": snapshot.format_id,
        "scope": "port_order_only_full_legal_pool_remains_required",
        "identity_resolution": "card names in representative lists; exact printings need separate source-to-engine resolution",
        "ranking_sources": [{"id": source.source_id, "url": source.url,
                             "observed_on": source.observed_on.isoformat(),
                             "format_id": source.format_id, "metric": source.metric,
                             "denominator": source.denominator}
                            for source in ranking_sources.values()],
        "ranking_source_ids": sorted(ranking_sources),
        "stale_or_unavailable_ranking_source_ids": sorted(source.source_id for source in snapshot.sources
            if source.role == "tournament_ranking" and source.source_id not in ranking_sources),
        "archetypes": [{"name": row.name, "rank": row.rank, "points": row.points,
                        "displayed_share_percent": row.displayed_share_percent,
                        "source_id": row.source_id, "representative_list_url": row.list_url,
                        "representative_event": row.list_event,
                        "representative_event_date": (row.list_event_date.isoformat()
                                                      if row.list_event_date else None)}
                       for row in eligible],
        "unsampled_archetypes": [row.name for row in eligible if not row.cards],
        "cards": cards,
    }
