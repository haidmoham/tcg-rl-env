from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit


SOURCE_KINDS = frozenset({"rotation", "card_database", "release_calendar", "promo_legality",
                          "live_exclusions", "errata", "rules"})
SOURCE_TYPES = SOURCE_KINDS | {"cabt_reference", "cohort"}
QUALIFICATION_FACETS = frozenset({"effect", "legal_prompts", "information_boundaries", "interactions"})
STATUSES = frozenset({"absent", "identity_unresolved", "present_untested", "effect_qualified",
                      "rule_divergent"})


@dataclass(frozen=True)
class EnginePin:
    version: int
    wrapper_commit: str
    library_sha256: str
    registry_sha256: str


@dataclass(frozen=True)
class Printing:
    official_id: str
    regulation_mark: str
    effect_identity: str
    effect_text_sha256: str
    legal_from: str
    legal_until: str | None
    source_ids: tuple[str, ...]
    required_cases: tuple[str, ...]


@dataclass(frozen=True)
class RegistryCard:
    card_id: int
    printing_id: str | None
    effect_text_sha256: str | None


@dataclass(frozen=True)
class Alias:
    printing_id: str
    card_id: int
    method: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class Qualification:
    effect_identity: str
    effect_text_sha256: str
    engine: EnginePin
    legality_as_of: str
    facets: tuple[str, ...]
    case_ids: tuple[str, ...]
    evidence_paths: tuple[str, ...]
    result: str


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _digest(value, label):
    value = _string(value, label).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _strings(value, label):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    strings = tuple(_string(item, label) for item in value)
    if len(strings) != len(set(strings)):
        raise ValueError(f"{label} contains duplicates")
    return strings


def _date(value, label):
    value = _string(value, label)
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO date") from error
    return value


def _engine(value):
    value = _object(value, "engine")
    version = value.get("version")
    if type(version) is not int or version < 1:
        raise ValueError("engine version must be a positive integer")
    return EnginePin(version, _string(value.get("wrapperCommit"), "wrapperCommit"),
                     _digest(value.get("librarySha256"), "librarySha256"),
                     _digest(value.get("registrySha256"), "registrySha256"))


def _sources(value):
    if not isinstance(value, list):
        raise ValueError("sources must be an array")
    sources = {}
    for source in value:
        source = _object(source, "source")
        identifier = _string(source.get("id"), "source id")
        if identifier in sources:
            raise ValueError(f"duplicate source {identifier}")
        kind = _string(source.get("kind"), "source kind")
        if kind not in SOURCE_TYPES:
            raise ValueError(f"unknown source kind {kind}")
        url = _string(source.get("url"), "source url")
        if not url.startswith("https://"):
            raise ValueError("source URL must use HTTPS")
        authoritative = source.get("authoritative") is True
        host = urlsplit(url).hostname
        if authoritative and (host is None or not host.endswith(".pokemon.com")):
            raise ValueError(f"authoritative source must be hosted by Pokemon: {identifier}")
        sources[identifier] = {"kind": kind, "url": url,
                               "observedAt": _date(source.get("observedAt"), "observedAt"),
                               "authoritative": authoritative,
                               "completeForKind": source.get("completeForKind") is True}
    return sources


def _printings(value, sources):
    if not isinstance(value, list):
        raise ValueError("legalPrintings must be an array")
    rows = []
    ids = set()
    for raw in value:
        raw = _object(raw, "legal printing")
        official_id = _string(raw.get("officialPrintingId"), "officialPrintingId")
        if official_id in ids:
            raise ValueError(f"duplicate official printing {official_id}")
        ids.add(official_id)
        mark = _string(raw.get("regulationMark"), "regulationMark")
        if len(mark) != 1 or not mark.isupper():
            raise ValueError(f"invalid regulation mark for {official_id}")
        refs = _strings(raw.get("legalitySourceIds"), "legalitySourceIds")
        if not refs or not set(refs) <= sources.keys():
            raise ValueError(f"unresolved legality source for {official_id}")
        legal_until = raw.get("liveLegalUntil")
        if legal_until is not None:
            legal_until = _date(legal_until, "liveLegalUntil")
        rows.append(Printing(official_id, mark,
                             _string(raw.get("canonicalEffectIdentity"), "canonicalEffectIdentity"),
                             _digest(raw.get("effectTextSha256"), "effectTextSha256"),
                             _date(raw.get("liveLegalFrom"), "liveLegalFrom"), legal_until,
                             refs, _strings(raw.get("requiredCaseIds", []), "requiredCaseIds")))
    return tuple(rows)


def _registry(value, expected_hash):
    value = _object(value, "registry")
    schema = value.get("schema")
    if schema != "cabt-registry-snapshot/1":
        raise ValueError("unsupported registry schema")
    digest_field = "rawSha256"
    if _digest(value.get(digest_field), digest_field) != expected_hash:
        raise ValueError("registry artifact hash differs from engine pin")
    cards = value.get("entries")
    if not isinstance(cards, list):
        raise ValueError("registry cards must be an array")
    actual = sha256(json.dumps(cards, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if actual != expected_hash:
        raise ValueError("registry entries do not match the pinned AllCard hash")
    result = {}
    for raw in cards:
        raw = _object(raw, "registry card")
        card_id = raw.get("cardId")
        if type(card_id) is not int or card_id < 0 or card_id in result:
            raise ValueError("CABT card IDs must be unique nonnegative integers")
        printing_id = raw.get("officialPrintingId")
        digest = raw.get("effectTextSha256")
        result[card_id] = RegistryCard(card_id,
                                       _string(printing_id, "officialPrintingId") if printing_id is not None else None,
                                       _digest(digest, "effectTextSha256") if digest is not None else None)
    return result


def _aliases(value, printing_ids, sources):
    if not isinstance(value, list):
        raise ValueError("aliases must be an array")
    result = defaultdict(list)
    for raw in value:
        raw = _object(raw, "alias")
        printing_id = _string(raw.get("officialPrintingId"), "officialPrintingId")
        if printing_id not in printing_ids:
            raise ValueError(f"alias references unknown printing {printing_id}")
        card_id = raw.get("cabtCardId")
        if type(card_id) is not int or card_id < 0:
            raise ValueError("cabtCardId must be a nonnegative integer")
        method = _string(raw.get("method"), "alias method")
        if method not in {"exact_printing", "effect_equivalent", "name_candidate"}:
            raise ValueError(f"unknown alias method {method}")
        refs = _strings(raw.get("identitySourceIds", []), "identitySourceIds")
        if not set(refs) <= sources.keys():
            raise ValueError(f"alias has unresolved identity source for {printing_id}")
        result[printing_id].append(Alias(printing_id, card_id, method, refs))
    return result


def _qualifications(value):
    if not isinstance(value, list):
        raise ValueError("qualifications must be an array")
    result = {}
    for raw in value:
        raw = _object(raw, "qualification")
        identity = _string(raw.get("canonicalEffectIdentity"), "canonicalEffectIdentity")
        if identity in result:
            raise ValueError(f"duplicate qualification for {identity}")
        facets = _strings(raw.get("facets", []), "facets")
        if not set(facets) <= QUALIFICATION_FACETS:
            raise ValueError(f"unknown qualification facet for {identity}")
        paths = _strings(raw.get("evidencePaths", []), "evidencePaths")
        if any(Path(path).is_absolute() or ".." in Path(path).parts for path in paths):
            raise ValueError("evidence paths must be relative and stay within the evidence root")
        result[identity] = Qualification(identity, _digest(raw.get("effectTextSha256"), "effectTextSha256"),
                                         _engine(raw.get("engine")),
                                         _date(raw.get("legalityAsOf"), "legalityAsOf"), facets,
                                         _strings(raw.get("caseIds", []), "caseIds"), paths,
                                         _string(raw.get("result"), "qualification result"))
    return result


def _verified_alias(alias, printing, card, sources):
    if card is None:
        return False
    if alias.method == "exact_printing":
        return bool(alias.source_ids) and card.printing_id == printing.official_id
    if alias.method == "effect_equivalent":
        return (card.effect_text_sha256 == printing.effect_text_sha256 and bool(alias.source_ids)
                and all(sources[ref]["authoritative"] for ref in alias.source_ids))
    return False


def _qualification_claim_matches(qualification, printing, engine, as_of):
    return bool(qualification and qualification.result == "passed" and qualification.engine == engine
                and qualification.legality_as_of == as_of
                and qualification.effect_text_sha256 == printing.effect_text_sha256
                and QUALIFICATION_FACETS <= set(qualification.facets)
                and set(printing.required_cases) <= set(qualification.case_ids)
                and qualification.evidence_paths)


def inventory(catalog, registry_data, alias_data, qualification_data):
    catalog = _object(catalog, "catalog")
    if catalog.get("schema") != "tcgl-standard-catalog/1":
        raise ValueError("unsupported catalog schema")
    as_of = _date(catalog.get("asOf"), "asOf")
    engine = _engine(catalog.get("engine"))
    sources = _sources(catalog.get("sources"))
    printings = _printings(catalog.get("legalPrintings"), sources)
    registry = _registry(registry_data, engine.registry_sha256)
    aliases = _aliases(alias_data, {printing.official_id for printing in printings}, sources)
    qualifications = _qualifications(qualification_data)
    rules = catalog.get("ruleGates")
    if not isinstance(rules, list):
        raise ValueError("ruleGates must be an array")
    rule_rows = []
    for rule in rules:
        rule = _object(rule, "rule gate")
        refs = _strings(rule.get("sourceIds", []), "rule sourceIds")
        if not refs or not set(refs) <= sources.keys():
            raise ValueError("rule gate needs known source evidence")
        status = _string(rule.get("status"), "rule status")
        if status not in {"present_untested", "effect_qualified", "rule_divergent"}:
            raise ValueError("unsupported rule gate status")
        pin = _engine(rule["engine"]) if status == "effect_qualified" else None
        if status == "effect_qualified" and (pin != engine
                                              or _date(rule.get("legalityAsOf"), "legalityAsOf") != as_of
                                              or not _strings(rule.get("evidencePaths", []), "evidencePaths")):
            status = "present_untested"
        claimed_status = status
        if status == "effect_qualified":
            status = "present_untested"
        rule_rows.append({"id": _string(rule.get("id"), "rule id"), "status": status,
                          "claimedStatus": claimed_status})
    if len({row["id"] for row in rule_rows}) != len(rule_rows):
        raise ValueError("duplicate rule gate")

    effects = defaultdict(list)
    for printing in printings:
        if printing.legal_from > as_of or (printing.legal_until is not None and as_of >= printing.legal_until):
            raise ValueError(f"catalog includes printing outside Live legal dates: {printing.official_id}")
        effects[printing.effect_identity].append(printing)
    for identity, group in effects.items():
        if len({printing.effect_text_sha256 for printing in group}) != 1:
            raise ValueError(f"effect identity has conflicting text hashes: {identity}")

    rows = []
    for printing in sorted(printings, key=lambda item: item.official_id):
        candidates = aliases[printing.official_id]
        verified = sorted({alias.card_id for alias in candidates
                           if _verified_alias(alias, printing, registry.get(alias.card_id), sources)})
        if len(verified) > 1:
            status = "identity_unresolved"
        elif verified:
            status = "present_untested"
        else:
            status = "identity_unresolved" if candidates else "absent"
        rows.append({"officialPrintingId": printing.official_id,
                     "canonicalEffectIdentity": printing.effect_identity,
                     "qualificationClaimMatchesPins": _qualification_claim_matches(
                         qualifications.get(printing.effect_identity), printing, engine, as_of),
                     "status": status, "cabtCardId": verified[0] if len(verified) == 1 else None})

    complete_kinds = sorted(kind for kind in SOURCE_KINDS if any(
        source["kind"] == kind and source["authoritative"] and source["completeForKind"]
        and source["observedAt"] == as_of for source in sources.values()))
    attestation = catalog.get("denominatorAttestation")
    attested = False
    if attestation is not None:
        attestation = _object(attestation, "denominatorAttestation")
        refs = _strings(attestation.get("sourceIds"), "denominator sourceIds")
        evidence_path = _string(attestation.get("evidencePath"), "denominator evidencePath")
        if (not refs or not set(refs) <= sources.keys() or Path(evidence_path).is_absolute()
                or ".." in Path(evidence_path).parts):
            raise ValueError("denominator attestation needs known sources and a relative evidence path")
        _digest(attestation.get("evidenceSha256"), "denominator evidenceSha256")
        expected_refs = {identifier for identifier, source in sources.items() if source["kind"] in SOURCE_KINDS}
        attested = (type(attestation.get("legalPrintingCount")) is int
                    and attestation["legalPrintingCount"] == len(printings)
                    and type(attestation.get("legalEffectIdentityCount")) is int
                    and attestation["legalEffectIdentityCount"] == len(effects)
                    and set(refs) == expected_refs)
    denominator_ready = False
    # Source declarations are candidates until independent artifact and required-rule verification exists.
    effect_rows = []
    for identity, group in sorted(effects.items()):
        statuses = {row["status"] for row in rows if row["canonicalEffectIdentity"] == identity}
        status = next(candidate for candidate in ("rule_divergent", "absent", "identity_unresolved",
                                                   "present_untested", "effect_qualified") if candidate in statuses)
        effect_rows.append({"canonicalEffectIdentity": identity, "status": status,
                            "legalPrintingCount": len(group)})
    fully_qualified = (denominator_ready and all(row["status"] == "effect_qualified" for row in rows)
                       and all(row["status"] == "effect_qualified" for row in rule_rows))
    return {"schema": "tcgl-standard-inventory/1", "asOf": as_of,
            "engine": {"version": engine.version, "wrapperCommit": engine.wrapper_commit,
                       "librarySha256": engine.library_sha256, "registrySha256": engine.registry_sha256},
            "denominator": {"status": "attested" if denominator_ready else "unresolved",
                            "missingSourceKinds": sorted(SOURCE_KINDS - set(complete_kinds)),
                            "verificationStatus": "independent_artifact_and_required_rules_unverified",
                            "declaredAttestationMatchesCounts": attested,
                            "sourceDeclarationsComplete": set(complete_kinds) == SOURCE_KINDS,
                            "legalPrintingCount": len(rows) if denominator_ready else None,
                            "legalEffectIdentityCount": len(effect_rows) if denominator_ready else None,
                            "candidatePrintingCount": len(rows), "candidateEffectIdentityCount": len(effect_rows)},
            "printingStatusCounts": dict(sorted(Counter(row["status"] for row in rows).items())),
            "effectStatusCounts": dict(sorted(Counter(row["status"] for row in effect_rows).items())),
            "legalPrintings": rows, "effectIdentities": effect_rows, "ruleGates": rule_rows,
            "fullStandardQualified": fully_qualified,
            "coveragePercent": (100.0 if fully_qualified else None)}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _name_key(name):
    key = (unicodedata.normalize("NFKC", name).replace("’", "'").replace("é", "e")
           .replace("É", "e").casefold().strip())
    energy_names = {"grass energy": "basic {g} energy", "fire energy": "basic {r} energy",
                    "water energy": "basic {w} energy", "lightning energy": "basic {l} energy",
                    "psychic energy": "basic {p} energy", "fighting energy": "basic {f} energy",
                    "darkness energy": "basic {d} energy", "metal energy": "basic {m} energy",
                    "telepathic psychic energy": "telepath psychic energy",
                    "growing grass energy": "grow grass energy"}
    return energy_names.get(key, key)


def read_reference_csv(path):
    result = defaultdict(set)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if not row["Card ID"].isdigit():
                continue
            identifier = f'{row["Expansion"]}-{row["Collection No."]}'
            result[identifier].add((int(row["Card ID"]), _name_key(row["Card Name"])))
    return result


def _cohort_registry(snapshot, library_sha256, registry_sha256):
    snapshot = _object(snapshot, "registry snapshot")
    if snapshot.get("schema") != "cabt-registry-snapshot/1":
        raise ValueError("cohort report requires a pinned CABT registry snapshot")
    if (_digest(snapshot.get("librarySha256"), "librarySha256") != _digest(library_sha256, "librarySha256")
            or _digest(snapshot.get("rawSha256"), "rawSha256") != _digest(registry_sha256, "registrySha256")):
        raise ValueError("cohort registry snapshot differs from the requested CABT engine pin")
    cards = snapshot.get("entries")
    if not isinstance(cards, list) or not cards:
        raise ValueError("registry snapshot has no cards")
    actual = sha256(json.dumps(cards, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if actual != registry_sha256:
        raise ValueError("registry entries do not match the pinned AllCard hash")
    by_id = {card["cardId"]: card for card in cards}
    if len(by_id) != len(cards):
        raise ValueError("registry card IDs are duplicated")
    by_name = defaultdict(set)
    for card in cards:
        by_name[_name_key(_string(card.get("name"), "CABT card name"))].add(card["cardId"])
    return by_id, by_name


def _classify_cohort_card(name_key, printing_id, by_id, by_name, reference_printings):
    named_candidates = set(by_name[name_key])
    exact = set()
    conflict = False
    if printing_id is not None:
        for card_id, reference_name in reference_printings.get(printing_id, ()):
            if card_id not in by_id:
                continue
            if reference_name == name_key and _name_key(by_id[card_id]["name"]) == name_key:
                exact.add(card_id)
            else:
                conflict = True
    if len(exact) == 1:
        return "present_untested", "exact_printing_join", exact
    if len(exact) > 1:
        return "identity_unresolved", "ambiguous_exact_printing", exact
    if conflict:
        return "identity_unresolved", "printing_name_conflict", named_candidates
    if named_candidates:
        reason = "printing_unmapped_name_candidate" if printing_id else "name_only_candidate"
        return "identity_unresolved", reason, named_candidates
    reason = "exact_printing_absent_from_registry" if printing_id else "no_name_match_in_registry"
    return "absent", reason, set()


def read_deck_pages(paths):
    result = {}
    pattern = re.compile(
        r'<div class="decklist-card" data-set="([^"]+)" data-number="([^"]+)"[^>]*>'
        r'\s*<a class="card-link" href="/cards/[^"]+">\s*'
        r'<span class="card-count">([^<]+)</span>\s*<span class="card-name">([^<]+)</span>')
    for path in paths:
        path = Path(path)
        deck_id = re.search(r'(\d+)(?:\.html?)$', path.name)
        if deck_id is None:
            raise ValueError(f"deck page filename needs the Limitless list ID: {path}")
        url = f"https://limitlesstcg.com/decks/list/{deck_id.group(1)}"
        if url in result:
            raise ValueError(f"duplicate deck page {url}")
        raw = path.read_bytes()
        rows = [(int(count), unescape(name), set_code, number)
                for set_code, number, count, name in pattern.findall(raw.decode("utf-8"))]
        if not rows or sum(row[0] for row in rows) != 60:
            raise ValueError(f"deck page does not contain 60 parsed cards: {path}")
        result[url] = {"rows": rows, "htmlSha256": sha256(raw).hexdigest()}
    return result


def cohort_inventory(cohort, registry_snapshot, reference_printings, *, library_sha256,
                     registry_sha256, printing_overrides=(), editable_registry_snapshot=None,
                     editable_library_sha256=None, editable_registry_sha256=None,
                     editable_source_commit=None, deck_pages=None):
    cohort = _object(cohort, "cohort")
    if cohort.get("schema") != "cabt-standard-port-priority/1":
        raise ValueError("unsupported cohort schema")
    cards_by_id, by_name = _cohort_registry(registry_snapshot, library_sha256, registry_sha256)
    editable = None
    if editable_registry_snapshot is not None:
        if not all((editable_library_sha256, editable_registry_sha256, editable_source_commit)):
            raise ValueError("editable source snapshot needs both hashes and a source commit")
        editable = _cohort_registry(editable_registry_snapshot, editable_library_sha256,
                                    editable_registry_sha256)
    overrides = {}
    for override in printing_overrides:
        override = _object(override, "printing override")
        key = (_string(override.get("deckUrl"), "deckUrl"),
               _name_key(_string(override.get("name"), "override name")))
        if key in overrides:
            raise ValueError("duplicate cohort printing override")
        overrides[key] = _string(override.get("officialPrintingId"), "officialPrintingId")

    decks = []
    tournament_lists = [(row["name"], "tournament_name_only", row["representative_list"])
                        for row in cohort.get("archetypes", []) if "representative_list" in row]
    simulator_lists = [(row["name"], "simulator_exact_printing", row)
                       for row in cohort.get("simulator_example_lists", [])]
    if deck_pages is not None and set(deck_pages) != {listing["url"] for _, _, listing in tournament_lists}:
        raise ValueError("captured deck pages must match every sampled tournament list")
    for deck_name, source_kind, listing in tournament_lists + simulator_lists:
        rows = []
        captured = deck_pages.get(listing["url"]) if deck_pages is not None else None
        captured_index = 0
        if captured is not None:
            source_kind = "tournament_exact_printing"
        for group in ("pokemon", "trainer", "energy"):
            for item in listing["cards"][group]:
                if len(item) not in (2, 4):
                    raise ValueError("cohort card row needs name/count and optional exact printing")
                copies, name = item[:2]
                if type(copies) is not int or copies < 1:
                    raise ValueError("cohort copies must be positive integers")
                key = _name_key(_string(name, "cohort card name"))
                if captured is not None:
                    captured_row = captured["rows"][captured_index]
                    captured_index += 1
                    if (copies, name) != captured_row[:2]:
                        raise ValueError(f"captured deck printing differs from listed card: {deck_name}, {name}")
                    printing_id = f"{captured_row[2]}-{captured_row[3]}"
                else:
                    printing_id = (f"{item[2]}-{item[3]}" if len(item) == 4 else
                                   overrides.get((listing["url"], key)))
                status, reason, candidates = _classify_cohort_card(
                    key, printing_id, cards_by_id, by_name, reference_printings)
                row = {"name": name, "printingId": printing_id, "copies": copies,
                       "group": group, "status": status, "reason": reason,
                       "cabtCandidateIds": sorted(candidates)}
                if editable:
                    source_status, source_reason, source_candidates = _classify_cohort_card(
                        key, printing_id, editable[0], editable[1], reference_printings)
                    row.update({"editableSourceStatus": source_status,
                                "editableSourceReason": source_reason,
                                "editableSourceCandidateIds": sorted(source_candidates)})
                rows.append(row)
        if sum(row["copies"] for row in rows) != 60:
            raise ValueError(f"cohort deck is not 60 cards: {deck_name}")
        if captured is not None and captured_index != len(captured["rows"]):
            raise ValueError(f"captured deck has unexpected extra rows: {deck_name}")
        deck = {"name": deck_name, "sourceKind": source_kind, "sourceUrl": listing["url"],
                "statusCopies": {status: sum(row["copies"] for row in rows if row["status"] == status)
                                 for status in ("absent", "identity_unresolved", "present_untested")},
                "cards": rows}
        if captured is not None:
            deck["sourceHtmlSha256"] = captured["htmlSha256"]
        if editable:
            deck["editableSourceStatusCopies"] = {
                status: sum(row["copies"] for row in rows if row["editableSourceStatus"] == status)
                for status in ("absent", "identity_unresolved", "present_untested")}
        decks.append(deck)
    result = {"schema": "cabt-standard-bounded-cohort/1", "asOf": cohort["as_of"],
            "scope": ("six tournament exact-printing lists and distinct simulator examples; not Standard denominator"
                      if deck_pages is not None else
                      "six tournament name-only lists and distinct exact-printing simulator examples; not Standard denominator"),
            "engine": {"version": registry_snapshot.get("engineVersion"),
                       "librarySha256": registry_snapshot.get("librarySha256"),
                       "registrySha256": registry_snapshot.get("rawSha256")},
            "standardDenominatorVerified": False, "decks": decks}
    if editable:
        result["editableSource"] = {"commit": editable_source_commit,
                                     "registryCount": len(editable[0]),
                                     "librarySha256": editable_library_sha256,
                                     "registrySha256": editable_registry_sha256}
    return result


def candidate_search_inventory(page_paths, registry_snapshot, reference_printings, *,
                               as_of, library_sha256, registry_sha256, editable_registry_snapshot=None,
                               editable_library_sha256=None, editable_registry_sha256=None,
                               editable_source_commit=None):
    if not page_paths:
        raise ValueError("candidate search requires captured pages")
    as_of = _date(as_of, "asOf")
    v2_by_id, v2_by_name = _cohort_registry(registry_snapshot, library_sha256, registry_sha256)
    editable = None
    if editable_registry_snapshot is not None:
        if not all((editable_library_sha256, editable_registry_sha256, editable_source_commit)):
            raise ValueError("editable source snapshot needs both hashes and a source commit")
        editable = _cohort_registry(editable_registry_snapshot, editable_library_sha256,
                                    editable_registry_sha256)
    pages = []
    entries = {}
    advertised_count = None
    advertised_pages = None
    pattern = re.compile(r'<span class="card-text-name"><a href="/cards/([^/]+)/([^"]+)">([^<]+)</a>')
    for path in page_paths:
        raw = Path(path).read_bytes()
        html = raw.decode("utf-8")
        current = re.search(r'<ul class="pagination" data-current="(\d+)" data-max="(\d+)"', html)
        count = re.search(r'(\d+) cards found where', html)
        if current is None or count is None:
            raise ValueError(f"captured page lacks search count or pagination: {path}")
        page_number, page_count = map(int, current.groups())
        total = int(count.group(1))
        if advertised_count is None:
            advertised_count, advertised_pages = total, page_count
        elif (advertised_count, advertised_pages) != (total, page_count):
            raise ValueError("captured pages have inconsistent search totals")
        rows = pattern.findall(html)
        if not rows:
            raise ValueError(f"captured page has no card rows: {path}")
        for set_code, number, raw_name in rows:
            printing_id = f"{set_code}-{number}"
            if printing_id in entries:
                raise ValueError(f"duplicate candidate printing {printing_id}")
            entries[printing_id] = unescape(raw_name)
        pages.append({"page": page_number, "url":
                      f"https://limitlesstcg.com/cards?q=format%3Astandard&show=all&display=full&page={page_number}",
                      "htmlSha256": sha256(raw).hexdigest(), "printingCount": len(rows)})
    if (len(pages) != advertised_pages or {page["page"] for page in pages} != set(range(1, advertised_pages + 1))
            or len(entries) != advertised_count):
        raise ValueError("candidate search capture does not contain every advertised printing")
    output_rows = []
    for printing_id, name in sorted(entries.items()):
        status, reason, candidates = _classify_cohort_card(
            _name_key(name), printing_id, v2_by_id, v2_by_name, reference_printings)
        row = {"officialPrintingIdCandidate": printing_id, "name": name,
               "shippedV2Status": status, "shippedV2Reason": reason,
               "shippedV2CandidateIds": sorted(candidates)}
        if editable:
            source_status, source_reason, source_candidates = _classify_cohort_card(
                _name_key(name), printing_id, editable[0], editable[1], reference_printings)
            row.update({"editableSourceStatus": source_status,
                        "editableSourceReason": source_reason,
                        "editableSourceCandidateIds": sorted(source_candidates)})
        output_rows.append(row)
    result = {"schema": "cabt-standard-candidate-search/1", "source": "Limitless TCG", "asOf": as_of,
              "query": "format:standard, English", "sourcePages": sorted(pages, key=lambda page: page["page"]),
              "candidatePrintingCount": len(output_rows),
              "officialDenominatorVerified": False, "coveragePercent": None,
              "shippedV2": {"librarySha256": library_sha256, "registrySha256": registry_sha256,
                            "statusCounts": dict(sorted(Counter(row["shippedV2Status"] for row in output_rows).items()))},
              "printings": output_rows}
    if editable:
        result["editableSource"] = {"commit": editable_source_commit,
                                     "librarySha256": editable_library_sha256,
                                     "registrySha256": editable_registry_sha256,
                                     "statusCounts": dict(sorted(Counter(row["editableSourceStatus"]
                                                                         for row in output_rows).items()))}
    return result
