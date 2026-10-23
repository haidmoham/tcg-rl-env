"""Exact printing candidates from public import mappings and inert card tables."""

import base64
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import struct
import unicodedata


CORPUS_SHA256 = "afe7931aab5a6d4f77cdecbb7ae58502dd17fce4695986a348c2b27ebdd6c4ec"
INDEX_SHA256 = "6ceef4c2f6e0fe1cfd47947e8829936528633b67a2ea910a7d7592120fe643ad"
CORPUS_ROWS = 22945
TABLE_FILE = re.compile(r"^card-database-(.+?)_[0-9]+_en_0\.0\.json$", re.I)
IDENTITY_COLUMNS = (
    "cardID", "EN Card Name", "EN Card #", "setCode", "HP", "Retreat", "EN Type",
    "category", "archetypeID", "variant", "Variant Suffix", "EN Evolves From",
    "EN Attack Name", "EN Attack Name 2", "EN Attack Name 3", "EN Attack Name 4",
    "EN Weakness Type", "EN Resistance Type", "Regulations symbol",
)
LIVE_TYPES = {"C": 1, "R": 2, "G": 3, "W": 4, "F": 5, "Y": 6,
              "D": 7, "L": 8, "N": 9, "M": 10, "P": 11}
STAGE_COLUMN = "Stage (Pokémon)/Type (Energy and Trainer)"


def digest(raw):
    return sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def program_hash(card):
    return digest(canonical({key: value for key, value in card.items()
                             if key not in {"cardSourceID", "releaseDate"}}).encode())


def revision_conflicts(bindings):
    return [column for column in IDENTITY_COLUMNS
            if len({canonical(binding["fields"][column]) for binding in bindings
                    if binding["fields"].get(column) is not None}) > 1]


def complete_binding(binding, pokemon):
    fields = binding["fields"]
    if any(not isinstance(fields.get(column), str) or not fields[column].strip()
           for column in ("cardID", "EN Card Name", "EN Card #", "setCode")):
        return False
    return not pokemon or (type(fields.get("HP")) is int and fields["HP"] > 0
                           and type(fields.get("Retreat")) is int and fields["Retreat"] >= 0
                           and isinstance(fields.get("EN Type"), str) and bool(fields["EN Type"]))


def text_key(value):
    return unicodedata.normalize("NFKC", value).translate(str.maketrans({
        "’": "'", "‘": "'", "–": "-", "−": "-", "é": "e"})).strip().casefold()


class TableReader:
    """Read the DataTableCustomFormatter layout without loading client code."""

    def __init__(self, data):
        self.data = data
        self.position = 0

    def take(self, count):
        if count < 0 or self.position + count > len(self.data):
            raise ValueError("truncated card database table")
        result = self.data[self.position:self.position + count]
        self.position += count
        return result

    def number(self, code):
        return struct.unpack("<" + code, self.take(struct.calcsize("<" + code)))[0]

    def string(self):
        length = 0
        for shift in (0, 7, 14, 21, 28):
            part = self.number("B")
            length |= (part & 127) << shift
            if not part & 128:
                if length > 16 * 1024 * 1024:
                    raise ValueError("oversized table string")
                return self.take(length).decode("utf-8")
        raise ValueError("invalid .NET string length")

    def value(self, kind):
        marker = self.number("B")
        if marker in (1, 2):
            return None
        if marker != 0:
            raise ValueError("unknown table value marker")
        if kind == "System.String":
            return self.string()
        if kind == "System.Byte[]":
            return self.take(self.number("i"))
        codes = {"System.Boolean": "?", "System.Byte": "B", "System.SByte": "b",
                 "System.Int16": "h", "System.Int32": "i", "System.Int64": "q",
                 "System.UInt16": "H", "System.UInt32": "I", "System.UInt64": "Q",
                 "System.Single": "f", "System.Double": "d", "System.DateTime": "q",
                 "System.TimeSpan": "q", "CardDatabase.DataAccess.CardCategory": "B"}
        if kind in codes:
            return self.number(codes[kind])
        if kind in ("System.Decimal", "System.Guid"):
            return self.take(16)
        if kind == "System.Char":
            first = self.take(1)[0]
            width = 1 if first < 128 else 2 if first < 224 else 3 if first < 240 else 4
            return (bytes([first]) + self.take(width - 1)).decode("utf-8")
        raise ValueError(f"unsupported table column type: {kind}")


def read_table(binary):
    reader = TableReader(binary)
    name = reader.value("System.String")
    count = reader.number("i")
    if not 1 <= count <= 512:
        raise ValueError("implausible table column count")
    schema = [(reader.string(), reader.string()) for _ in range(count)]
    if len({column for column, _ in schema}) != count:
        raise ValueError("duplicate table column")
    row_count = reader.number("i")
    if not 0 <= row_count <= 100000:
        raise ValueError("implausible table row count")
    rows = [{column: reader.value(kind) for column, kind in schema} for _ in range(row_count)]
    if reader.position != len(binary):
        raise ValueError("unread bytes after card table")
    return name, rows


def load_public_metadata(cache, catalog_receipt, join_receipt):
    cache = Path(cache).resolve()
    if cache.name != "config-cache":
        raise ValueError("public metadata root must be the explicit config-cache directory")
    catalog_raw = Path(catalog_receipt).read_bytes()
    join_raw = Path(join_receipt).read_bytes()
    catalog = json.loads(catalog_raw)
    expected = {entry["file"]: entry["sha256"] for entry in catalog["sources"]}
    mapping_path = cache / "deckImportData_0.0.json"
    if mapping_path.resolve().parent != cache:
        raise ValueError("public import mapping path escapes config-cache")
    mapping_raw = mapping_path.read_bytes()
    mapping_document = json.loads(mapping_raw)
    mapping = json.loads(mapping_document["keys"]["deckImportData"]["contentString"])["setIDMappings"]
    mapping = {key.upper(): value.lower() for key, value in mapping.items()}
    mapping_source = {"file": "deckImportData_0.0.json", "sha256": digest(mapping_raw),
                      "matchesCapturedReceipt": digest(mapping_raw) == json.loads(join_raw)["mappingSha256"]}
    bindings, sources = [], []
    for path in sorted(cache.glob("card-database-*.json")):
        match = TABLE_FILE.fullmatch(path.name)
        if not match:
            continue
        if path.resolve().parent != cache:
            raise ValueError("public table path escapes config-cache")
        raw = path.read_bytes()
        document = json.loads(raw)
        table = document["keys"]["table"]
        if table.get("isBinary") is not True:
            raise ValueError("public card table is not binary")
        table_name, rows = read_table(base64.b64decode(table["contentBinary"], validate=True))
        source = {"file": path.name, "sha256": digest(raw), "revision": document.get("revision"),
                  "tableName": table_name, "rows": len(rows),
                  "matchesCapturedReceipt": expected.get(path.name) == digest(raw)}
        sources.append(source)
        for row_index, row in enumerate(rows):
            fields = {column: row[column] for column in IDENTITY_COLUMNS if column in row}
            source_id = fields.get("cardID")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("missing full cardID in English table")
            bindings.append({"fields": fields, "rowIndex": row_index,
                             "source": source, "rowSha256": digest(canonical(fields).encode())})
    observed = {source["file"] for source in sources}
    missing = sorted(filename for filename in expected if TABLE_FILE.fullmatch(filename)
                     and filename not in observed)
    return {"mapping": mapping, "mappingSource": mapping_source, "bindings": bindings,
            "missingCapturedTables": missing,
            "missingCapturedPrefixes": sorted({TABLE_FILE.fullmatch(filename).group(1).lower()
                                                for filename in missing}),
            "sources": sources, "receiptSha256": {"catalog": digest(catalog_raw), "selectedJoins": digest(join_raw)}}


def load_corpus(corpus_path, index_path, *, corpus_sha256=CORPUS_SHA256,
                index_sha256=INDEX_SHA256, expected_rows=CORPUS_ROWS):
    raw, index_raw = Path(corpus_path).read_bytes(), Path(index_path).read_bytes()
    if digest(raw) != corpus_sha256 or digest(index_raw) != index_sha256:
        raise ValueError("frozen corpus or index hash mismatch")
    index = json.loads(index_raw)
    if index.get("schemaVersion") != 2 or index.get("corpusSha256") != corpus_sha256:
        raise ValueError("index does not bind the frozen corpus")
    indexed = index["cards"]
    by_id = {entry["cardSourceID"]: entry for entry in indexed}
    if len(by_id) != expected_rows or len(indexed) != expected_rows:
        raise ValueError("index has duplicate IDs or a different denominator")
    cards, offset = {}, 0
    for line in raw.splitlines(keepends=True):
        card = json.loads(line)
        source_id = card["cardSourceID"]
        entry = by_id.get(source_id)
        row_hash = digest(line)
        if (source_id in cards or entry is None or entry.get("cardSha256") != row_hash
                or entry.get("jsonlOffset") != offset or entry.get("jsonlBytes") != len(line)):
            raise ValueError("index does not bind the exact JSONL row")
        cards[source_id] = (card, row_hash)
        offset += len(line)
    if len(cards) != expected_rows:
        raise ValueError("corpus has a different denominator")
    return cards


def read_reference(raw):
    grouped = defaultdict(list)
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        card_id = int(row["Card ID"])
        if card_id <= 0 or str(card_id) != row["Card ID"]:
            raise ValueError("invalid CABT reference card ID")
        grouped[card_id].append(row)
    return dict(grouped)


def _checks(reference, table, card, native):
    checks = []

    def compare(field, values):
        checks.append({"field": field, "values": values,
                       "matches": len({canonical(value) for value in values.values()}) == 1})

    compare("name", {"reference": text_key(reference[0]["Card Name"]),
                     "table": text_key(table["EN Card Name"]),
                     "corpus": text_key(card["cardName"]), "native": text_key(native["name"])})
    if reference[0]["HP"].isdigit():
        for column, field in (("HP", "hp"), ("Retreat", "retreatCost")):
            values = {"table": table.get(column), "corpus": card.get(field, 0),
                      "native": native.get(field)}
            if reference[0][column].isdigit():
                values["reference"] = int(reference[0][column])
            compare(field, values)
    pokemon = "Pokémon" in reference[0][STAGE_COLUMN]
    if pokemon:
        symbols = (["N"] if reference[0]["Type"] == "竜"
                   else re.findall(r"\{([A-Z])\}", reference[0]["Type"]))
        compare("typing", {"reference": symbols, "table": list(table.get("EN Type", ""))})
        if symbols and all(symbol in LIVE_TYPES for symbol in symbols):
            compare("corpusTyping", {"reference": sorted(LIVE_TYPES[symbol] for symbol in symbols),
                                     "corpus": sorted(card.get("cardTyping", []))})
        stages = {"Basic Pokémon": (1, "basic"), "Stage 1 Pokémon": (2, "stage1"),
                  "Stage 2 Pokémon": (3, "stage2")}
        stage = stages.get(reference[0][STAGE_COLUMN])
        if stage:
            compare("stage", {"reference": stage[0], "corpus": card.get("cardStage", 1)})
            compare("nativeStage", {"reference": True, "native": native.get(stage[1])})
        reference_moves = {text_key(row["Move Name"]).removeprefix("[ability] ")
                           for row in reference if text_key(row["Move Name"]) not in ("", "n/a", "[tera]", "tera")}
        table_moves = {text_key(table.get(column) or "").removeprefix("[ability] ")
                       for column in IDENTITY_COLUMNS if column.startswith("EN Attack Name")}
        table_moves.discard("")
        corpus_moves = {text_key(action.get("actionName") or action.get("EffectName") or "").removeprefix("[ability] ")
                        for channel in ("useActions", "passiveActions", "triggeredActions")
                        for action in card.get(channel, [])}
        corpus_moves.discard("")
        corpus_moves.discard("tera")
        corpus_moves.discard("[tera]")
        compare("actionNames", {"reference": sorted(reference_moves), "table": sorted(table_moves),
                                "corpus": sorted(corpus_moves)})
        compare("attackCount", {"corpus": sum(action.get("actionType") == 1
                                              for action in card.get("useActions", [])),
                                "native": len(native.get("attacks", []))})
    return checks


def map_identities(cards, reference, snapshot, metadata):
    if snapshot.get("schema") != "cabt-registry-snapshot/1":
        raise ValueError("expected a CABT registry snapshot")
    entries = snapshot["entries"]
    raw_hash = digest(json.dumps(entries, separators=(",", ":"), ensure_ascii=False).encode())
    if raw_hash != snapshot["rawSha256"]:
        raise ValueError("CABT registry hash mismatch")
    native = {entry["cardId"]: entry for entry in entries}
    if len(native) != len(entries):
        raise ValueError("duplicate CABT native card ID")
    tables = defaultdict(list)
    for binding in metadata["bindings"]:
        tables[binding["fields"]["cardID"]].append(binding)
    aliases, unresolved = [], []
    requested = defaultdict(list)
    for card_id, rows in sorted(reference.items()):
        tuples = {(row["Expansion"], row["Collection No."]) for row in rows}
        if len(tuples) == 1:
            expansion, number = next(iter(tuples))
            prefix = metadata["mapping"].get(expansion.upper())
            if prefix:
                requested[prefix + "_" + number].append(card_id)
    for card_id in sorted(native):
        reasons, checks = [], []
        rows = reference.get(card_id, [])
        source_id = None
        bindings = []
        if not rows:
            reasons.append("native_id_absent_from_reference")
        else:
            stable_columns = set(rows[0]) - {"Move Name", "Cost", "Damage", "Effect Explanation"}
            if any(any(row[column] != rows[0][column] for column in stable_columns) for row in rows):
                reasons.append("conflicting_reference_printing_metadata")
            expansion, number = rows[0]["Expansion"], rows[0]["Collection No."]
            prefix = metadata["mapping"].get(expansion.upper())
            if not prefix:
                reasons.append("expansion_absent_from_import_mapping")
            else:
                source_id = prefix + "_" + number
                if prefix in metadata.get("missingCapturedPrefixes", []):
                    reasons.append("captured_table_revision_missing")
                bindings = tables.get(source_id, [])
                if len(requested[source_id]) != 1:
                    reasons.append("ambiguous_reference_printing")
                if not metadata["mappingSource"]["matchesCapturedReceipt"]:
                    reasons.append("import_mapping_differs_from_capture")
                if not bindings:
                    reasons.append("exact_full_id_absent_from_english_table")
                elif any(not binding["source"]["matchesCapturedReceipt"] for binding in bindings):
                    reasons.append("table_differs_from_capture")
                if revision_conflicts(bindings):
                    reasons.append("conflicting_table_revisions")
                pokemon = "Pokémon" in rows[0][STAGE_COLUMN]
                complete = all(complete_binding(binding, pokemon) for binding in bindings)
                if not complete:
                    reasons.append("incomplete_table_revision")
                if source_id not in cards:
                    reasons.append("exact_full_id_absent_from_frozen_corpus")
                if bindings and complete:
                    fields = bindings[0]["fields"]
                    if fields.get("setCode", "").lower() != prefix or fields.get("EN Card #") != number:
                        reasons.append("table_printing_tuple_mismatch")
                    if source_id in cards and "EN Card Name" in fields:
                        checks = [check for binding in bindings
                                  for check in _checks(rows, binding["fields"], cards[source_id][0], native[card_id])]
                        reasons.extend("metadata_conflict/" + check["field"]
                                       for check in checks if not check["matches"])
        evidence = {"cabtCardId": card_id, "cardSourceID": source_id, "checks": checks,
                    "tableBindings": bindings, "reasons": sorted(set(reasons))}
        if reasons:
            unresolved.append(evidence)
        else:
            aliases.append({**evidence, "cardSha256": cards[source_id][1],
                            "method": "exact_import_set_collector_full_id",
                            "publicImportMapping": metadata["mappingSource"]})
    base_aliases = list(aliases)
    bases_by_program = defaultdict(list)
    for alias in base_aliases:
        bases_by_program[program_hash(cards[alias["cardSourceID"]][0])].append(alias)
    mapped_ids = {alias["cardSourceID"] for alias in aliases}
    unresolved_rows = []
    for source_id, (card, row_hash) in sorted(cards.items()):
        if source_id in mapped_ids:
            continue
        identical_bases = bases_by_program.get(program_hash(card), [])
        bindings = tables.get(source_id, [])
        reasons = []
        if not identical_bases:
            reasons.append("no_exact_native_base_with_identical_full_program")
        if not bindings:
            reasons.append("exact_full_id_absent_from_english_table")
        if source_id.split("_", 1)[0] in metadata.get("missingCapturedPrefixes", []):
            reasons.append("captured_table_revision_missing")
        if any(not binding["source"]["matchesCapturedReceipt"] for binding in bindings):
            reasons.append("table_differs_from_capture")
        if revision_conflicts(bindings):
            reasons.append("conflicting_table_revisions")
        candidates = []
        if identical_bases and bindings and not reasons:
            for base in identical_bases:
                base_fields = base["tableBindings"][0]["fields"]
                pokemon = "Pokémon" in reference[base["cabtCardId"]][0][STAGE_COLUMN]
                if not all(complete_binding(binding, pokemon) for binding in bindings):
                    reasons.append("incomplete_table_revision")
                    continue
                tuple_matches = source_id.startswith(base_fields["setCode"].lower() + "_") and all(
                                    binding["fields"].get("setCode", "").lower() == base_fields["setCode"].lower()
                                    and binding["fields"].get("EN Card #") == base_fields["EN Card #"]
                                    for binding in bindings)
                if not tuple_matches:
                    continue
                checks = [check for binding in bindings
                          for check in _checks(reference[base["cabtCardId"]], binding["fields"], card,
                                               native[base["cabtCardId"]])]
                if all(check["matches"] for check in checks):
                    candidates.append((base, checks))
            if len(candidates) != 1:
                reasons.append("variant_printing_tuple_or_native_base_unresolved")
        if candidates and not reasons:
            base, checks = candidates[0]
            aliases.append({"cabtCardId": base["cabtCardId"], "cardSourceID": source_id,
                            "cardSha256": row_hash, "checks": checks, "tableBindings": bindings,
                            "method": "identical-source-program", "programSha256": program_hash(card),
                            "baseCardSourceID": base["cardSourceID"], "baseCardSha256": base["cardSha256"],
                            "publicImportMapping": metadata["mappingSource"], "reasons": []})
        else:
            unresolved_rows.append({"cardSourceID": source_id, "cardSha256": row_hash,
                                    "reasons": sorted(set(reasons))})
    return {"schema": "live-printing-identities/1", "aliases": aliases, "unresolved": unresolved,
            "unresolvedFrozenRows": unresolved_rows,
            "engineVersion": snapshot["engineVersion"], "registrySha256": snapshot["rawSha256"],
            "librarySha256": snapshot["librarySha256"], "fullParityQualified": False,
            "counts": {"nativeCards": len(native), "identityCandidates": len(aliases),
                       "exactBaseCandidates": len(base_aliases),
                       "identicalProgramVariantCandidates": len(aliases) - len(base_aliases),
                       "unresolved": len(unresolved), "frozenRows": len(cards),
                       "candidateFrozenRows": len({alias["cardSourceID"] for alias in aliases})},
            "unresolvedReasonCounts": dict(sorted(Counter(reason for row in unresolved
                                                            for reason in row["reasons"]).items()))}


def private_output(root, output):
    output = Path(output).resolve()
    runs = (Path(root).resolve() / "runs").resolve()
    if output == runs or not output.is_relative_to(runs):
        raise ValueError("output must remain under this checkout's resolved runs/ directory")
    if output.exists():
        raise ValueError("output exists; preserve the frozen report and choose a new path")
    return output
