"""Private, static provenance for candidate CABT mechanics fixtures.

This module never executes a card program or qualifies an effect port.
"""

from collections import defaultdict
import hashlib
import json


SCHEMA = "local-mechanics-index/1"
ACTION_CHANNELS = ("useActions", "passiveActions", "triggeredActions")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def program_sha256(card):
    encoded = json.dumps(card, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(encoded.encode("utf-8"))


def _family(type_name):
    if not isinstance(type_name, str):
        return None
    name = type_name.split(",", 1)[0].rsplit(".", 1)[-1]
    if name.isidentifier():
        return name
    return None


def _operation_families(subactions):
    if not isinstance(subactions, list):
        return
    for subaction in subactions:
        if not isinstance(subaction, dict):
            continue
        family = _family(subaction.get("$type"))
        if family:
            yield family
        yield from _operation_families(subaction.get("subActionsToPerformOnEachEntity", []))
        yield from _operation_families(subaction.get("subActions", []))
        added = subaction.get("addedAction")
        if isinstance(added, dict):
            yield from _operation_families(added.get("subActions", []))


def _mechanics(card):
    channels = {}
    for channel in ACTION_CHANNELS:
        actions = card.get(channel, [])
        operations = set()
        conditions = set()
        for action in actions:
            operations.update(_operation_families(action.get("subActions", [])))
            passive = action.get("passiveEffect")
            if isinstance(passive, dict):
                family = _family(passive.get("$type"))
                if family:
                    operations.add(family)
            for condition in action.get("conditions", []):
                if isinstance(condition, dict):
                    family = _family(condition.get("$type"))
                    if family:
                        conditions.add(family)
        channels[channel] = {
            "actionCount": len(actions),
            "operationFamilies": sorted(operations),
            "conditionFamilies": sorted(conditions),
        }
    return channels


def build_index(cards, joins, mapping, *, card_data_sha256, joins_sha256,
                game_data_sha256=None, mapping_sha256=None):
    """Build an identity and static-operation index from validated private input."""
    cards_by_source = {}
    for card in cards:
        source_id = card["cardSourceID"]
        if source_id in cards_by_source:
            raise ValueError("duplicate cardSourceID in cardData")
        cards_by_source[source_id] = card
    if set(mapping) - set(cards_by_source):
        raise ValueError("mapping receipt refers to absent cardSourceID")

    joins_by_printing = defaultdict(list)
    joins_by_source = defaultdict(list)
    for join in joins:
        printing = join["printingId"]
        source_id = join.get("cardSourceID")
        joins_by_printing[printing].append(join)
        if source_id is not None:
            joins_by_source[source_id].append(join)

    printings = {}
    for printing, candidates in sorted(joins_by_printing.items()):
        source_ids = sorted({item["cardSourceID"] for item in candidates
                             if item.get("cardSourceID") is not None})
        statuses = sorted({item.get("status") if item.get("status") in
                           ("joined", "exact-native-source", "unresolved") else "other"
                           for item in candidates})
        resolved = (len(source_ids) == 1 and source_ids[0] in cards_by_source
                    and all(item.get("cardSourceID") is not None for item in candidates)
                    and all(status in ("joined", "exact-native-source") for status in statuses))
        printings[printing] = {
            "cardSourceIDs": source_ids,
            "joinStatuses": statuses,
            "resolution": "unique-source" if resolved else "needs-identity-review",
        }

    records = {}
    by_logic = defaultdict(list)
    by_hash = defaultdict(list)
    for source_id, card in sorted(cards_by_source.items()):
        digest = program_sha256(card)
        logic_source = card["logicSourceID"]
        by_logic[logic_source].append(source_id)
        by_hash[digest].append(source_id)
        mapped = mapping.get(source_id)
        method = mapped["method"] if mapped and mapped["method"] in (
            "exact-printing", "unique-normalized-name-reprint") else None
        needs = ["native-transition-fixture", "cabt-transition-fixture"]
        if method == "unique-normalized-name-reprint":
            needs.append("effect-equivalence-review")
        elif method != "exact-printing":
            needs.append("cabt-identity-review")
        if any(printings[item["printingId"]]["resolution"] != "unique-source"
               for item in joins_by_source[source_id]):
            needs.append("printing-identity-review")
        records[source_id] = {
            "logicSourceID": logic_source,
            "programSha256": digest,
            "printingIds": sorted({item["printingId"] for item in joins_by_source[source_id]}),
            "joinSha256": {
                key: sorted({item[key] for item in joins_by_source[source_id]
                             if isinstance(item.get(key), str)})
                for key in ("cardSha256", "programSha256")
            },
            "mechanics": _mechanics(card),
            "cabtCandidate": {"cardId": mapped["cabtId"], "mappingMethod": method}
                if mapped else None,
            "fixtureNeeds": needs,
            "qualification": "static-candidate-only",
        }

    return {
        "schema": SCHEMA,
        "inputSha256": {
            "cardData": card_data_sha256,
            "gameData": game_data_sha256,
            "printingJoins": joins_sha256,
            "mappingReceipt": mapping_sha256,
        },
        "cardsBySourceID": records,
        "printings": printings,
        "sourceIDsByLogicSourceID": {key: sorted(value) for key, value in sorted(by_logic.items())},
        "sourceIDsByProgramSha256": {key: sorted(value) for key, value in sorted(by_hash.items())},
    }
