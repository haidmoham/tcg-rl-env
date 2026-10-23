"""Canonical printed-card identities shared by Live and the frozen Mew table."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import re
import unicodedata


TYPOGRAPHY = str.maketrans({'’': "'", '‘': "'", 'ʼ': "'", '−': '-', '–': '-', '—': '-', '\u00a0': ' '})
LIVE_TYPES = {1: 'C', 2: 'R', 3: 'G', 4: 'W', 5: 'F', 6: 'Y', 7: 'D', 8: 'L', 9: 'N', 10: 'M', 11: 'P'}
MEW_TYPES = {0: 'C', 1: 'G', 2: 'R', 3: 'W', 4: 'L', 5: 'P', 6: 'F', 7: 'D', 8: 'M', 9: 'N', 14: 'Y'}
LIVE_KINDS = {1: 'basic_energy', 2: 'special_energy', 3: 'pokemon', 4: 'item', 5: 'supporter', 6: 'tool', 7: 'stadium', 8: 'fossil'}
MEW_KINDS = {0: 'pokemon', 1: 'item', 2: 'tool', 3: 'supporter', 4: 'stadium', 5: 'basic_energy', 6: 'special_energy'}


def normalize_text(value):
    return ' '.join(unicodedata.normalize('NFKC', value).translate(TYPOGRAPHY).split()).casefold()


def normalize_card_name(value):
    energy_names = {'C': 'Colorless', 'G': 'Grass', 'R': 'Fire', 'W': 'Water',
                    'L': 'Lightning', 'P': 'Psychic', 'F': 'Fighting', 'D': 'Darkness',
                    'M': 'Metal', 'N': 'Dragon', 'Y': 'Fairy'}
    return normalize_text(re.sub(r'\{([CGRWLPFDMNY])\}',
                                 lambda match: energy_names[match[1]], value))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def _energy_cost(value):
    tokens = re.findall(r'\{([CGRWLPFDMNY])\}', value)
    if ''.join('{' + token + '}' for token in tokens) != value:
        raise ValueError('unsupported printed Energy cost')
    return tuple(sorted(tokens))


@dataclass(frozen=True)
class CanonicalCard:
    name: str
    kind: str
    hp: int
    retreat: int
    stage: int
    types: tuple[str, ...]
    attacks: tuple[tuple[str, tuple[str, ...]], ...]
    abilities: tuple[str, ...]

    def key(self):
        return digest(asdict(self))


def live_identity(card):
    kind = LIVE_KINDS[card.get('cardFormat', 3)]
    attacks, abilities = [], set()
    if kind == 'pokemon':
        for channel in ('useActions', 'passiveActions', 'triggeredActions'):
            for action in card.get(channel, []):
                name = normalize_text(action.get('actionName', action.get('EffectName', '')))
                if channel == 'useActions' and action.get('actionType') == 1:
                    attacks.append((name, _energy_cost(action.get('energyCost', ''))))
                elif name and name not in ('<empty>', 'tera', '[tera]'):
                    abilities.add(name.removeprefix('[ability] '))
    return CanonicalCard(
        normalize_card_name(card['cardName']), kind,
        card.get('hp', 0) if kind in ('pokemon', 'fossil') else 0,
        card.get('retreatCost', 0) if kind == 'pokemon' else 0,
        card.get('cardStage', 1) if kind == 'pokemon' else 0,
        tuple(sorted(LIVE_TYPES[value] for value in card.get('cardTyping', []))) if kind == 'pokemon' else (),
        tuple(sorted(attacks)), tuple(sorted(abilities)))


def mew_identity(card, attacks):
    kind = MEW_KINDS[card['cardType']]
    if card['cardType'] == 1 and card.get('pokemonType'):
        kind = 'fossil'
    stage = card.get('evolutionType', 0) if kind == 'pokemon' else 0
    if stage not in (0, 1, 2, 3):
        raise ValueError('unsupported printed evolution stage')
    moves = []
    if kind == 'pokemon':
        for identity in card['attacks']:
            move = attacks[identity]
            moves.append((normalize_text(move['name']), tuple(sorted(MEW_TYPES[value] for value in move['energies']))))
    return CanonicalCard(
        normalize_card_name(card['name']), kind,
        card['hp'] if kind in ('pokemon', 'fossil') else 0,
        card['retreatCost'] if kind == 'pokemon' else 0,
        stage,
        (MEW_TYPES[card['energyType']],) if kind == 'pokemon' else (),
        tuple(sorted(moves)),
        tuple(sorted({normalize_text(skill['name']).removeprefix('[ability] ') for skill in card['skills']
                      if kind == 'pokemon' and not skill.get('ancientTrait')
                      and normalize_text(skill['name']) not in ('', '<empty>', 'tera', '[tera]')})))


def native_signature(card, attacks):
    def normalized(value):
        if isinstance(value, str):
            return normalize_text(value)
        if isinstance(value, list):
            return [normalized(item) for item in value]
        if isinstance(value, dict):
            return {key: normalized(item) for key, item in value.items()}
        return value
    static = {key: value for key, value in card.items() if key != 'cardId'}
    boilerplate = {
        1: 'you may play as many item cards as you like during your turn (before your attack).',
        3: 'you may play only 1 supporter card during your turn (before your attack).',
    }.get(card['cardType'])
    static['skills'] = []
    for skill in card['skills']:
        text = normalize_text(skill['text'])
        if boilerplate:
            text = text.removesuffix(boilerplate).strip()
        text = re.sub(r'\b(?:then,?\s+)?shuffle your deck(?: afterward)?\.', 'shuffle your deck.', text)
        static['skills'].append({**skill, 'text': text})
    static['attacks'] = [{key: value for key, value in attacks[identity].items() if key != 'attackId'}
                         for identity in card['attacks']]
    return digest(normalized(static))


def live_selection_limits(card):
    limits = set()

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if value.get('$type', '').split(',')[0] == 'MatchLogic.MoveCards':
                settings = value.get('from', {}).get('entitySelectionSettings', {})
                bounds = []
                for key in ('minSelectionAmount', 'maxSelectionAmount'):
                    total = settings.get(key, {})
                    number = total.get('value1', {})
                    if (total.get('$type', '').split(',')[0] != 'MatchLogic.SingleVariableTotal'
                            or number.get('$type', '').split(',')[0] != 'MatchLogic.ExplicitNumberVariable'):
                        break
                    bounds.append(number['explicitValue'])
                if len(bounds) == 2:
                    limits.add(tuple(bounds))
            for item in value.values():
                visit(item)

    visit(card.get('useActions', []))
    return limits


def native_selection_limits(card):
    limits = set()
    for skill in card['skills']:
        for match in re.finditer(r'\b(?:shuffle|put|return|choose|discard|attach)\s+(up to\s+)?(\d+)\b', normalize_text(skill['text'])):
            maximum = int(match[2])
            limits.add((1 if match[1] else maximum, maximum))
    return limits


def build_mapping(corpus, metadata):
    attacks = {attack['attackId']: attack for attack in metadata['attacks']}
    index = defaultdict(list)
    report = {'schema': '86-tcg-agent/live-card-normalization/1', 'matched': 0,
              'unmatched': [], 'conflicts': [], 'unsupported_native_ids': [], 'equivalent_native_duplicates': {}}
    for card in metadata['cards']:
        try:
            identity = mew_identity(card, attacks)
        except (KeyError, ValueError):
            report['unsupported_native_ids'].append(card['cardId'])
            continue
        index[identity].append(card)
    mapping = {}
    for card in corpus:
        source = card['cardSourceID']
        try:
            identity = live_identity(card)
        except (KeyError, ValueError):
            report['unmatched'].append({'source': source, 'reason': 'unsupported_printed_metadata'})
            continue
        candidates = index.get(identity, [])
        if not candidates:
            report['unmatched'].append({'source': source, 'reason': 'no_normalized_printed_identity', 'identity': asdict(identity)})
            continue
        signatures = {native_signature(candidate, attacks) for candidate in candidates}
        if len(signatures) > 1:
            limits = live_selection_limits(card)
            if limits:
                candidates = [candidate for candidate in candidates if native_selection_limits(candidate) == limits]
                signatures = {native_signature(candidate, attacks) for candidate in candidates}
        if len(signatures) != 1:
            report['conflicts'].append({'source': source, 'native_ids': [c['cardId'] for c in candidates],
                                        'reason': 'different_native_printed_descriptors', 'identity': asdict(identity)})
            continue
        canonical = min(candidates, key=lambda candidate: candidate['cardId'])
        attack_ids = {}
        for attack_id in canonical['attacks']:
            key = normalize_text(attacks[attack_id]['name'])
            if key in attack_ids:
                break
            attack_ids[key] = attack_id
        else:
            mapping[source] = {'id': canonical['cardId'], 'name': card['cardName'], 'attacks': attack_ids,
                               'normalization_digest': identity.key()}
            report['matched'] += 1
            if len(candidates) > 1:
                report['equivalent_native_duplicates'][identity.key()] = sorted(c['cardId'] for c in candidates)
            continue
        report['conflicts'].append({'source': source, 'reason': 'duplicate_normalized_attack_name',
                                   'native_ids': [canonical['cardId']]})
    report['source_count'] = report['matched'] + len(report['unmatched']) + len(report['conflicts'])
    return mapping, report
