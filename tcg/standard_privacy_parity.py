"""Finite two-view observations from captured source and current native fixtures."""

import ctypes
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile

from tcg.live_parity import LIVE_PINS, digest, read_pinned
from tcg.standard_effect_parity import (_ready as effect_ready,
    standard_effect_expected, standard_effect_provenance, normalize_native, source_record)
from tcg.standard_missing_pool_parity import (_domain as missing_domain,
    standard_missing_pool_expected, standard_missing_pool_provenance)

PREPARED_SHA256 = '555dbc1e5cf9cb1f1d049cda8df55684cca8ee7bf41535802d72c552164c0aaf'
PROJECTION_SHA256 = 'a4ff365a580fda1cc8b189d5ad891be4921a89658ff995e5a1478efa0a9de51d'
REGISTRY_SHA256 = 'a81584982eb02325cf5d558a1c3fd091c74a980702bc2be6d424708725fbe5e0'
LIBRARY_SHA256 = 'eb0ba3176dfcc572da029d5765529f1d032a0d764a3e62aa5e937a20243673a0'
SUITE = 'standard-information-boundaries/1'
LIMITATIONS = [
    'Only the declared finite before/after two-view privacy observations are admitted.',
    'Visible identities, private-zone counts and hidden hand, deck and prize identities are compared.',
    'Every mode retains its exact complete physical effect and actor/action identity witness.',
    'Pending visibility, future states, alternative selections, rules, legality and interactions remain unqualified.',
]

EXTRA_IDENTITIES = {'sv6_128': 119, 'sv6_129': 120, 'sv6_130': 121}
SOURCE_ENERGY = {f'bw1_{105 + index}': index + 1 for index in range(8)}


def source_identity(source_id, item):
    if source_id == item['sourceID']:
        return item['nativeCardId']
    fixed = {'whole-export-control': 9990600, 'me5_78': 1428, **SOURCE_ENERGY, **EXTRA_IDENTITIES}
    if source_id not in fixed:
        raise ValueError('Privacy projection encountered an unbound actual source card. ' + source_id)
    return fixed[source_id]


def hidden_source_card(card):
    if (card['cardSourceID'] != '' or card['entityID'] != 'PRIVATE' or card['cacheVersion'] is not None
            or card['currentParentEntityID'] not in ('', None) or card['previousParentEntityID'] not in ('', None)
            or card['ownerPlayerId'] != ''
            or any(card[name] for name in ('attachedPokemon', 'attachedEnergy', 'attachedTools', 'attachedStadiums',
                'cardMovementHistory', 'entityMetaData', 'appliedStatusEffects', 'battleFlagCounts',
                'blockedAbilities', 'blockedEffectIDs', 'lastStateOfAppliedEffects'))):
        raise ValueError('Actual source privatization leaks a hidden identity, attachment or card-specific state')


def source_card(card, item):
    if not isinstance(card, dict) or card['entityID'] in ('', 'PRIVATE') or not card['cardSourceID']:
        raise ValueError('Actual public source card identity is missing')
    return {'id': source_identity(card['cardSourceID'], item),
        'energyCards': [source_card(child, item)['id'] for child in card['attachedEnergy']],
        'tools': [source_card(child, item)['id'] for child in card['attachedTools']],
        'preEvolution': [source_card(child, item)['id'] for child in card['attachedPokemon']]}


def source_view(snapshot, viewer, item):
    view = snapshot['views']['owner' if viewer == 0 else 'opponent']
    if any(view[name] for name in ('p1VisiblePending', 'p2VisiblePending', 'p1HiddenPending', 'p2HiddenPending')):
        raise ValueError('Pending visibility requires a separate measured projection')
    players = []
    for player in range(2):
        prefix = 'p' + str(player + 1)
        hand, deck, prizes = [view[prefix + area] for area in ('Hand', 'Deck', 'Prize')]
        if not all(isinstance(zone, list) for zone in (hand, deck, prizes)):
            raise ValueError('Actual source private-zone counts are absent')
        for card in [*deck, *prizes, *(hand if viewer != player else [])]:
            hidden_source_card(card)
        active = view[prefix + 'Active']
        players.append({'handCount': len(hand),
            'hand': [source_card(card, item)['id'] for card in hand] if player == viewer else None,
            'deckCount': len(deck), 'deck': None, 'prizeCount': len(prizes), 'prize': None,
            'active': [source_card(active, item)] if active else [],
            'bench': [source_card(card, item) for card in view[prefix + 'Bench']],
            'discard': [source_card(card, item)['id'] for card in view[prefix + 'Discard']],
            'lostZone': [source_card(card, item)['id'] for card in view[prefix + 'LostZone']]})
    stadium = view['stadium']
    return {'viewer': viewer, 'players': players, 'looking': None,
            'stadium': [source_card(stadium, item)['id']] if stadium else []}


def native_card(card):
    if not isinstance(card, dict) or type(card.get('id')) is not int or card['id'] <= 0:
        raise ValueError('Actual native public card identity is missing')
    return {'id': card['id'], 'energyCards': [native_card(child)['id'] for child in card.get('energyCards', [])],
        'tools': [native_card(child)['id'] for child in card.get('tools', [])],
        'preEvolution': [native_card(child)['id'] for child in card.get('preEvolution', [])]}


def native_view(snapshot, viewer):
    view = snapshot['views']['owner' if viewer == 0 else 'opponent']
    if view['yourIndex'] != viewer or view['looking'] is not None or len(view['players']) != 2:
        raise ValueError('Native view identity, pending visibility or player inventory changed')
    players = []
    for player, observed in enumerate(view['players']):
        hand = observed['hand']
        if (('deck' in observed) or ('deckCards' in observed)
                or not isinstance(observed['prize'], list) or any(card is not None for card in observed['prize'])
                or (hand is not None if player != viewer else not isinstance(hand, list))):
            raise ValueError('Actual native privatization leaks a hidden hand, deck or prize identity')
        if player == viewer and len(hand) != observed['handCount']:
            raise ValueError('Native owner hand is incomplete')
        players.append({'handCount': observed['handCount'],
            'hand': [native_card(card)['id'] for card in hand] if player == viewer else None,
            'deckCount': observed['deckCount'], 'deck': None,
            'prizeCount': len(observed['prize']), 'prize': None,
            'active': [native_card(card) for card in observed['active']],
            'bench': [native_card(card) for card in observed['bench']],
            'discard': [native_card(card)['id'] for card in observed['discard']],
            'lostZone': [native_card(card)['id'] for card in observed['lostZone']]})
    return {'viewer': viewer, 'players': players, 'looking': None,
            'stadium': [native_card(card)['id'] for card in view['stadium']]}


def source_projection(snapshot, item):
    return [source_view(snapshot, viewer, item) for viewer in range(2)]


def native_projection(snapshot):
    physical = {card['serial']: card for card in snapshot['physical']}
    if len(physical) != len(snapshot['physical']):
        raise ValueError('Native physical serials are duplicated')

    def visible(card, owner, area):
        actual = physical.get(card.get('serial'))
        if (actual is None or actual['owner'] != owner or actual['area'] != area
                or actual['cardId'] != card['id'] or card['playerIndex'] != owner):
            raise ValueError('Visible native card does not bind its physical serial, owner, zone and definition')
        for field, child_area in (('energyCards', 8), ('tools', 9), ('preEvolution', 10)):
            for child in card.get(field, []):
                visible(child, owner, child_area)

    for viewer in range(2):
        view = snapshot['views']['owner' if viewer == 0 else 'opponent']
        for owner, player in enumerate(view['players']):
            for field, area in (('active', 4), ('bench', 5), ('discard', 3), ('lostZone', 25), ('hand', 2)):
                for card in player[field] or []:
                    visible(card, owner, area)
            for area, count in ((1, player['deckCount']), (2, player['handCount']), (6, len(player['prize']))):
                if sum(card['owner'] == owner and card['area'] in ((1, 14) if area == 1 else (area,))
                       for card in physical.values()) != count:
                    raise ValueError('Native private-zone count does not match the actual physical board')
        for card in view['stadium']:
            visible(card, card['playerIndex'], 7)
        visible_serials = {serial for serial, card in physical.items()
            if card['area'] in (3, 4, 5, 7, 8, 9, 10, 25) or card['area'] == 2 and card['owner'] == viewer}

        def audit_serials(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key.lower().endswith('serial') and (type(child) is not int or child not in visible_serials):
                        raise ValueError('Native supplemental view emits a serial hidden from this viewer')
                    if key.lower().endswith('serials') and (not isinstance(child, list)
                            or any(type(serial) is not int or serial not in visible_serials for serial in child)):
                        raise ValueError('Native supplemental view emits a hidden serial collection')
                    audit_serials(child)
            elif isinstance(value, list):
                for child in value:
                    audit_serials(child)

        audit_serials(view)
    return [native_view(snapshot, viewer) for viewer in range(2)]


def _bytes(reference):
    return read_pinned(reference['path'], reference['sha256'])


def _read(path, sha256):
    return json.loads(read_pinned(path, sha256))


def _reference(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()}


@lru_cache(maxsize=2)
def _prepared(path):
    prepared = _read(path, PREPARED_SHA256)
    root = path.parent
    for name, pin in prepared['outputs'].items():
        read_pinned(root / name, pin)
    if (prepared['schema'] != 'standard-information-boundary-preparation/1'
            or prepared['projectionSha256'] != PROJECTION_SHA256 or prepared['selectedCases'] != 472
            or prepared['selectedModes'] != 655 or prepared['heldCases'] or prepared['sourceRecaptures']
            or prepared['selectedKinds'] != {'single': 343, 'sequence': 30, 'controls': 49, 'missing': 50}):
        raise ValueError('Privacy preparation differs from the declared captured finite domain')
    baseline = _read(**prepared['baseline'])
    statuses = {case['caseId']: case['status'] for case in baseline['cases']}
    selected = _read(root / 'selected.json', prepared['outputs']['selected.json'])
    if (len(selected) != 472 or len({item['caseId'] for item in selected}) != 472
            or sum(len(item['modes']) for item in selected) != 655):
        raise ValueError('Privacy preparation lost or duplicated a declared case or mode')
    packet_items, provenances = {}, {}
    for reference in prepared['packets']:
        packet_path = Path(reference['path']).resolve(strict=True)
        packet = _read(packet_path, reference['sha256'])
        if packet['schema'] == 'standard-missing-pool-qualified-candidate/1':
            _, _, _, actions, _ = missing_domain(packet_path)
            ready = {item['key'].replace('source/', 'program/', 1) + '/effect':
                     {**item, 'caseId': item['key'].replace('source/', 'program/', 1) + '/effect'} for item in actions}
            expected, provenance = standard_missing_pool_expected, standard_missing_pool_provenance
        else:
            ready = effect_ready(packet_path)
            expected, provenance = standard_effect_expected, standard_effect_provenance
            measured = _read(**packet['measurementReceipt'])
            metadata = _read(**measured['metadataSelection'])
            native_ids = {row['cardSourceID']: row['qualifiedNativeId'] for row in metadata['rows']
                          if row.get('qualifiedNativeId') is not None}
            ordered_attacks = {row['cardId']: row['printedAttackIds'] for row in metadata['nativeCardActions']}
            if any(native_ids.get(item['sourceID']) != item['nativeCardId']
                   or ordered_attacks[item['nativeCardId']][item['nativeAttackOrdinal']] != item['attackId']
                   for item in ready.values()):
                raise ValueError('Privacy packet changed its measured printing identity or ordered attack')
        packet_items[reference['sha256']] = ready
        provenances[reference['sha256']] = provenance(packet_path)
        for item in selected:
            if item['packet'] != reference:
                continue
            binding = item['source']
            if (ready.get(item['effectCaseId']) != binding or binding['caseId'] != item['effectCaseId']
                    or item['caseId'] != item['effectCaseId'].rsplit('/', 1)[0] + '/information_boundaries'
                    or statuses.get(item['caseId']) != 'unobserved'):
                raise ValueError('Privacy requirement differs from its retained exact source action')
            expected(packet_path, item['effectCaseId'])
            modes = binding.get('modes', ['heads'])
            indexes = binding.get('sourceRecordsByMode', {'heads': binding.get('sourceRecord')})
            if item['modes'] != modes or item['sourceRecordsByMode'] != indexes:
                raise ValueError('Privacy fixture modes differ from the actual captured effect controls')
            for mode in modes:
                row = source_record(indexes[mode])
                source_key = binding.get('sourceKey', binding.get('key'))
                if (row['sourceID'] != binding['sourceID'] or row['actionIndex'] != binding['actionIndex']
                        or row['channel'] != 'useActions' or row['mode'] != mode
                        or row['key'] not in (source_key, source_key + '/controls/' + mode)
                        or not row['legal'] or not row['completed'] or row['status'] != 'executed'):
                    raise ValueError('Privacy source record lacks its exact positive action and control mode')
                actual = {stage: source_projection(row[stage], binding) for stage in ('before', 'after')}
                if actual != item['expectedByMode'][mode]:
                    raise ValueError('Prepared privacy expectation changed its actual source view bytes')
    if len(packet_items) != 4 or any(item['packet']['sha256'] not in packet_items for item in selected):
        raise ValueError('Privacy preparation changed its captured packet inventory')
    for key in ('corpusSha256', 'librarySha256', 'workerSha256'):
        if len({value[key] for value in provenances.values()}) != 1:
            raise ValueError('Privacy source packets use different captured runtimes or corpora')
    if next(iter(provenances.values()))['corpusSha256'] != LIVE_PINS.corpus_sha256:
        raise ValueError('Privacy source packets differ from the frozen complete source corpus')
    return prepared, selected, provenances


@lru_cache(maxsize=2)
def _measurement(prepared_path, measured_path, measured_pin):
    prepared, selected, _ = _prepared(prepared_path)
    measured = _read(measured_path, measured_pin)
    root = measured_path.parent
    for name, pin in measured['outputs'].items():
        read_pinned(root / name, pin)
    if (measured['schema'] != 'standard-information-boundary-native-measurement/1'
            or measured['prepared'] != _reference(prepared_path)
            or measured['requestedCases'] != 472 or measured['requestedModes'] != 655
            or measured['sourceRecaptures'] or not measured['sourceAndObjectRetained']
            or measured['library']['sha256'] != LIBRARY_SHA256
            or measured['registryReceipt']['sha256'] != REGISTRY_SHA256
            or measured['projection']['sha256'] != PROJECTION_SHA256
            or measured['projectionScope'] != prepared['projectionScope']):
        raise ValueError('Privacy measurement changed the prepared source or native candidate domain')
    _bytes(measured['projection'])
    _bytes(measured['driver'])
    actual = _read(root / 'actual.json', measured['outputs']['actual.json'])
    differences = _read(root / 'differences.json', measured['outputs']['differences.json'])
    held = _read(root / 'held.json', measured['outputs']['held.json'])
    ready = _read(root / 'ready.json', measured['outputs']['ready.json'])
    selected_by_id = {item['caseId']: item for item in selected}
    actual_by_key = {(row['caseId'], row['mode']): row for row in actual}
    held_keys = {(row['caseId'], row['mode']) for row in held}
    difference_keys = {(row['caseId'], row['mode']) for row in differences}
    expected_keys = {(item['caseId'], mode) for item in selected for mode in item['modes']}
    failed = {row['caseId'] for row in differences + held}
    expected_ready = [item for item in selected if item['caseId'] not in failed]
    if (len(actual_by_key) != len(actual) or len(held_keys) != len(held)
            or len(difference_keys) != len(differences) or set(actual_by_key) & held_keys
            or not difference_keys <= set(actual_by_key)
            or set(actual_by_key) | held_keys != expected_keys or ready != expected_ready
            or measured['readyCases'] != len(ready) or measured['heldModes'] != len(held)
            or measured['differences'] != len(differences)
            or measured['matchedModes'] != sum(row['matches'] is True for row in actual)):
        raise ValueError('Privacy measurement does not account for every declared source mode and held case')
    for key, row in actual_by_key.items():
        item, mode = selected_by_id[key[0]], key[1]
        expected = item['expectedByMode'][mode]
        if (row['sourceRecord'] != item['sourceRecordsByMode'][mode]
                or row['matches'] != (row['native'] == expected)
                or (key in difference_keys) != (row['native'] != expected)):
            raise ValueError('Privacy readiness does not follow the actual measured two-view observations')
    if not ready:
        raise ValueError('Privacy measurement has no qualified cases to admit')
    return measured, ready


def standard_privacy_configuration(prepared_path, measurement_path):
    prepared_path, measurement_path = Path(prepared_path).resolve(strict=True), Path(measurement_path).resolve(strict=True)
    measured_pin = _reference(measurement_path)['sha256']
    _measurement(prepared_path, measurement_path, measured_pin)
    return {'preparedPath': str(prepared_path), 'preparedSha256': PREPARED_SHA256,
        'measurementReceiptPath': str(measurement_path), 'measurementReceiptSha256': measured_pin}


def _ready(prepared_path, measurement_path):
    path = Path(measurement_path).resolve(strict=True)
    return _measurement(Path(prepared_path).resolve(strict=True), path, _reference(path)['sha256'])[1]


def _command(item, mode):
    binding = item['source']
    return {'operation': 'observeCapturedActionPrivacy', 'cardSourceID': binding['sourceID'],
        **{key: binding[key] for key in ('programSha256', 'sourceRowSha256', 'actionIndex', 'actionGuid',
                                       'actionSha256', 'nativeCardId', 'attackId')}, 'mode': mode}


def _scenario(item):
    return {'suite': SUITE, 'commands': [_command(item, mode) for mode in item['modes']], 'limitations': LIMITATIONS}


def standard_privacy_cases(prepared_path, measurement_path):
    return [{'caseId': item['caseId'], 'cabtCardId': item['source']['nativeCardId'], 'scenario': _scenario(item)}
            for item in _ready(prepared_path, measurement_path)]


def standard_privacy_expected(prepared_path, measurement_path, case_id):
    item = next(item for item in _ready(prepared_path, measurement_path) if item['caseId'] == case_id)
    return {'commands': _scenario(item)['commands'], 'responses': [item['expectedByMode'][mode] for mode in item['modes']],
            'limitations': LIMITATIONS}


def standard_privacy_provenance(prepared_path, measurement_path):
    prepared_path = Path(prepared_path).resolve(strict=True)
    _, _, provenances = _prepared(prepared_path)
    standard_privacy_configuration(prepared_path, measurement_path)
    source = next(iter(provenances.values()))
    return {key: source[key] for key in ('corpusSha256', 'librarySha256', 'workerSha256')} | {
        'prepared': _reference(prepared_path), 'measurementReceipt': _reference(measurement_path),
        'sourceRecaptures': 0, 'finiteDomain': LIMITATIONS}


def _execute(command):
    flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 0
    return subprocess.run([str(value) for value in command], capture_output=True, check=True,
                          timeout=180, creationflags=flags).stdout


def _generated_views(source):
    start = source.index('void PhysicalSnapshot(Board& board, JsonBuilder& output) {')
    end = source.index('\n}\n\nint main(', start)
    before = "output.append(']'); output.append('}');"
    after = "output.append(']'); output.appendCommaKey(\"views\"); board.Views(output); output.append('}');"
    body = source[start:end]
    if body.count(before) != 1:
        raise ValueError('Privacy wrapper changed its one reviewed snapshot insertion point')
    return source[:start] + body.replace(before, after) + source[end:], {'before': before, 'after': after}


def _fixtures(measured, prepared, candidate, artifacts, catalog):
    registry = _read(**measured['registryReceipt'])
    if (registry['sourceCommit'] != candidate['sourceCommit'] or registry['sourceSha256'] != candidate['sourceSha256']
            or measured['sourceCommit'] != candidate['sourceCommit'] or measured['sourceSha256'] != candidate['sourceSha256']
            or Path(registry['sourceRoot']).resolve() != Path(candidate['sourceRoot']).resolve()
            or measured['registryObjectSha256'] != registry['objectSha256']
            or candidate['librarySha256'] != LIBRARY_SHA256
            or Path(measured['library']['path']).resolve() != Path(artifacts['library']).resolve()):
        raise ValueError('Privacy fixture source, registration object or library belongs to another candidate')
    for name, pin in candidate['sourceSha256'].items():
        read_pinned(Path(candidate['sourceRoot']) / name, pin)
    read_pinned(registry['objectPath'], registry['objectSha256'])
    read_pinned(artifacts['library'], LIBRARY_SHA256)
    if len(measured['builds']) != len(prepared['packets']) or len(measured['commands']) != 12:
        raise ValueError('Privacy measurement omitted or duplicated an actual fixture build or execution')
    fixtures = {}
    for index, build in enumerate(measured['builds']):
        if build['packet'] != prepared['packets'][index]:
            raise ValueError('Privacy native fixtures reordered the captured source packet inventory')
        packet = _read(**build['packet'])
        original_measurement = _read(**packet['measurementReceipt'])
        original = Path(packet['measurementReceipt']['path']).parent
        original_source = read_pinned(original / 'native.cpp', original_measurement['outputs']['native.cpp']).decode('utf-8')
        original_source = original_source.replace('\r\n', '\n').replace('\r', '\n')
        generated, delta = _generated_views(original_source)
        if (build['originalSource'] != _reference(original / 'native.cpp') or build['delta'] != delta
                or _bytes(build['generatedSource']).decode('utf-8') != generated
                or _bytes(build['project']) != read_pinned(original / 'native.vcxproj', original_measurement['outputs']['native.vcxproj'])):
            raise ValueError('Privacy native wrapper changed more than the reviewed two-view serialization')
        fixture = Path(build['fixture']['path']).resolve(strict=True)
        read_pinned(fixture, build['fixture']['sha256'])
        destination = Path(build['generatedSource']['path']).parent
        execution = _read(**build['registryExecution'])
        if (execution['exitCode'] or not execution['sourceAndObjectRetained']
                or execution['sourceCommit'] != candidate['sourceCommit']
                or execution['registryReceiptSha256'] != REGISTRY_SHA256
                or execution['objectSha256'] != registry['objectSha256']
                or execution['command'][0] != registry['toolchain']['msbuildPath']
                or str(Path(build['project']['path'])) not in execution['command']
                or '/p:EngineSource=' + str(Path(candidate['sourceRoot'])) not in execution['command']):
            raise ValueError('Privacy adapter did not compile against the exact candidate registration object and source')
        for offset, stem in enumerate(('build', 'catalog', 'native')):
            command = json.loads((destination / (stem + '-command.json')).read_bytes())
            if command != measured['commands'][index * 3 + offset] or command['exitCode']:
                raise ValueError('Privacy measurement changed an actual native command')
            for suffix in ('stdout', 'stderr'):
                read_pinned(destination / (stem + '.' + suffix), command[suffix + 'Sha256'])
        if (json.loads((destination / 'catalog.stdout').read_bytes()) != catalog
                or json.loads(_execute([fixture, 'registry'])) != catalog):
            raise ValueError('Privacy fixture catalog differs from the initialized candidate library')
        fixtures[build['packet']['sha256']] = fixture
    return fixtures


def run_standard_privacy(cases, artifacts, candidate, cards, source_rows):
    family = candidate['fixtureFamilies']['standardPrivacy']
    prepared_path, measured_path = Path(artifacts['standardPrivacyPrepared']), Path(artifacts['standardPrivacyMeasurementReceipt'])
    if (family != standard_privacy_configuration(prepared_path, measured_path)
            or set(family) != {'preparedPath', 'preparedSha256', 'measurementReceiptPath', 'measurementReceiptSha256'}):
        raise ValueError('Privacy family differs from its exact prepared and measured receipts')
    prepared, _, _ = _prepared(prepared_path.resolve(strict=True))
    measured, ready = _measurement(prepared_path.resolve(strict=True), measured_path.resolve(strict=True), family['measurementReceiptSha256'])
    expected_cases = {case['caseId']: case for case in standard_privacy_cases(prepared_path, measured_path)}
    if (len({case['caseId'] for case in cases}) != len(cases)
            or any(expected_cases.get(case['caseId']) != {key: case[key] for key in ('caseId', 'cabtCardId', 'scenario')} for case in cases)):
        raise ValueError('Privacy cases changed their declared source action, mode or native identity')
    library = ctypes.CDLL(str(artifacts['library']))
    catalog = {}
    for export, key in (('AllCard', 'entries'), ('AllAttack', 'attacks')):
        function = getattr(library, export)
        function.argtypes, function.restype = [], ctypes.c_char_p
        catalog[key] = json.loads(function())
    if {row['cardId']: row for row in catalog['entries']} != cards:
        raise ValueError('Privacy actual candidate card catalog differs from the initialized runner')
    fixtures = _fixtures(measured, prepared, candidate, artifacts, catalog)
    source = {card['cardSourceID']: (card, row_hash, program) for card, _, row_hash, program in source_rows}
    selected_ids = {case['caseId'] for case in cases}
    selected = [item for item in ready if item['caseId'] in selected_ids]
    for item in selected:
        binding = item['source']
        card, row_hash, program = source[binding['sourceID']]
        action = card['useActions'][binding['actionIndex']]
        ordinal = sum(previous['actionType'] == 1 for previous in card['useActions'][:binding['actionIndex']])
        if (action['actionType'] != 1 or program != binding['programSha256'] or row_hash != binding['sourceRowSha256']
                or digest(action) != binding['actionSha256'] or action['actionGuid'] != binding['actionGuid']
                or cards[binding['nativeCardId']]['attacks'][ordinal] != binding['attackId']
                or binding.get('nativeAttackOrdinal', ordinal) != ordinal):
            raise ValueError('Privacy source row/program/GUID or current native ordered attack changed')
    observed = {}
    for packet_pin, fixture in fixtures.items():
        requests = {item['caseId'] + '/' + mode: (item, mode)
                    for item in selected if item['packet']['sha256'] == packet_pin for mode in item['modes']}
        if not requests:
            continue
        with tempfile.TemporaryDirectory(prefix='cabt-standard-privacy-') as temporary:
            commands = Path(temporary) / 'requests.tsv'
            commands.write_text(''.join(f'{key}\t{item["source"]["nativeCardId"]}\tattack\t{item["source"]["attackId"]}\tattack\t{mode}\n'
                for key, (item, mode) in requests.items()), encoding='utf-8')
            rows = [json.loads(line) for line in _execute([fixture, commands]).splitlines()]
        actual = {row['key']: row for row in rows}
        if len(actual) != len(rows) or set(actual) != set(requests):
            raise ValueError('Privacy actual native replay omitted or duplicated a requested mode')
        for key, (item, mode) in requests.items():
            row, binding = actual[key], item['source']
            captured = source_record(item['sourceRecordsByMode'][mode])
            actors = [card for card in row['before']['physical'] if card['label'] == 'actor']
            prompt = row['initialPrompt']
            if (row.get('error') or not row['legal'] or not row['completed']
                    or row['requestedNativeCardId'] != binding['nativeCardId'] or row['requestedNativeAttackId'] != binding['attackId']
                    or len(actors) != 1 or actors[0]['serial'] != row['requestedActorSerial']
                    or actors[0]['cardId'] != binding['nativeCardId'] or actors[0]['owner'] != 0 or actors[0]['area'] != 4
                    or prompt['type'] != 0 or prompt['context'] != 0
                    or not any(option.get('type') == 13 and option.get('attackId') == binding['attackId'] for option in prompt['option'])
                    or row['coins'] != captured['coins'] or len(row['choices']) != row['coins']
                    or any(not choice['validated'] or choice['selected'] != [0]
                        or choice['prompt']['type'] != 9 or choice['prompt']['context'] != 46
                        or choice['prompt']['minCount'] != 1 or choice['prompt']['maxCount'] != 1
                        or choice['prompt']['option'] != [{'type': 1}, {'type': 2}] for choice in row['choices'])):
                raise ValueError('Privacy actual native action, completion or coin selections changed')
            physical = {stage: normalize_native(row[stage]) for stage in ('before', 'after')}
            if physical != binding.get('expectedByMode', {}).get(mode, binding.get('expected')):
                raise ValueError('Privacy query changed its retained complete physical effect observation')
            views = {stage: native_projection(row[stage]) for stage in ('before', 'after')}
            if views != item['expectedByMode'][mode]:
                raise ValueError('Fresh native privacy views differ from their actual captured source views')
            observed[item['caseId'], mode] = views
    return [{'caseId': case['caseId'], 'cabtCardId': case['cabtCardId'], 'scenarioSha256': digest(case['scenario']),
        'observation': {'commands': case['scenario']['commands'],
            'responses': [observed[case['caseId'], command['mode']] for command in case['scenario']['commands']],
            'limitations': LIMITATIONS},
        'scenarioComponents': [{'suite': SUITE, 'scenarioSha256': digest(case['scenario'])}]} for case in cases]
