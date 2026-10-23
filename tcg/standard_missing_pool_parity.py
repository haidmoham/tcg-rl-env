"""Finite observations for the measured Standard missing-card definition batch."""

import ctypes
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile

from tcg.live_parity import LIVE_PINS, digest, read_pinned
from tcg.native_metadata_parity import (NATIVE_METADATA_LABEL_POLICY, internal_catalog,
    metadata_label_projection, native_projection, source_metadata, source_projection)
from tcg.standard_effect_parity import normalize_native, normalize_source, source_record


PACKET_SHA256 = '8881b40de5ee6496c9414166c1f617a332613464cfabb46eea0f1d0bdf79a28e'
SUITE = 'standard-missing-pool/1'
LIMITATIONS = [
    'Only the 67 exact printing metadata observations and 50 declared positive attack samples are admitted.',
    'Every physical card identity, owner, area, attachment parent, damage, status and turn is compared.',
    'Coins use the retained all-heads source observations and validated native coin selections.',
    'Other action facets, alternative states, interactions, information boundaries and rulings remain unqualified.',
]


def _read(path, expected):
    return json.loads(read_pinned(path, expected))


@lru_cache(maxsize=2)
def _domain(packet_path):
    packet = _read(packet_path, PACKET_SHA256)
    if (packet['schema'] != 'standard-missing-pool-qualified-candidate/1'
            or packet['candidatePrograms'] != 35 or packet['candidatePrintings'] != 67
            or packet['metadataSamples'] != 67 or packet['positiveActionSamples'] != 50
            or packet['sourceRecaptures'] or packet['capturedRuntime']['corpusSha256'] != LIVE_PINS.corpus_sha256):
        raise ValueError('Missing-pool packet differs from its measured finite domain')
    reference = packet['measurementReceipt']
    measurement = _read(reference['path'], reference['sha256'])
    root = Path(reference['path']).parent
    for path, pin in packet['capturedRuntime']['files'].items():
        read_pinned(path, pin)
    for path, pin in packet['fixtureSources'].items():
        read_pinned(path, pin)
    for reference in packet['sourceOracle'].values():
        if isinstance(reference, dict):
            read_pinned(reference['path'], reference['sha256'])
    for key in ('capturedRuntimeReceipt', 'actualCaptureReference'):
        reference = packet[key]
        read_pinned(reference['path'], reference['sha256'])
    for path, pin in measurement['inputPins'].items():
        read_pinned(path, pin)
    for name, pin in measurement['outputs'].items():
        read_pinned(root / name, pin)
    metadata = _read(root / 'metadata-results.json', measurement['outputs']['metadata-results.json'])
    actions = _read(root / 'prepared-actions.json', measurement['outputs']['prepared-actions.json'])
    programs = _read(root / 'ready-programs.json', measurement['outputs']['ready-programs.json'])
    if (len(metadata) != 67 or len(actions) != 50 or len(programs) != 35
            or len({item['cardSourceID'] for item in metadata}) != 67
            or len({item['key'] for item in actions}) != 50
            or len({item['programSha256'] for item in programs}) != 35
            or not all(item['matches'] and item['source'] == item['native'] for item in metadata)):
        raise ValueError('Missing-pool packet omitted, duplicated or changed its measured observations')
    program_ids = {item['programSha256'] for item in programs}
    printing_ids = {sid for item in programs for sid in item['cardSourceIDs']}
    attack_keys = {f'source/{item["programSha256"]}/useActions/{index}'
                   for item in programs for index in range(len(item['attackIds']))}
    if (program_ids != {item['programSha256'] for item in metadata}
            or printing_ids != {item['cardSourceID'] for item in metadata}
            or attack_keys != {item['key'] for item in actions}):
        raise ValueError('Missing-pool program, printing and attack inventories do not form the complete measured batch')
    return packet, measurement, metadata, actions, programs


def _effect_command(item):
    return {'operation': 'executeCapturedMissingPoolAttack', 'cardSourceID': item['sourceID'],
            **{key: item[key] for key in ('programSha256', 'sourceRowSha256', 'actionIndex',
                'actionGuid', 'actionSha256', 'nativeCardId', 'attackId')}, 'mode': 'heads'}


def _metadata_command(item, row_hash):
    return {'operation': 'nativeMissingPoolMetadata', 'cardSourceID': item['cardSourceID'],
            'sourceRowSha256': row_hash, 'programSha256': item['programSha256'],
            'nativeCardId': item['cabtCardId']}


def _scenario(command, facet):
    return {'suite': SUITE, 'facet': facet, 'commands': [command], 'limitations': LIMITATIONS}


def standard_missing_pool_configuration(packet_path, replay_receipt=None):
    packet_path = Path(packet_path).resolve(strict=True)
    packet = _domain(packet_path)[0]
    replay = Path(replay_receipt or packet['measurementReceipt']['path']).resolve(strict=True)
    return {'packetPath': str(packet_path), 'packetSha256': PACKET_SHA256,
            'replayReceiptPath': str(replay), 'replayReceiptSha256': sha256(replay.read_bytes()).hexdigest()}


def standard_missing_pool_cases(packet_path):
    _, _, metadata, actions, programs = _domain(Path(packet_path).resolve(strict=True))
    hashes = {sid: pin for item in programs for sid, pin in item['cardSha256'].items()}
    return [
        {'caseId': f'row/{item["cardSourceID"]}/metadata_identity', 'cabtCardId': item['cabtCardId'],
         'scenario': _scenario(_metadata_command(item, hashes[item['cardSourceID']]), 'metadata_identity')}
        for item in metadata
    ] + [
        {'caseId': item['key'].replace('source/', 'program/', 1) + '/effect', 'cabtCardId': item['nativeCardId'],
         'scenario': _scenario(_effect_command(item), 'effect')} for item in actions
    ]


@lru_cache(maxsize=2)
def _expected(packet_path):
    packet, _, metadata, actions, _ = _domain(packet_path)
    cases = {case['caseId']: case for case in standard_missing_pool_cases(packet_path)}
    values = {}
    for item in metadata:
        case_id = f'row/{item["cardSourceID"]}/metadata_identity'
        values[case_id] = {'commands': cases[case_id]['scenario']['commands'], 'responses': [item['source']],
                           'limitations': LIMITATIONS}
    for item in actions:
        index = item['sourceRecord']
        row = source_record(index)
        if (packet['usedSourceRecords'].get(item['key']) != index
                or packet['rawSourceFiles'].get(index['rawPath']) != index['rawSha256']
                or index['oracleSha256'] != packet['sourceOracle']['compiledDll']['sha256']
                or index['fixtureContract'] != 'rich-board-v1-prize-history-unset'
                or row['key'] != item['key'] or row['sourceID'] != item['sourceID']
                or row['channel'] != 'useActions' or row['actionIndex'] != item['actionIndex']
                or row['mode'] != 'heads' or row['status'] != 'executed' or not row['legal']
                or not row['completed'] or row['choices'] or row['coins'] != item['coins']):
            raise ValueError('Missing-pool source action lacks its exact positive actual capture')
        states = {stage: normalize_source(row[stage], row['before'], {'cardSourceID': item['sourceID']},
                                         item['nativeCardId']) for stage in ('before', 'after')}
        if states != item['expected']:
            raise ValueError('Missing-pool expected states differ from their actual captured bytes')
        case_id = item['key'].replace('source/', 'program/', 1) + '/effect'
        values[case_id] = {'commands': cases[case_id]['scenario']['commands'],
                          'responses': [{**states, 'coins': row['coins']}], 'limitations': LIMITATIONS}
    return values


def standard_missing_pool_expected(packet_path, case_id):
    return _expected(Path(packet_path).resolve(strict=True))[case_id]


def standard_missing_pool_provenance(packet_path):
    packet_path = Path(packet_path).resolve(strict=True)
    packet = _domain(packet_path)[0]
    return {key: packet['capturedRuntime'][key] for key in ('corpusSha256', 'librarySha256', 'workerSha256')} | {
        'packetPath': str(packet_path), 'packetSha256': PACKET_SHA256,
        'measurementReceipt': packet['measurementReceipt'], 'sourceRecaptures': 0, 'finiteDomain': LIMITATIONS}


def _execute(command):
    flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 0
    return subprocess.run(command, capture_output=True, check=True, timeout=180, creationflags=flags).stdout


def _candidate_replay(artifacts, candidate, packet, original):
    path = Path(artifacts['standardMissingPoolReplayReceipt']).resolve(strict=True)
    family = candidate['fixtureFamilies']['standardMissingPool']
    replay = _read(path, family['replayReceiptSha256'])
    root = path.parent
    expected_counts = {'metadataPrintings': 67, 'metadataMatches': 67, 'preparedActions': 50,
        'matchedActions': 50, 'heldActions': 0, 'readyPrograms': 35, 'readyPrintings': 67, 'sourceRecaptures': 0}
    if (replay['schema'] != 'standard-missing-pool-native-replay/1'
            or any(replay[key] != value for key, value in expected_counts.items())
            or replay['failedPrograms'] or not replay['sourceAndObjectRetained'] or not replay['actualDllCatalogEqual']
            or replay['sourceCommit'] != candidate['sourceCommit'] or replay['sourceSha256'] != candidate['sourceSha256']
            or replay['library']['sha256'] != candidate['librarySha256']
            or replay['inputPins'] != original['inputPins']):
        raise ValueError('Missing-pool replay does not bind the complete domain and exact candidate')
    for relative, pin in candidate['sourceSha256'].items():
        read_pinned(Path(candidate['sourceRoot']) / relative, pin)
    for name, pin in replay['outputs'].items():
        read_pinned(root / name, pin)
    registry_ref = replay['registryReceipt']
    registry = _read(registry_ref['path'], registry_ref['sha256'])
    if (registry['sourceCommit'] != candidate['sourceCommit'] or registry['sourceSha256'] != candidate['sourceSha256']
            or registry['objectSha256'] != replay['registryObjectSha256']):
        raise ValueError('Missing-pool registry object belongs to a different candidate')
    read_pinned(registry['objectPath'], registry['objectSha256'])
    for key in ('fixture', 'probe'):
        read_pinned(replay[key]['path'], replay[key]['sha256'])
    read_pinned(artifacts['library'], candidate['librarySha256'])
    original_root = Path(packet['measurementReceipt']['path']).parent
    for name in ('native.cpp', 'native.vcxproj'):
        if replay['outputs'][name] != original['outputs'][name]:
            raise ValueError('Missing-pool fresh fixture source differs from the reviewed capture adapter')
    for name in ('probe.cpp', 'probe.vcxproj'):
        relative = 'experiments/native_metadata/' + name
        if candidate['sourceSha256'][relative] != packet['sourceSha256'][relative]:
            raise ValueError('Missing-pool metadata adapter source changed')
    for name in ('build-library', 'build-native', 'build-probe'):
        command = _read(root / (name + '-command.json'), replay['outputs'][name + '-command.json'])
        if command['exitCode'] or command['registryObjectSha256'] != registry['objectSha256']:
            raise ValueError('Missing-pool compiled adapter did not bind the exact candidate registry object')
    for name in ('ready-programs.json', 'prepared-actions.json'):
        if _read(root / name, replay['outputs'][name]) != _read(original_root / name, original['outputs'][name]):
            raise ValueError('Missing-pool replay changed the retained source observation inventory')
    return replay


def run_standard_missing_pool(cases, artifacts, candidate, cards, source_rows):
    packet_path = Path(artifacts['standardMissingPoolPacket']).resolve(strict=True)
    family = candidate['fixtureFamilies']['standardMissingPool']
    if (family['packetSha256'] != PACKET_SHA256 or Path(family['packetPath']).resolve() != packet_path
            or Path(family['replayReceiptPath']).resolve() != Path(artifacts['standardMissingPoolReplayReceipt']).resolve()):
        raise ValueError('Missing-pool fixture family changed its pinned paths')
    packet, original, metadata, actions, programs = _domain(packet_path)
    replay = _candidate_replay(artifacts, candidate, packet, original)
    expected_cases = {case['caseId']: case for case in standard_missing_pool_cases(packet_path)}
    if (len({case['caseId'] for case in cases}) != len(cases)
            or any(expected_cases.get(case['caseId']) != {key: case[key] for key in ('caseId', 'cabtCardId', 'scenario')}
                   for case in cases)):
        raise ValueError('Missing-pool case changed its exact measured scenario')
    source = {card['cardSourceID']: (card, row_hash, program) for card, _, row_hash, program in source_rows}
    library = ctypes.CDLL(str(artifacts['library']))
    catalog = {}
    for export, field in (('AllCard', 'entries'), ('AllAttack', 'attacks')):
        function = getattr(library, export)
        function.argtypes, function.restype = [], ctypes.c_char_p
        catalog[field] = json.loads(function())
    fixture, probe = Path(replay['fixture']['path']), Path(replay['probe']['path'])
    if ({row['cardId']: row for row in catalog['entries']} != cards
            or json.loads(_execute([str(fixture), 'registry'])) != catalog
            or json.loads(_execute([str(probe), 'registry'])) != catalog):
        raise ValueError('Missing-pool fresh adapters and actual candidate DLL catalogs differ')
    raw = json.loads(_execute([str(probe)]))
    if len({row['cardId'] for row in raw}) != len(raw) or {row['cardId'] for row in raw} != set(cards):
        raise ValueError('Missing-pool metadata export omitted or duplicated a native definition')
    native = {row['cardId']: row for row in raw}
    by_internal = internal_catalog(raw)
    projections = {}
    bindings = {item['programSha256']: item for item in programs}
    metadata_path = next(Path(path) for path in original['inputPins'] if Path(path).name == 'metadata.json')
    captured_metadata = {row['cardSourceID']: row for row in _read(metadata_path, original['inputPins'][str(metadata_path)])}
    for item in metadata:
        sid = item['cardSourceID']
        card, row_hash, program = source[sid]
        binding = bindings[program]
        captured = captured_metadata[sid]
        if (program != item['programSha256'] or row_hash != binding['cardSha256'][sid]
                or item['cabtCardId'] != binding['cabtCardId'] or sid not in binding['cardSourceIDs']
                or captured['rawRowSha256'] != row_hash or captured['programSha256'] != program
                or cards[item['cabtCardId']]['attacks'] != binding['attackIds']):
            raise ValueError('Missing-pool printing/program/native ordered-attack binding changed')
        decoded = source_metadata(card)
        expected = metadata_label_projection(source_projection(decoded, decoded['passiveActions']), NATIVE_METADATA_LABEL_POLICY)
        captured_projection = metadata_label_projection(source_projection(captured, []), NATIVE_METADATA_LABEL_POLICY)
        actual = metadata_label_projection(native_projection(native[item['cabtCardId']], by_internal), NATIVE_METADATA_LABEL_POLICY)
        if expected != item['source'] or captured_projection != expected or actual != expected:
            raise ValueError('Missing-pool raw source, actual captured typed metadata and fresh native facts differ')
        projections[f'row/{sid}/metadata_identity'] = actual
    selected_actions = {}
    for item in actions:
        card, row_hash, program = source[item['sourceID']]
        action = card['useActions'][item['actionIndex']]
        ordinal = sum(previous['actionType'] == 1 for previous in card['useActions'][:item['actionIndex']])
        if (program != item['programSha256'] or row_hash != item['sourceRowSha256']
                or digest(action) != item['actionSha256'] or action['actionGuid'] != item['actionGuid']
                or item['nativeCardId'] != bindings[program]['cabtCardId']
                or cards[item['nativeCardId']]['attacks'][ordinal] != item['attackId']
                or item['key'] != f'source/{program}/useActions/{item["actionIndex"]}'):
            raise ValueError('Missing-pool source action, GUID or native attack order changed')
        selected_actions[item['key']] = item
    _expected(packet_path)
    with tempfile.TemporaryDirectory(prefix='cabt-standard-missing-pool-') as temporary:
        request_path = Path(temporary) / 'requests.tsv'
        request_path.write_text(''.join(f'{item["key"]}\t{item["nativeCardId"]}\tattack\t{item["attackId"]}\tattack\theads\n'
                                        for item in actions), encoding='utf-8')
        replay_rows = [json.loads(line) for line in _execute([str(fixture), str(request_path)]).splitlines()]
    actual_rows = {row['key']: row for row in replay_rows}
    if len(actual_rows) != len(replay_rows) or set(actual_rows) != set(selected_actions):
        raise ValueError('Missing-pool fresh replay omitted or duplicated an attack')
    observations = {}
    for key, item in selected_actions.items():
        row = actual_rows[key]
        actors = [physical for physical in row['before']['physical'] if physical['label'] == 'actor']
        prompt = row['initialPrompt']
        available = any(option.get('type') == 13 and option.get('attackId') == item['attackId'] for option in prompt['option'])
        if (row.get('error') or not row['legal'] or not row['completed']
                or row['requestedNativeCardId'] != item['nativeCardId'] or row['requestedNativeAttackId'] != item['attackId']
                or len(actors) != 1 or actors[0]['serial'] != row['requestedActorSerial']
                or prompt['type'] != 0 or prompt['context'] != 0 or not available
                or row['coins'] != item['coins'] or len(row['choices']) != row['coins']
                or any(not choice['validated'] or choice['selected'] != [0]
                       or choice['prompt']['type'] != 9 or choice['prompt']['context'] != 46
                       or choice['prompt']['minCount'] != 1 or choice['prompt']['maxCount'] != 1
                       or choice['prompt']['option'] != [{'type': 1}, {'type': 2}]
                       for choice in row['choices'])):
            raise ValueError('Missing-pool native request, legal attack, completion or coin choices changed')
        case_id = key.replace('source/', 'program/', 1) + '/effect'
        observations[case_id] = {'commands': expected_cases[case_id]['scenario']['commands'],
            'responses': [{**{stage: normalize_native(row[stage]) for stage in ('before', 'after')}, 'coins': row['coins']}],
            'limitations': LIMITATIONS}
    for case_id, projection in projections.items():
        observations[case_id] = {'commands': expected_cases[case_id]['scenario']['commands'],
                               'responses': [projection], 'limitations': LIMITATIONS}
    if observations != _expected(packet_path):
        raise ValueError('Missing-pool fresh native observations differ from actual captured source')
    return [{'caseId': case['caseId'], 'cabtCardId': case['cabtCardId'],
             'scenarioSha256': digest(case['scenario']), 'observation': observations[case['caseId']],
             'scenarioComponents': [{'suite': SUITE, 'scenarioSha256': digest(case['scenario'])}]} for case in cases]
