"""Finite physical-effect observations from retained source oracles and fresh native replay."""

import ctypes
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile

from tcg.live_parity import LIVE_PINS, digest, read_pinned


PACKET_SHA256 = "7ddab3c992647a13fb0be3626a54a99d23e6b8248ad16247a24a410d400a8c17"
SUITE = "standard-physical-effect/1"
READY_CASES = 343
EXPANDED_PACKET_SHA256 = "9033f78eb9fda47bf5cbc0352e87aec4bedb16130d00c08a334ab131b31521b4"
CONTROL_PACKET_SHA256 = "d6e9f9cb17fb9f53a6ed796fe750f931760d63a1a105e4c69e28bc18d20cc111"
CONTROL_MODES = {'variable': ['heads', 'lean', 'behind-prizes', 'homogeneous', 'wounded'],
                 'ignore': ['heads', 'weakness', 'resistance', 'shield']}
CALIBRATION = {'heads': 70, 'weakness': 140, 'resistance': 40, 'shield': 50}

NATIVE_ZONES = {0: 'none', 1: 'deck', 2: 'hand', 3: 'discard', 4: 'active', 5: 'bench', 6: 'prize',
                7: 'stadium', 8: 'energy', 9: 'tool', 10: 'preEvolution', 12: 'pending', 13: 'playing', 14: 'deck'}


SOURCE_ZONES = {7: 'deck', 8: 'deck', 9: 'discard', 10: 'discard', 11: 'hand', 12: 'hand',
                13: 'bench', 14: 'bench', 15: 'active', 16: 'active', 19: 'prize', 20: 'prize'}


SOURCE_ENERGY = {f'bw1_{105 + i}': i + 1 for i in range(8)}


LIMITATIONS = ['One exact rich-board-v1-prize-history-unset positive attack sample.',
               'Only explicit constant damage to opposing Active followed by ResolveAttack.',
               'Native attack must have no flags, pre/post effects or conditional alternate cost.',
               'Every physical identity, owner, area, attachment parent, damage and status absence is compared.',
               'Only the exact opponent turn-start top-card draw may move a physical card between zones.',
               'No prompts, coins, status effects, terminal outcomes or randomized draw effects are admitted.',
               'Other actions, future states, legal restrictions, information boundaries, interactions and rulings remain unqualified.']

EXPANDED_LIMITATIONS = ['One exact rich-board-v1-prize-history-unset positive attack sample.',
               'Only explicit constant ApplyDamage operations followed by ResolveAttack.',
               'Every declared damage operation must visibly change a physical Pokemon damage value.',
               'Native attack has no flags, cost overrides, pre-effects or non-damage post-effects.',
               'Every physical identity, owner, area, attachment parent, damage and status absence is compared.',
               'Only the exact opponent turn-start top-card draw may move a physical card between zones.',
               'Variable expressions, ignored weakness/resistance/effects, prompts, coins, statuses, knockout and future effects remain held.',
               'No other action facet, arbitrary state, interaction or ruling is qualified.']

CONTROL_LIMITATIONS = [
    'Only the declared finite fixture matrix and complete physical observations are qualified.',
    'Variable damage must change under the explicit energy/board/identity/damage controls.',
    'Ignore branches use calibrated weakness, resistance and actual damage-reduction passive controls.',
    'Prompts, coins, terminal outcomes, special conditions and nonphysical or persistent native effects remain held.',
    'No legal-prompts, privacy, interactions or rules facet is added.',
]

DOMAINS = {
    PACKET_SHA256: {'schema': 'standard-constant-damage-effect-candidate/1', 'kind': 'single',
                   'prepared': 343, 'ready': 343, 'differences': 0, 'heldNative': 0,
                   'heldSource': 3045, 'limitations': LIMITATIONS},
    EXPANDED_PACKET_SHA256: {'schema': 'standard-multiple-constant-damage-effect-candidate/1', 'kind': 'sequence',
                            'prepared': 32, 'ready': 30, 'differences': 1, 'heldNative': 1,
                            'heldSource': 3356, 'limitations': EXPANDED_LIMITATIONS},
    CONTROL_PACKET_SHA256: {'schema': 'standard-physical-effect-discriminating-controls/1', 'kind': 'controls',
                           'ready': 49, 'limitations': CONTROL_LIMITATIONS},
}

MEASURED_DAMAGE_TARGETS = frozenset({
    '09cef8a901e3b03f0d2e26f55cd32a3266e8d0c17f009b25eb5134ad101e938f',
    '5ac1227ae9c6250291ccdb40ceff15895690383c750594347ea6b1b132d67654',
    'aa6325d80cd1fb83a4da3255463be581546f4d13f80b3a7f65aeea073abd1595',
    '05668c1c516e06121a7eebe0c920f6ce30e6d7bb38d12eaed021a4c8378b10ad',
    '3f52d62228ac7f8f9d3c029ed37752a3132c71209877b00683624cd77fa43a53',
})


def damage_sequence(action):
    if (set(action) - {'actionGuid', 'actionName', 'actionProperties', 'actionSourceID', 'actionType', 'energyCost', 'subActions'}
            or action.get('actionType') != 1 or action.get('actionProperties') != [1]):
        return None
    operations = action.get('subActions', [])
    if (not 2 <= len(operations) <= 8 or operations[-1].get('$type') != 'MatchLogic.ResolveAttack, MatchLogic'
            or set(operations[-1]) - {'$type', 'subActionName'}):
        return None
    amounts = []
    for effect in operations[:-1]:
        if (effect.get('$type') != 'MatchLogic.ApplyDamage, MatchLogic'
                or set(effect) - {'$type', 'damageValue', 'editorSettings', 'subActionName', 'target',
                                  'applyEffectsOpActive', 'applyResistanceActive', 'applyWeaknessActive'}
                or any(effect.get(name) is False for name in ('applyEffectsOpActive', 'applyResistanceActive', 'applyWeaknessActive'))
                or digest(effect.get('target')) not in MEASURED_DAMAGE_TARGETS):
            return None
        value = effect.get('damageValue', {})
        number = value.get('value1', {})
        amount = number.get('explicitValue')
        if (set(value) != {'$type', 'value1'} or value.get('$type') != 'MatchLogic.SingleVariableTotal, MatchLogic'
                or set(number) != {'$type', 'explicitValue'} or number.get('$type') != 'MatchLogic.ExplicitNumberVariable, MatchLogic'
                or type(amount) is not int or not 0 < amount <= 1000 or amount % 10):
            return None
        amounts.append(amount)
    return amounts


def plain_damage(action):
    if set(action) - {'actionGuid', 'actionName', 'actionProperties', 'actionSourceID', 'actionType', 'energyCost', 'subActions'}:
        return None
    if action.get('actionType') != 1 or action.get('actionProperties') != [1]:
        return None
    sub = action.get('subActions', [])
    kind = lambda name: 'MatchLogic.' + name + ', MatchLogic'
    if len(sub) != 2 or sub[0].get('$type') != kind('ApplyDamage') or sub[1].get('$type') != kind('ResolveAttack'):
        return None
    damage, resolve = sub
    if (set(damage) - {'$type', 'damageValue', 'editorSettings', 'subActionName', 'target'}
            or set(resolve) - {'$type', 'subActionName'} or damage.get('editorSettings', 'DamageOpponentsActive') != 'DamageOpponentsActive'):
        return None
    target = damage.get('target', {})
    if (set(target) - {'entityLocations', 'groupingSize', 'editorSettings'}
            or target.get('entityLocations') != [{'category': 12, 'entityPositions': [7]}]
            or target.get('groupingSize') != {'$type': kind('AllNumberVariableTotal')}
            or target.get('editorSettings', 'OpponentActive') != 'OpponentActive'):
        return None
    value = damage.get('damageValue', {})
    number = value.get('value1', {})
    amount = number.get('explicitValue')
    if (set(value) != {'$type', 'value1'} or value.get('$type') != kind('SingleVariableTotal')
            or set(number) != {'$type', 'explicitValue'} or number.get('$type') != kind('ExplicitNumberVariable')
            or type(amount) is not int or amount <= 0 or amount > 1000 or amount % 10):
        return None
    return amount


def source_record(index):
    with Path(index['rawPath']).open('rb') as stream:
        stream.seek(index['byteOffset'])
        raw = stream.read(index['byteLength'])
    if sha256(raw).hexdigest() != index['rawRecordSha256']:
        raise ValueError('Actual captured source record changed')
    return json.loads(raw.decode(index['encoding']))


def source_identity(source_id, card, native_id):
    if source_id == card['cardSourceID']:
        return native_id
    if source_id == 'whole-export-control':
        return 9990600
    if source_id == 'me5_78':
        return 1428
    if source_id in SOURCE_ENERGY:
        return SOURCE_ENERGY[source_id]
    if source_id.startswith('bw1_'):
        raise ValueError('Unrecognized fixture basic Energy identity')
    return None


def normalize_source(snapshot, before, card, native_id):
    board = before['board']
    parents = {board['p1Active']['entityID']: 'actor', board['p2Active']['entityID']: 'p1-active'}
    physical = []
    for row in snapshot['physical']:
        if any(row['status']):
            raise ValueError('Source special condition amount requires another domain')
        parent = parents.get(row.get('parent')) if row.get('parent') else None
        if row.get('parent') and parent is None:
            raise ValueError('Unknown captured physical attachment parent')
        position = row['position']
        zone = 'energy' if parent else SOURCE_ZONES[position]
        identity = source_identity(row['sourceID'], card, native_id)
        if identity is None:
            raise ValueError('Unbound captured physical definition')
        physical.append({'label': row['label'], 'cardId': identity, 'owner': 0 if position in (7, 9, 11, 13, 15, 19) else 1,
                         'zone': zone, 'parent': parent, 'damage': row['damage'], 'status': [0] * 5})
    turns = snapshot['board']['boardEntity']['entityMetaData']['TotalTurnCount']
    if len(turns) != 1 or snapshot['gameOver']:
        raise ValueError('Source terminal or turn counter outside finite domain')
    return {'currentPlayer': snapshot['currentPlayer'] - 1, 'turn': turns[0]['value'] + 1, 'terminal': False,
            'physical': sorted(physical, key=lambda row: row['label'])}


def normalize_native(snapshot):
    by_move = {row['moveCounter']: row['label'] for row in snapshot['physical']}
    if len(by_move) != len(snapshot['physical']):
        raise ValueError('Native physical move counters are not unique')
    if snapshot['gameResult'] or any(row['badStatus'] or row['poisonDamageCounter'] or row['burned'] for row in snapshot['activeStatus']):
        raise ValueError('Native terminal or special condition requires another domain')
    physical = []
    for row in snapshot['physical']:
        parent = by_move.get(row['parentMoveCounter']) if row['area'] in (8, 9, 10) else None
        if row['area'] in (8, 9, 10) and parent is None:
            raise ValueError('Native physical attachment parent is unresolved')
        physical.append({'label': row['label'], 'cardId': row['cardId'], 'owner': row['owner'],
                         'zone': NATIVE_ZONES[row['area']], 'parent': parent, 'damage': row['damage'], 'status': [0] * 5})
    return {'currentPlayer': snapshot['currentPlayer'], 'turn': snapshot['turn'], 'terminal': False,
            'physical': sorted(physical, key=lambda row: row['label'])}


def normalize_control_source(snapshot, before, source_id, native_id):
    parents = {}
    for player in range(2):
        stem = f'p{player + 1}'
        active = before['board'][stem + 'Active']
        if active:
            parents[active['entityID']] = 'actor' if player == 0 else 'p1-active'
        parents.update({entity['entityID']: f'p{player}-bench{index}'
                        for index, entity in enumerate(before['board'][stem + 'Bench'])})
    physical = []
    for row in snapshot['physical']:
        parent = parents.get(row.get('parent')) if row.get('parent') else None
        if any(row['status']) or (row.get('parent') and parent is None):
            raise ValueError('Captured control has a special condition or unresolved physical parent')
        identity = source_identity(row['sourceID'], {'cardSourceID': source_id}, native_id)
        if identity is None:
            identity = {'sv6_128': 119, 'sv6_129': 120, 'sv6_130': 121}.get(row['sourceID'])
        if identity is None:
            raise ValueError('Captured control contains an unbound physical definition')
        zone = ('preEvolution' if row['label'].startswith('p0-parent') else 'energy') if parent else SOURCE_ZONES[row['position']]
        physical.append({'label': row['label'], 'cardId': identity,
                         'owner': 0 if row['position'] in (7, 9, 11, 13, 15, 19) else 1,
                         'zone': zone, 'parent': parent, 'damage': row['damage'], 'status': [0] * 5})
    turns = snapshot['board']['boardEntity']['entityMetaData']['TotalTurnCount']
    if len(turns) != 1 or snapshot['gameOver']:
        raise ValueError('Captured control has an unsupported terminal or turn state')
    return {'currentPlayer': snapshot['currentPlayer'] - 1, 'turn': turns[0]['value'] + 1, 'terminal': False,
            'physical': sorted(physical, key=lambda row: row['label'])}


def _captured_control_mode(item, mode, index, packet):
    if packet['rawSourceFiles'].get(index['rawPath']) != index['rawSha256']:
        raise ValueError('Captured control byte range has no reviewed raw-file lineage')
    row = source_record(index)
    key = item['sourceKey'] if 'oracleSha256' in index else item['sourceKey'] + '/controls/' + mode
    if (row['key'] != key or row['sourceID'] != item['sourceID'] or row['channel'] != 'useActions'
            or row['actionIndex'] != item['actionIndex'] or row['mode'] != mode
            or row['status'] != 'executed' or not row['legal'] or not row['completed']
            or row['choices'] or row['coins'] or row.get('error')):
        raise ValueError('Captured control lacks the exact positive prompt-free request')
    if 'oracleSha256' in index:
        if (mode != 'heads' or index != item['cachedSourceRecord']
                or index['oracleSha256'] != 'e26941d466984bde749e23f2d41e1460f196a1ff98aa64a3cfb59bf511b303ff'
                or index['fixtureContract'] != 'rich-board-v1-prize-history-unset'):
            raise ValueError('Cached ordinary control differs from the retained source oracle')
    elif (row['requestedActionGuid'] != item['actionGuid']
          or row['requestedActor'] != row['before']['board']['p1Active']['entityID']):
        raise ValueError('Captured control action or actor differs from actual source execution')
    return {stage: normalize_control_source(row[stage], row['before'], item['sourceID'], item['nativeCardId'])
            for stage in ('before', 'after')}


def _captured_controls(item, packet):
    if item['modes'] != CONTROL_MODES.get(item['family']) or set(item['sourceRecordsByMode']) != set(item['modes']):
        raise ValueError('Control fixture matrix differs from its qualified family')
    states = {mode: _captured_control_mode(item, mode, item['sourceRecordsByMode'][mode], packet)
              for mode in item['modes']}
    if states != item['expectedByMode']:
        raise ValueError('Control expectations do not derive from actual captured source bytes')
    return states


def _damage_vector(states):
    before = {row['label']: row['damage'] for row in states['before']['physical']}
    return tuple((row['label'], row['damage'] - before[row['label']]) for row in states['after']['physical']
                 if row['damage'] != before[row['label']])


def _control_qualification(item, action, states):
    effects = action.get('subActions', [])
    if (not effects or effects[-1].get('$type') != 'MatchLogic.ResolveAttack, MatchLogic'
            or any(effect.get('$type') != 'MatchLogic.ApplyDamage, MatchLogic' for effect in effects[:-1])):
        raise ValueError('Controlled source action contains non-damage operations')
    vectors = {mode: _damage_vector(observation) for mode, observation in states.items()}
    changed = {label for vector in vectors.values() for label, amount in vector if amount}
    if len(changed) < len(effects) - 1:
        return 'source-damage-component-remains-unexercised'
    if item['family'] == 'ignore':
        for effect in effects[:-1]:
            value = effect.get('damageValue', {})
            number = value.get('value1', {})
            if (set(value) != {'$type', 'value1'} or value.get('$type') != 'MatchLogic.SingleVariableTotal, MatchLogic'
                    or set(number) != {'$type', 'explicitValue'}
                    or number.get('$type') != 'MatchLogic.ExplicitNumberVariable, MatchLogic'
                    or type(number.get('explicitValue')) is not int):
                return 'ignore-flag-hybrid-variable-requires-its-own-discriminator'
    if item['family'] == 'variable' and len(set(vectors.values())) < 2:
        return 'source-variable-expression-remains-unexercised'
    return None


def _read(path, expected):
    return json.loads(read_pinned(path, expected))


@lru_cache(maxsize=3)
def _packet_pin(packet_path):
    pin = sha256(Path(packet_path).read_bytes()).hexdigest()
    if pin not in DOMAINS:
        raise ValueError('Unreviewed Standard physical-effect domain')
    return pin


def _domain(packet_path):
    return DOMAINS[_packet_pin(packet_path)]


@lru_cache(maxsize=3)
def _packet(packet_path):
    packet = _read(packet_path, _packet_pin(packet_path))
    schema = 'standard-physical-effect-control-packet/2' if _domain(packet_path)['kind'] == 'controls' else 'standard-physical-effect-packet/1'
    if packet['schema'] != schema:
        raise ValueError('Unreviewed Standard physical-effect packet')
    return packet


@lru_cache(maxsize=3)
def _measurement(packet_path):
    reference = _packet(packet_path)['measurementReceipt']
    return _read(reference['path'], reference['sha256'])


@lru_cache(maxsize=3)
def _ready(packet_path):
    measurement = _measurement(packet_path)
    reference = _packet(packet_path)['measurementReceipt']
    if _domain(packet_path)['kind'] == 'controls':
        ready = _packet(packet_path)['qualifiedReady']
        rows = _read(ready['path'], ready['sha256'])
    else:
        rows = _read(Path(reference['path']).parent / 'ready.json', measurement['outputs']['ready.json'])
    count = _domain(packet_path)['ready']
    if len(rows) != count or len({row['caseId'] for row in rows}) != count:
        raise ValueError('Standard physical-effect domain lost or duplicated measured cases')
    return {row['caseId']: row for row in rows}


def _command(item):
    command = {'operation': 'executeCapturedPhysicalEffect', 'cardSourceID': item['sourceID'],
            **{key: item[key] for key in ('programSha256', 'sourceRowSha256', 'actionIndex', 'actionGuid',
                                         'actionSha256', 'nativeCardId', 'attackId', 'nativeAttackOrdinal')}}
    for key in ('damage', 'damageOperations', 'visibleDamagedLabels'):
        if key in item:
            command[key] = item[key]
    return command


def _scenario(item, domain):
    if domain['kind'] == 'controls':
        return {'suite': SUITE, 'fixtureContract': 'rich-board-physical-controls-v1-prize-history-unset', 'facet': 'effect',
                'commands': [{**_command(item), 'family': item['family'], 'mode': mode} for mode in item['modes']],
                'limitations': domain['limitations']}
    return {'suite': SUITE, 'fixtureContract': 'rich-board-v1-prize-history-unset', 'facet': 'effect',
            'commands': [_command(item)], 'limitations': domain['limitations']}


def standard_effect_configuration(packet_path, replay_receipt=None):
    packet_path = Path(packet_path).resolve(strict=True)
    measurement = _measurement(packet_path)
    configuration = {'packetPath': str(packet_path), 'packetSha256': _packet_pin(packet_path),
                     'fixturePath': measurement['nativeExecutable']['path'],
                     'fixtureSha256': measurement['nativeExecutable']['sha256']}
    if replay_receipt is not None:
        replay_receipt = Path(replay_receipt).resolve(strict=True)
        replay = json.loads(replay_receipt.read_bytes())
        if replay['packet'] != {'path': str(packet_path), 'sha256': _packet_pin(packet_path)}:
            raise ValueError('Candidate replay belongs to a different immutable source packet')
        configuration.update(fixturePath=replay['fixture']['path'], fixtureSha256=replay['fixture']['sha256'],
                             replayReceiptPath=str(replay_receipt),
                             replayReceiptSha256=sha256(replay_receipt.read_bytes()).hexdigest())
    return configuration


def standard_effect_cases(packet_path):
    packet_path = Path(packet_path).resolve(strict=True)
    return [{'caseId': item['caseId'], 'cabtCardId': item['nativeCardId'], 'scenario': _scenario(item, _domain(packet_path))}
            for item in _ready(packet_path).values()]


def standard_effect_native_requests(packet_path):
    packet_path = Path(packet_path).resolve(strict=True)
    cases = standard_effect_cases(packet_path)
    return sum(len(case['scenario']['commands']) for case in cases) + (4 if _domain(packet_path)['kind'] == 'controls' else 0)


def _captured(item):
    index = item['sourceRecord']
    if (index['status'] != 'executed' or not index['legal'] or not index['completed']
            or index['fixtureContract'] != 'rich-board-v1-prize-history-unset'):
        raise ValueError('Source effect lacks a positive completed actual oracle')
    row = source_record(index)
    if (row['key'] != item['key'] or row['sourceID'] != item['sourceID'] or row['channel'] != 'useActions'
            or row['actionIndex'] != item['actionIndex'] or row['mode'] != 'heads'
            or row['status'] != 'executed' or not row['legal'] or not row['completed']
            or row['choices'] or row['coins']):
        raise ValueError('Captured physical-effect request, completion, prompts or coins changed')
    card = {'cardSourceID': item['sourceID']}
    states = {stage: normalize_source(row[stage], row['before'], card, item['nativeCardId'])
              for stage in ('before', 'after')}
    if states != item['expected']:
        raise ValueError('Physical-effect expected states no longer derive from actual captured bytes')
    if 'damageOperations' in item:
        before, after = states['before']['physical'], states['after']['physical']
        stable_before = [{key: ('hand' if key == 'zone' and row['label'] == 'p1-deck0' else value)
                          for key, value in row.items() if key != 'damage'} for row in before]
        stable_after = [{key: value for key, value in row.items() if key != 'damage'} for row in after]
        changed = [one['label'] for one, two in zip(before, after) if one['damage'] != two['damage']]
        if (stable_before != stable_after or changed != item['visibleDamagedLabels']
                or len(changed) < len(item['damageOperations'])):
            raise ValueError('Physical damage sequence contains an unobserved operation or another state change')
    return states


def standard_effect_expected(packet_path, case_id):
    packet_path = Path(packet_path).resolve(strict=True)
    item = _ready(packet_path)[case_id]
    if _domain(packet_path)['kind'] == 'controls':
        states = _captured_controls(item, _packet(packet_path))
        return {'commands': _scenario(item, _domain(packet_path))['commands'],
                'responses': [{**states[mode], 'coins': 0, 'choices': []} for mode in item['modes']],
                'limitations': CONTROL_LIMITATIONS}
    return {'commands': [_command(item)], 'responses': [{**_captured(item), 'coins': 0, 'choices': []}],
            'limitations': _domain(packet_path)['limitations']}


def standard_effect_provenance(packet_path):
    packet_path = Path(packet_path).resolve(strict=True)
    packet, measurement = _packet(packet_path), _measurement(packet_path)
    runtime = packet['capturedRuntime']
    return {key: runtime[key] for key in ('corpusSha256', 'librarySha256', 'workerSha256')} | {
        'packetPath': str(packet_path), 'packetSha256': _packet_pin(packet_path),
        'measurementReceipt': packet['measurementReceipt'], 'sourceRecaptures': 0,
        'nativeLibrarySha256': measurement['library']['sha256'], 'finiteDomain': _domain(packet_path)['limitations']}


def _execute(command):
    flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 0
    return subprocess.run(command, capture_output=True, check=True, timeout=180, creationflags=flags).stdout


def run_standard_effects(cases, artifacts, candidate, cards, source_rows):
    if 'standardEffectsBatches' not in artifacts:
        return _run_packet(cases, artifacts, candidate, cards, source_rows)
    batches = artifacts['standardEffectsBatches']
    all_ids = set()
    grouped = []
    for batch in batches:
        ids = set(_ready(Path(batch['standardEffectsPacket']).resolve(strict=True)))
        if all_ids & ids:
            raise ValueError('Effect packets overlap measured source requirements')
        all_ids.update(ids)
        grouped.append((batch, [case for case in cases if case['caseId'] in ids]))
    if any(case['caseId'] not in all_ids for case in cases):
        raise ValueError('Effect case has no reviewed packet')
    observations = []
    for batch, selected in grouped:
        if selected:
            observations.extend(_run_packet(selected, {**artifacts, **batch}, candidate, cards, source_rows))
    return observations


def _control_action(item, source, qualified, ordered, cards):
    card, row_hash, program = source[item['sourceID']]
    action = card['useActions'][item['actionIndex']]
    ordinal = sum(previous['actionType'] == 1 for previous in card['useActions'][:item['actionIndex']])
    if (item['caseId'] != f'program/{program}/useActions/{item["actionIndex"]}/effect'
            or item['sourceKey'] != f'source/{program}/useActions/{item["actionIndex"]}'
            or row_hash != item['sourceRowSha256'] or program != item['programSha256']
            or digest(action) != item['actionSha256'] or action['actionGuid'] != item['actionGuid']
            or action['actionType'] != 1 or qualified.get(item['sourceID']) != item['nativeCardId']
            or ordinal != item['nativeAttackOrdinal'] or ordered[item['nativeCardId']][ordinal] != item['attackId']
            or cards[item['nativeCardId']]['attacks'][ordinal] != item['attackId']):
        raise ValueError('Controlled source row, program, action or ordered native attack changed')
    return action


def _run_controls(cases, packet_path, packet, measurement, root, fixture, cards, source, qualified, ordered, base_ids):
    for field in ('qualificationReceipt', 'parentPacket', 'capturedRuntimeReceipt', 'sourceFixtureDelta'):
        reference = packet[field]
        read_pinned(reference['path'], reference['sha256'])
    for field in ('rawSourceFiles', 'normalizerSources'):
        for path, pin in packet[field].items():
            read_pinned(path, pin)
    qualification_ref = packet['qualificationReceipt']
    qualification = _read(qualification_ref['path'], qualification_ref['sha256'])
    read_pinned(qualification['driver']['path'], qualification['driver']['sha256'])
    if (qualification['schema'] != 'standard-physical-effect-control-qualification/1'
            or qualification['parentPacket'] != packet['parentPacket']
            or qualification['measuredReady'] != 53 or qualification['qualifiedReady'] != 49
            or qualification['heldUnexercised'] != 4 or qualification['sourceRecaptures'] != 0
            or qualification['nativeReruns'] != 0 or packet['caseCount'] != 49 or packet['measuredReady'] != 53
            or packet['limitations'] != CONTROL_LIMITATIONS or not packet['actualDllCatalogEqual']
            or not packet['sourceAndObjectRetained']):
        raise ValueError('Controlled effect qualification no longer describes the measured finite domain')
    for name, pin in qualification['outputs'].items():
        read_pinned(Path(qualification_ref['path']).parent / name, pin)
    measured = _read(root / 'ready.json', measurement['outputs']['ready.json'])
    held_ref = packet['heldUnexercised']
    held = _read(held_ref['path'], held_ref['sha256'])
    ready = _ready(packet_path)
    classified = {}
    captured = {}
    for item in measured:
        if item['caseId'] in classified:
            raise ValueError('Controlled effect measurement duplicated a source action')
        action = _control_action(item, source, qualified, ordered, cards)
        states = _captured_controls(item, packet)
        classified[item['caseId']] = _control_qualification(item, action, states)
        captured[item['caseId']] = states
        if item['caseId'] in ready and item != ready[item['caseId']]:
            raise ValueError('Qualified effect differs from its actual measured source item')
    qualified_ids = {case_id for case_id, reason in classified.items() if reason is None}
    held_reasons = {case_id: reason for case_id, reason in classified.items() if reason is not None}
    if (len(measured) != 53 or qualified_ids != set(ready) or len(held) != 4
            or {item['caseId']: item['reason'] for item in held} != held_reasons
            or sum(item['family'] == 'variable' for item in ready.values()) != 36
            or sum(item['family'] == 'ignore' for item in ready.values()) != 13):
        raise ValueError('Actual source control witnesses no longer support the qualified49/held4 partition')
    selected = {}
    for case in cases:
        item = ready.get(case['caseId'])
        if (item is None or case['caseId'] in selected or case['caseId'] in base_ids
                or case['cabtCardId'] != item['nativeCardId'] or case['scenario'] != _scenario(item, DOMAINS[CONTROL_PACKET_SHA256])):
            raise ValueError('Controlled effect differs from its qualified unobserved source requirement')
        selected[case['caseId']] = item
    prepared = _read(root / 'prepared.json', measurement['outputs']['prepared.json'])
    calibrators = [item for item in prepared if item['family'] == 'calibration']
    if len(calibrators) != 1:
        raise ValueError('Control fixture lost its unique calibration action')
    calibration = calibrators[0]
    action = _control_action(calibration, source, qualified, ordered, cards)
    if plain_damage(action) != 70 or calibration['modes'] != CONTROL_MODES['ignore']:
        raise ValueError('Control calibration is not the retained ordinary70damage action')
    indexes = _read(root / 'source-records.json', measurement['outputs']['source-records.json'])
    source_calibration = {mode: _captured_control_mode(calibration, mode, indexes[calibration['sourceKey'] + '/controls/' + mode], packet)
                          for mode in calibration['modes']}
    if {mode: _damage_vector(states) for mode, states in source_calibration.items()} != {
            mode: (('p1-active', amount),) for mode, amount in CALIBRATION.items()}:
        raise ValueError('Actual source control calibration does not exercise weakness, resistance and SolidShell')
    requests = {}
    for item in [*selected.values(), calibration]:
        for mode in item['modes']:
            key = item['sourceKey'] + '/controls/' + mode
            if key in requests:
                raise ValueError('Controlled native requests overlap calibration or another action')
            requests[key] = item, mode
    with tempfile.TemporaryDirectory(prefix='cabt-physical-controls-') as temporary:
        request_path = Path(temporary) / 'requests.tsv'
        request_path.write_text(''.join(f'{key}\t{item["nativeCardId"]}\tattack\t{item["attackId"]}\tattack\t{mode}\n'
                                       for key, (item, mode) in requests.items()), encoding='utf-8')
        rows = [json.loads(line) for line in _execute([str(fixture), str(request_path)]).splitlines()]
    actual = {row['key']: row for row in rows}
    if len(actual) != len(rows) or set(actual) != set(requests):
        raise ValueError('Controlled native replay omitted or duplicated a requested fixture mode')
    normalized = {}
    for key, (item, mode) in requests.items():
        row = actual[key]
        actors = [physical for physical in row['before']['physical'] if physical['label'] == 'actor']
        prompt = row['initialPrompt']
        if (row.get('error') or not row['plainNativeAttack'] or not row['legal'] or not row['completed']
                or row['choices'] or row['coins'] or row['requestedNativeCardId'] != item['nativeCardId']
                or row['requestedNativeAttackId'] != item['attackId'] or len(actors) != 1
                or row['requestedActorSerial'] != actors[0]['serial'] or prompt['type'] != 0 or prompt['context'] != 0
                or not any(option.get('type') == 13 and option.get('attackId') == item['attackId'] for option in prompt['option'])):
            raise ValueError('Controlled native replay lost its positive physical actor/action contract')
        states = {stage: normalize_native(row[stage]) for stage in ('before', 'after')}
        expected = source_calibration[mode] if item is calibration else captured[item['caseId']][mode]
        if states != expected:
            raise ValueError(f'Fresh controlled physical observation differs from actual source: {item["sourceID"]}/{mode}')
        normalized[key] = states
    observations = []
    for case in cases:
        item = selected[case['caseId']]
        observation = {'commands': case['scenario']['commands'],
                       'responses': [{**normalized[item['sourceKey'] + '/controls/' + mode], 'coins': 0, 'choices': []}
                                     for mode in item['modes']], 'limitations': CONTROL_LIMITATIONS}
        observations.append({'caseId': case['caseId'], 'cabtCardId': case['cabtCardId'],
                             'scenarioSha256': digest(case['scenario']), 'observation': observation,
                             'scenarioComponents': [{'suite': SUITE, 'scenarioSha256': digest(case['scenario'])}]})
    return observations


def _candidate_replay(packet_path, packet, artifacts, candidate):
    replay = json.loads(Path(artifacts['standardEffectsReplayReceipt']).read_bytes())
    if (replay['schema'] != 'standard-effect-candidate-replay/1'
            or replay['packet'] != {'path': str(packet_path), 'sha256': _packet_pin(packet_path)}
            or replay['measurementReceipt'] != packet['measurementReceipt']
            or replay['sourceRoot'] != candidate['sourceRoot']
            or replay['sourceCommit'] != candidate['sourceCommit']
            or replay['sourceSha256'] != candidate['sourceSha256']
            or replay['library'] != {'path': str(artifacts['library']), 'sha256': candidate['librarySha256']}
            or replay['sourceRecaptures'] != 0):
        raise ValueError('Candidate effect replay differs from its exact source, library or retained packet')
    for relative, pin in candidate['sourceSha256'].items():
        read_pinned(Path(candidate['sourceRoot']) / relative, pin)
    registry_ref = replay['registryReceipt']
    registry = _read(registry_ref['path'], registry_ref['sha256'])
    execution_ref = replay['buildExecutionReceipt']
    execution = _read(execution_ref['path'], execution_ref['sha256'])
    if (registry['schema'] != 'cabt-native-registry-object/1'
            or registry['sourceRoot'] != candidate['sourceRoot']
            or registry['sourceCommit'] != candidate['sourceCommit']
            or registry['sourceSha256'] != candidate['sourceSha256']
            or registry['objectSha256'] != replay['registryObjectSha256']
            or execution['schema'] != 'cabt-registry-object-execution/1'
            or execution['registryReceiptPath'] != registry_ref['path']
            or execution['registryReceiptSha256'] != registry_ref['sha256']
            or execution['sourceCommit'] != candidate['sourceCommit']
            or execution['objectSha256'] != registry['objectSha256']
            or execution['exitCode'] != 0 or not execution['sourceAndObjectRetained']
            or execution['scopedEnvironment']['CL'] != '/DCABT_SEPARATE_REGISTRY'
            or execution['scopedEnvironment']['LINK'] != '"' + registry['objectPath'] + '"'):
        raise ValueError('Candidate effect fixture was not compiled against its exact frozen registry object')
    read_pinned(registry['objectPath'], registry['objectSha256'])
    for name in ('msbuild', 'compiler'):
        read_pinned(registry['toolchain'][name + 'Path'], registry['toolchain'][name + 'Sha256'])
    read_pinned(replay['driver']['path'], replay['driver']['sha256'])
    original_root = Path(packet['measurementReceipt']['path']).parent
    for name, field in (('native.cpp', 'fixtureSource'), ('native.vcxproj', 'project')):
        original = original_root / name
        pin = packet['nativeFixtureSources'].get(str(original))
        reference = replay[field]
        if pin is None or reference['sha256'] != pin:
            raise ValueError('Candidate replay altered the previously qualified native fixture source')
        read_pinned(original, pin)
        read_pinned(reference['path'], pin)
    fixture = Path(replay['fixture']['path']).resolve(strict=True)
    project = Path(replay['project']['path']).resolve(strict=True)
    fixture_source = Path(replay['fixtureSource']['path']).resolve(strict=True)
    output = project.parent
    if any((Path(path).parent / 'All.h').exists() for path in packet['nativeFixtureSources']) or (output / 'All.h').exists():
        raise ValueError('Native fixture directory shadows the frozen candidate engine headers')
    command = [registry['toolchain']['msbuildPath'], str(project), '/nologo', '/v:normal',
               '/p:Configuration=Release', '/p:Platform=x64', '/p:EngineSource=' + candidate['sourceRoot'],
               '/p:OutDir=' + str(output / 'bin') + '/', '/p:IntDir=' + str(output / 'obj') + '/']
    if (execution['command'] != command or fixture_source != output / 'native.cpp'
            or fixture != output / 'bin/native.exe'
            or Path(artifacts['standardEffectsFixture']).resolve(strict=True) != fixture):
        raise ValueError('Candidate replay build command or fixture paths changed')
    read_pinned(fixture, replay['fixture']['sha256'])
    read_pinned(replay['buildLog']['path'], replay['buildLog']['sha256'])
    read_pinned(artifacts['library'], candidate['librarySha256'])
    return fixture


def _run_packet(cases, artifacts, candidate, cards, source_rows):
    packet_path = Path(artifacts['standardEffectsPacket']).resolve(strict=True)
    packet, measurement = _packet(packet_path), _measurement(packet_path)
    root = Path(packet['measurementReceipt']['path']).parent
    runtime = packet['capturedRuntime']
    domain = _domain(packet_path)
    controlled = domain['kind'] == 'controls'
    counts_valid = (measurement.get('candidateActions') == 119 and measurement.get('newSourceRequests') == 462
                    and measurement.get('newNativeRequests') == 581 and measurement.get('ready') == 53
                    and measurement.get('differences') == 5 and measurement.get('heldControls') == 61
                    and measurement.get('heldPreparation') == 6 and measurement.get('calibrationQualified') is True
                    and measurement.get('calibration') == CALIBRATION) if controlled else (
                        all(measurement[key] == domain[key] for key in ('ready', 'prepared', 'differences', 'heldNative', 'heldSource'))
                        and measurement['sourceActions'] == 3388 and measurement['sourceRecaptures'] == 0
                        and measurement['actualDllCatalogEqual'] and measurement['sourceAndObjectRetained']
                        and measurement['limitations'] == domain['limitations'])
    rebound = 'standardEffectsReplayReceipt' in artifacts
    if (measurement['schema'] != domain['schema']
            or (not rebound and (measurement['sourceCommit'] != candidate['sourceCommit']
                                 or measurement['sourceSha256'] != candidate['sourceSha256']
                                 or measurement['library']['sha256'] != candidate['librarySha256']))
            or not counts_valid or measurement['facet'] != 'effect'
            or measurement['otherFacetsAdded'] != 0
            or runtime['corpusSha256'] != LIVE_PINS.corpus_sha256):
        raise ValueError('Physical-effect packet does not bind this exact candidate and measured finite domain')
    for path, pin in runtime['files'].items():
        read_pinned(path, pin)
    for path, pin in packet['nativeFixtureSources'].items():
        read_pinned(path, pin)
    oracle_references = [packet['sourceOracle']] if controlled else packet['sourceOracle'].values()
    for reference in oracle_references:
        if isinstance(reference, dict):
            read_pinned(reference['path'], reference['sha256'])
    for path, pin in measurement['inputPins'].items():
        read_pinned(path, pin)
    for name, pin in measurement['outputs'].items():
        read_pinned(root / name, pin)
    for relative, pin in measurement['sourceSha256'].items():
        read_pinned(Path(measurement['sourceRoot']) / relative, pin)
    registry_ref = measurement['registryReceipt']
    registry = _read(registry_ref['path'], registry_ref['sha256'])
    execution = _read(root / 'registry-execution.json', measurement['outputs']['registry-execution.json'])
    if (registry['sourceCommit'] != measurement['sourceCommit'] or registry['sourceSha256'] != measurement['sourceSha256']
            or registry['objectSha256'] != measurement['registryObjectSha256']
            or execution['objectSha256'] != registry['objectSha256']
            or execution['registryReceiptSha256'] != registry_ref['sha256']
            or execution['sourceCommit'] != measurement['sourceCommit']
            or not execution['sourceAndObjectRetained'] or execution['exitCode']):
        raise ValueError('Physical-effect fixture object does not bind the exact candidate source')
    read_pinned(registry['objectPath'], registry['objectSha256'])
    fixture = Path(measurement['nativeExecutable']['path']).resolve(strict=True)
    read_pinned(fixture, measurement['nativeExecutable']['sha256'])
    read_pinned(measurement['library']['path'], measurement['library']['sha256'])
    if rebound:
        fixture = _candidate_replay(packet_path, packet, artifacts, candidate)
    elif (fixture != Path(artifacts['standardEffectsFixture']).resolve(strict=True)
          or Path(artifacts['library']).resolve(strict=True) != Path(measurement['library']['path']).resolve(strict=True)):
        raise ValueError('Physical-effect fixture or library differs from its measured current candidate')
    library = ctypes.CDLL(str(artifacts['library']))
    catalog = {}
    for export, field in (('AllCard', 'entries'), ('AllAttack', 'attacks')):
        function = getattr(library, export)
        function.argtypes, function.restype = [], ctypes.c_char_p
        catalog[field] = json.loads(function())
    if {row['cardId']: row for row in catalog['entries']} != cards or json.loads(_execute([str(fixture), 'registry'])) != catalog:
        raise ValueError('Fresh physical-effect fixture catalog differs from actual current DLL')
    metadata_ref = measurement['metadataSelection']
    metadata = _read(metadata_ref['path'], metadata_ref['sha256'])
    qualified = {row['cardSourceID']: row['qualifiedNativeId'] for row in metadata['rows'] if row.get('qualifiedNativeId') is not None}
    ordered = {row['cardId']: row['printedAttackIds'] for row in metadata['nativeCardActions']}
    source = {card['cardSourceID']: (card, row_hash, program) for card, _, row_hash, program in source_rows}
    base_ref = measurement['baseManifest']
    base_ids = {case['caseId'] for case in _read(base_ref['path'], base_ref['sha256'])['cases']}
    if controlled:
        return _run_controls(cases, packet_path, packet, measurement, root, fixture, cards,
                             source, qualified, ordered, base_ids)
    prepared = _ready(packet_path)
    selected = {}
    for case in cases:
        item = prepared.get(case['caseId'])
        if (item is None or case['caseId'] in selected or case['caseId'] in base_ids
                or case['cabtCardId'] != item['nativeCardId'] or case['scenario'] != _scenario(item, domain)):
            raise ValueError('Physical-effect case differs from the measured unobserved source action')
        card, row_hash, program = source[item['sourceID']]
        action = card['useActions'][item['actionIndex']]
        ordinal = sum(previous['actionType'] == 1 for previous in card['useActions'][:item['actionIndex']])
        source_damage = plain_damage(action) if domain['kind'] == 'single' else damage_sequence(action)
        expected_damage = item['damage'] if domain['kind'] == 'single' else item['damageOperations']
        if (item['caseId'] != f'program/{program}/useActions/{item["actionIndex"]}/effect'
                or item['key'] != f'source/{program}/useActions/{item["actionIndex"]}'
                or row_hash != item['sourceRowSha256'] or program != item['programSha256']
                or digest(action) != item['actionSha256'] or action['actionGuid'] != item['actionGuid']
                or source_damage is None or source_damage != expected_damage or qualified.get(item['sourceID']) != item['nativeCardId']
                or ordinal != item['nativeAttackOrdinal'] or ordered[item['nativeCardId']][ordinal] != item['attackId']
                or cards[item['nativeCardId']]['attacks'][ordinal] != item['attackId']):
            raise ValueError('Physical-effect raw source, program, action, measured identity or ordered attack changed')
        raw_index = item['sourceRecord']
        if (packet['rawSourceFiles'].get(raw_index['rawPath']) != raw_index['rawSha256']
                or raw_index['oracleSha256'] != packet['sourceOracle']['compiledDll']['sha256']):
            raise ValueError('Captured physical-effect byte range lacks the reviewed raw-file and oracle lineage')
        _captured(item)
        selected[case['caseId']] = item
    with tempfile.TemporaryDirectory(prefix='cabt-physical-effect-') as temporary:
        request_path = Path(temporary) / 'requests.tsv'
        request_path.write_text(''.join(f'{item["key"]}\t{item["nativeCardId"]}\tattack\t{item["attackId"]}\tattack\theads\n'
                                        for item in selected.values()), encoding='utf-8')
        raw = _execute([str(fixture), str(request_path)])
    rows = [json.loads(line) for line in raw.splitlines()]
    actual = {row['key']: row for row in rows}
    if len(rows) != len(actual) or set(actual) != {item['key'] for item in selected.values()}:
        raise ValueError('Fresh physical-effect replay omitted or duplicated an actual requested action')
    observations = []
    for case in cases:
        item = selected[case['caseId']]
        row = actual[item['key']]
        actors = [card for card in row['before']['physical'] if card['label'] == 'actor']
        prompt = row['initialPrompt']
        available = any(option.get('type') == 13 and option.get('attackId') == item['attackId'] for option in prompt['option'])
        if (row.get('error') or not row['plainNativeAttack']
                or (domain['kind'] == 'single' and row['nativeDamage'] != item['damage'])
                or not row['legal'] or not row['completed'] or row['choices'] or row['coins']
                or row['requestedNativeCardId'] != item['nativeCardId'] or row['requestedNativeAttackId'] != item['attackId']
                or len(actors) != 1 or row['requestedActorSerial'] != actors[0]['serial']
                or prompt['type'] != 0 or prompt['context'] != 0 or not available):
            raise ValueError('Actual native physical effect no longer meets its positive pure-damage actor/action contract')
        normalized = {stage: normalize_native(row[stage]) for stage in ('before', 'after')}
        observation = {'commands': [_command(item)], 'responses': [{**normalized, 'coins': 0, 'choices': []}],
                       'limitations': domain['limitations']}
        if observation != standard_effect_expected(packet_path, case['caseId']):
            raise ValueError('Fresh physical effect differs from its actual captured source observation')
        observations.append({'caseId': case['caseId'], 'cabtCardId': case['cabtCardId'],
                             'scenarioSha256': digest(case['scenario']), 'observation': observation,
                             'scenarioComponents': [{'suite': SUITE, 'scenarioSha256': digest(case['scenario'])}]})
    return observations
