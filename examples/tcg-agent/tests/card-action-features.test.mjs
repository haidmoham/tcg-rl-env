import test from 'node:test';
import assert from 'node:assert/strict';
import { encodeOrdinaryDecision } from '../integrations/local-worker/learning-features.mjs';
import { cardActionFeatureContract, encodeCardActionDecision, encodeCardChoiceDecision } from '../learning/card-action-features.mjs';

function fixture() {
  const creature = { instanceId: 'dragon', printingId: 'CARD-3', definitionId: 'definition-dragon', hp: 320,
    damage: 0, providedEnergy: 2, currentRetreat: 2, printedRetreat: 2, conditions: [],
    publicCards: { energy: [{ instanceId: 'fire', printingId: 'ENERGY-1', category: 'energy' }], tools: [], evolutions: [] } };
  const side = { active: creature, bench: [], prizes: 6, deckCount: 45, handCount: 5, discard: [] };
  return { source: 'live-matchlogic', status: 'playing', revision: 9, turn: 'you', turnNumber: 4,
    hand: [{ instanceId: 'trainer-a', printingId: 'TRAINER-1', category: 'trainer' },
      { instanceId: 'trainer-b', printingId: 'TRAINER-2', category: 'trainer' }], you: side,
    opponent: { ...side, active: { ...creature, instanceId: 'other', printingId: 'CARD-4' } }, stadium: null, prompts: [],
    actions: [{ id: 'a', kind: 'attack', actorId: 'dragon', actionKey: 'definition-dragon/first', attackName: 'Attack A' },
      { id: 'b', kind: 'attack', actorId: 'dragon', actionKey: 'definition-dragon/second', attackName: 'Attack B' },
      { id: 'c', kind: 'play', sourceCardId: 'trainer-a' }, { id: 'd', kind: 'play', sourceCardId: 'trainer-b' }] };
}

test('card/action identity separates indistinguishable old attacks and Trainers', () => {
  const view = fixture(), old = encodeOrdinaryDecision(view), encoded = encodeCardActionDecision(view, { actionId: 'b' });
  assert.deepEqual(old.candidates[0], old.candidates[1]);
  assert.deepEqual(old.candidates[2], old.candidates[3]);
  assert.notDeepEqual(encoded.candidates[0], encoded.candidates[1]);
  assert.notDeepEqual(encoded.candidates[2], encoded.candidates[3]);
  assert.equal(encoded.label, 1);
  assert.equal(encoded.state.length, cardActionFeatureContract.stateDimensions);
  assert.ok(encoded.candidates.every(row => row.length === cardActionFeatureContract.candidateDimensions));
});

test('opaque instance/action aliases and hand ordering do not change features', () => {
  const view = fixture(), changed = structuredClone(view);
  changed.revision = 999; changed.seed = 9999; changed.hiddenOpponentHand = ['private'];
  const rename = value => {
    if (!value || typeof value !== 'object') return;
    for (const [key, item] of Object.entries(value)) {
      if (['instanceId', 'sourceCardId', 'actorId', 'targetId', 'id'].includes(key) && typeof item === 'string') value[key] = `opaque-${item}`;
      else rename(item);
    }
  };
  rename(changed); changed.hand.reverse();
  assert.deepEqual(encodeCardActionDecision(changed), encodeCardActionDecision(view));
});

test('card-valued selection features preserve full choice order and hide choice aliases', () => {
  const view = fixture();
  view.prompts = [{ id: 'prompt', kind: 'cards', choices: [
    { id: 'choice-a', card: { instanceId: 'search-a', printingId: 'CARD-1', category: 'pokemon' } },
    { id: 'choice-b', card: { instanceId: 'search-b', printingId: 'CARD-2', category: 'pokemon' } }], min: 1, max: 2 }];
  const encoded = encodeCardChoiceDecision(view);
  assert.equal(encoded.kind, 'selection');
  assert.equal(encoded.candidates.length, 2);
  assert.notDeepEqual(encoded.candidates[0], encoded.candidates[1]);
  const reversed = structuredClone(view); reversed.prompts[0].choices.reverse();
  const alternate = encodeCardChoiceDecision(reversed);
  assert.deepEqual(encoded.state, alternate.state);
  assert.deepEqual(encoded.candidates, alternate.candidates.toReversed());
  assert.equal(encoded.state[0], 1);
});

test('printing equivalence requires explicit noncyclic aliases and hidden prompts fail', () => {
  const first = fixture(), second = structuredClone(first);
  second.hand[0].printingId = 'REPRINT-1';
  assert.notDeepEqual(encodeCardActionDecision(first), encodeCardActionDecision(second));
  assert.deepEqual(encodeCardActionDecision(first), encodeCardActionDecision(second, null, { printingAliases: { 'REPRINT-1': 'TRAINER-1' } }));
  assert.throws(() => encodeCardActionDecision(first, null, { printingAliases: { A: 'B', B: 'A' } }), /cycle/);
  first.prompts = [{ id: 'hidden', kind: 'prizes', choices: [{ id: 'unknown' }] }];
  assert.throws(() => encodeCardChoiceDecision(first), /noncard or hidden/);
});
