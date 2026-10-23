import test from 'node:test';
import assert from 'node:assert/strict';
import { enumerateChoiceSets, chooseValidatedChoiceSet } from '../learning/legal-choice-sets.mjs';

const view = (n = 3, min = 0, max = 2) => ({ status: 'playing', hand: [], you: { active: null, bench: [], discard: [] },
  opponent: { active: null, bench: [], discard: [] }, prompts: [{ id: 'p', kind: 'cards', min, max, ordered: false,
    cancelable: false, groups: [], choices: Array.from({ length: n }, (_, i) => ({ id: `c${i}`, card: { instanceId: `v${i}` } })) }] });

test('all optional cardinalities are retained, or the complete domain is rejected before sampling', () => {
  assert.deepEqual(enumerateChoiceSets(view()).sets, [[], [0], [1], [2], [0, 1], [0, 2], [1, 2]]);
  assert.equal(enumerateChoiceSets(view(7, 0, 7)).sets.length, 128);
  assert.deepEqual(enumerateChoiceSets(view(8, 0, 8)), { supported: false, reason: 'complete-choice-set-domain-exceeds-128' });
});

test('every set is native-validated and the selected index maps to its whole semantic answer', async () => {
  const checked = [];
  const result = await chooseValidatedChoiceSet(view(), async decision => {
    checked.push(decision.choiceIds); return decision.choiceIds.length !== 1;
  }, { async act([encoded]) {
    assert.deepEqual(encoded.legalMask, [true, false, false, false, true, true, true]);
    return [{ actionIndex: 6 }];
  } }, {}, true, () => ({ candidates: [0, 1, 2] }));
  assert.equal(checked.length, 7);
  assert.deepEqual(result.decision, { promptId: 'p', choiceIds: ['c1', 'c2'] });
});
