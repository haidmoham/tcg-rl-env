import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { assertStandardAdmission, projectRoot } from '../integrations/twinleaf/public-engine.mjs';
import { runTwinleafConformance } from '../integrations/twinleaf/conformance.mjs';

test('Twinleaf refuses unverified full Standard admission', () => {
  assert.throws(assertStandardAdmission, /UNVERIFIED_STANDARD_COVERAGE/);
});

const prepared = existsSync(resolve(projectRoot, '.cache/twinleaf/ptcg-server/src/game/store/store.ts'))
  && existsSync(resolve(projectRoot, '.cache/twinleaf-tooling/node_modules/typescript'));
test('pinned public Twinleaf executes modern printed-card transitions', { skip: !prepared }, () => {
  const receipt = runTwinleafConformance();
  assert.equal(receipt.count, 7);
  assert.equal(receipt.completeStandard, false);
});
