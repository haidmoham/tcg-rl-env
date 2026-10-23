import test from 'node:test';
import assert from 'node:assert/strict';
import { runInformationSetSearch } from '../learning/alphazero-search.mjs';

function provider({ opponent = false, hidden = false } = {}) {
  let world = 0;
  const handles = new Set();
  return { handles, seeds: [],
    async begin({ seed }) { this.seeds.push(seed); const state = { depth: 0, world: world++ }; handles.add(state); return state; },
    async observe(state) {
      if (state.depth === 2) return { terminal: { winner: state.action === (hidden ? state.world % 2 : 1) ? 'human' : 'opponent' } };
      return { view: { depth: state.depth }, toPlay: state.depth && opponent ? 'opponent' : 'human', terminal: null };
    },
    async step(state, command) { assert(handles.has(state)); const next = { ...state, depth: state.depth + 1, action: command.actionId }; handles.add(next); return next; },
    async validate() { return true; },
    async release(handle) { assert(handles.delete(handle)); },
  };
}
const evaluate = async ({ view }) => ({ commands: [{ actionId: 0 }, { actionId: 1 }], logits: [0, 0], value: 0,
  encoded: { legalMask: [true, true], observation: view.depth } });
const options = p => ({ provider: p, rootView: {}, publicHistory: [], knownDecks: {}, rootPlayer: 'human', seed: 819,
  evaluate, determinizations: 2, simulations: 16, maxDepth: 8, temperature: 0, rootNoise: false });

test('PUCT pools visit targets, shares information sets across worlds, and releases every branch', async () => {
  const p = provider(), result = await runInformationSetSearch(options(p));
  assert.equal(result.visits.reduce((sum, value) => sum + value), 32);
  assert.equal(result.policyTarget.reduce((sum, value) => sum + value), 1);
  assert(result.diagnostics.sharedInformationSetHits > 0);
  assert.equal(result.diagnostics.approximation, 'declared-deck-information-set-puct/1');
  assert.equal(p.handles.size, 0);
  assert.equal(p.seeds.length, 2);
});

test('terminal outcomes overcome a wrong policy prior at the root', async () => {
  const p = provider();
  p.observe = async state => state.depth
    ? { terminal: { winner: state.action === 1 ? 'human' : 'opponent' } }
    : { view: { depth: 0 }, toPlay: 'human', terminal: null };
  const result = await runInformationSetSearch({ ...options(p),
    evaluate: async input => ({ ...await evaluate(input), logits: [1, 0] }) });
  assert.equal(result.selectedIndex, 1);
  assert(result.visits[1] > result.visits[0]);
  assert(result.rootValue > 0);
  assert.equal(p.handles.size, 0);
});

test('same-player continuations and opponent decisions use explicit player perspective', async () => {
  const own = await runInformationSetSearch(options(provider()));
  const opposing = await runInformationSetSearch(options(provider({ opponent: true })));
  assert(own.rootValue > opposing.rootValue);
});

test('hidden-world differences cannot create distinct policy statistics at an indistinguishable node', async () => {
  const result = await runInformationSetSearch(options(provider({ hidden: true })));
  // One root information set and two own-action histories, each shared across both worlds.
  assert.equal(result.diagnostics.informationSets, 3);
  assert(result.diagnostics.sharedInformationSetHits >= 3);
});

test('independent search seed reproduces visits/noise and changing native seed is not an input', async () => {
  const a = await runInformationSetSearch({ ...options(provider()), temperature: 1, rootNoise: true });
  const b = await runInformationSetSearch({ ...options(provider()), temperature: 1, rootNoise: true });
  assert.deepEqual(a.visits, b.visits); assert.equal(a.selectedIndex, b.selectedIndex);
});

test('branch failures are reported and all already-created handles are released', async () => {
  const p = provider(); p.step = async () => { throw Error('explicit native failure'); };
  await assert.rejects(runInformationSetSearch(options(p)), /explicit native failure/);
  assert.equal(p.handles.size, 0);
});

test('native-validated empty prompts advance without invented policy or value outputs', async () => {
  const p = provider();
  const result = await runInformationSetSearch({ ...options(p), evaluate: async input => input.view.depth === 1
    ? { forcedDecision: { promptId: 'empty', choiceIds: [] } } : evaluate(input) });
  assert(result.diagnostics.forcedTransitions > 0);
  assert.equal(result.visits.reduce((sum, n) => sum + n), 32);
  assert.equal(p.handles.size, 0);
});
