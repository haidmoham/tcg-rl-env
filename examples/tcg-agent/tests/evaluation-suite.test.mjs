import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Environment } from '../src/environment.mjs';
import { policyInput, runSafeEpisode } from '../src/policy-boundary.mjs';
import { createPolicy } from '../src/tactical-policy.mjs';
import { evaluateSuite, validateSuite, pairSummary } from '../src/evaluation-suite.mjs';

const fixture = () => JSON.parse(readFileSync(new URL('../experiments/offline-suite-v2.json', import.meta.url), 'utf8'));
const smallSuite = () => { const value = fixture(); value.matchups = [value.matchups[0]]; value.matchups[0].seedCount = 2; value.bootstrapResamples = 100; return value; };

test('policy DTO drops privileged metadata and rejects engine objects in public scalar slots', () => {
  const env = new Environment({ seed: 7 });
  const raw = env.observe();
  raw.seed = 7; raw.full_state = env.replayRecord(); raw.opponentDeck = ['hidden'];
  raw.players[1 - raw.player].hand = ['secret']; raw.actions[0].hiddenTarget = { secret: true };
  const input = policyInput(raw, [{ type: 'coin', winner: raw.player, hiddenDeck: ['secret'] }]);
  assert.equal(input.observation.seed, undefined);
  assert.equal(input.observation.full_state, undefined);
  assert.equal(input.observation.players[1 - raw.player].hand, undefined);
  assert.equal(input.observation.actions[0].hiddenTarget, undefined);
  assert.equal(input.publicEvents[0].hiddenDeck, undefined);
  assert.throws(() => input.observation.actions.push({}), TypeError);
  assert.throws(() => policyInput({ ...raw, profile: 'Standard' }), /audited/);
  raw.players[raw.player].active = { card: { private: true }, hp: 30, damage: 0, energy: 0 };
  assert.throws(() => policyInput(raw), /strings/);
});

test('identical information and policy RNG produce identical choices in distinct hidden worlds', () => {
  const env = new Environment({ seed: 10 });
  const raw = env.observe(), alternate = structuredClone(raw);
  raw.players[1 - raw.player].hand = ['base1-52'];
  alternate.players[1 - raw.player].hand = ['base1-97'];
  raw.seed = 1; alternate.seed = 2;
  const a = policyInput(raw), b = policyInput(alternate);
  assert.deepEqual(a, b);
  assert.deepEqual(createPolicy('tactical-basic-v1', 123).act(a), createPolicy('tactical-basic-v1', 123).act(b));
});

test('tactical scoring selects a visible KO and remains within offered actions', () => {
  const obs = {
    schema: 'observation-v1', profile: 'basic-duel-v1', player: 0, actor: 0, firstPlayer: 1, phase: 'turn', turn: 3, revision: 11,
    players: [
      { handCount: 1, deckCount: 40, prizeCount: 1, hand: ['base1-52'], active: { card: 'base1-52', hp: 50, damage: 0, energy: 1 }, bench: [], discard: [], mulligans: 0 },
      { handCount: 4, deckCount: 40, prizeCount: 4, active: { card: 'base1-61', hp: 30, damage: 0, energy: 1 }, bench: [], discard: [], mulligans: 0 },
    ],
    actions: [{ kind: 'pass', revision: 11 }, { kind: 'bench', hand: 0, revision: 11 }, { kind: 'attack', revision: 11 }],
  };
  const action = createPolicy('tactical-basic-v1', 2).act(policyInput(obs));
  assert.equal(action.kind, 'attack');
  assert.ok(obs.actions.some(a => JSON.stringify(a) === JSON.stringify(action)));
});

test('safe episodes supply only frozen views and per-player public-event deltas', () => {
  const env = new Environment({ seed: 20 });
  const calls = [0, 0], events = [[], []];
  const policies = [0, 1].map(player => {
    const base = createPolicy('tactical-basic-v1', player + 10);
    return { act(input) {
      calls[player]++;
      assert.equal(input.observation.player, player);
      assert.equal(input.environment, undefined);
      assert.ok(Object.isFrozen(input.observation.players));
      events[player].push(...input.publicEvents);
      return base.act(input);
    } };
  });
  const outcome = runSafeEpisode(env, policies);
  assert.equal(outcome.terminated, true);
  assert.ok(calls.every(n => n > 0));
  assert.ok(events.every(rows => rows.filter(event => event.type === 'coin').length === 1));
});

test('suite pins exact deck and policy versions, swaps seats, and reproduces outcomes', () => {
  const report = evaluateSuite(smallSuite());
  assert.equal(report.games.length, 8);
  assert.equal(report.standardCoverageClaim, false);
  assert.equal(report.groups.length, 2);
  assert.ok(report.games.every(g => /^[a-f0-9]{64}$/.test(g.decks[0].sha256)));
  assert.deepEqual(report.games.slice(0, 2).map(g => g.candidateSeat), [0, 1]);
  assert.deepEqual(report, evaluateSuite(smallSuite()));
});

test('suite prevents unverified Standard execution and invalid weights or card counts', () => {
  const suite = smallSuite(); suite.standardCoverageClaim = true;
  assert.throws(() => validateSuite(suite), /gated/);
  const badWeight = smallSuite(); badWeight.opponents[0].weight = 0.9;
  assert.throws(() => validateSuite(badWeight), /sum/);
  const badCount = smallSuite(); badCount.decks[0].cards[0].count = 1e9;
  assert.throws(() => validateSuite(badCount), /count/);
});

test('a fresh policy instance is made for each seat/game, with no game seed in inputs', () => {
  const instances = [];
  const report = evaluateSuite(smallSuite(), { policyFactory(id, seed) {
    const base = createPolicy(id, seed), instance = { calls: 0 };
    instances.push(instance);
    return { act(input) { instance.calls++; assert.equal(input.seed, undefined); return base.act(input); } };
  } });
  assert.equal(instances.length, report.games.length * 2);
  assert.ok(instances.every(instance => instance.calls > 0));
});

test('errors and truncations invalidate score rather than becoming losses', () => {
  const suite = smallSuite(); suite.maxActions = 1;
  const truncated = evaluateSuite(suite);
  assert.equal(truncated.weightedScore, null);
  assert.ok(truncated.groups.every(g => g.totals.truncations === 4 && g.totals.losses === 0));
  const failed = evaluateSuite(smallSuite(), { policyFactory() { return { act() { throw new Error('private diagnostic'); } }; } });
  assert.equal(failed.weightedScore, null);
  assert.ok(failed.groups.every(g => g.totals.errors === 4 && g.totals.losses === 0));
  assert.ok(failed.games.every(g => !JSON.stringify(g).includes('private diagnostic')));
});

test('bootstrap resamples complete seat pairs and reports cell-specific uncertainty', () => {
  const result = pairSummary([[{ status: 'wins' }, { status: 'losses' }], [{ status: 'wins' }, { status: 'losses' }]], { seed: 2, resamples: 100 });
  assert.equal(result.meanPairScore, 0.5);
  assert.deepEqual(result.pairBootstrap95, [0.5, 0.5]);
});
