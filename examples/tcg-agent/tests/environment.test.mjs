import test from 'node:test';
import assert from 'node:assert/strict';
import { Environment, IllegalActionError, replay, stepBatch } from '../src/environment.mjs';
import { CARDS, BASELINE_DECK, attackDamage, importDeckCorpus, UnsupportedMechanicError } from '../src/cards.mjs';
import { Random } from '../src/rng.mjs';
import { greedyPolicy, delayedAttackPolicy, randomPolicy, runEpisode } from '../src/policies.mjs';

function choose(env, kind, predicate = () => true) {
  const action = env.legalActions().find(a => a.kind === kind && predicate(a));
  assert.ok(action, `expected ${kind} in ${env.observe().phase}`);
  env.step(action);
}

function start(seed = 1) {
  const env = new Environment({ seed });
  while (env.observe().turn === 0) env.step(greedyPolicy(env.observe()));
  return env;
}

function assertConservation(observation) {
  if (observation.turn === 0) return;
  for (const p of observation.players) {
    const onBoard = [p.active, ...p.bench].filter(Boolean).reduce((n, slot) => n + 1 + slot.energy, 0);
    assert.equal(p.deckCount + p.handCount + p.prizeCount + p.discard.length + onBoard, 60);
  }
}

test('printed attack facts, typed costs, weakness and resistance arithmetic', () => {
  assert.equal(CARDS['base1-52'].hp, 50);
  assert.deepEqual(CARDS['base1-52'].attack.cost, ['fighting']);
  assert.equal(CARDS['base1-61'].retreat, 0);
  assert.equal(attackDamage('base1-52', 'base1-61'), 40);
  assert.equal(attackDamage('base1-61', 'base1-52'), 20);
  assert.equal(attackDamage('base1-52', 'base1-52'), 20);
});

test('unknown cards, unsupported profiles and invalid decks fail at admission', () => {
  assert.throws(() => new Environment({ profile: 'Standard' }), UnsupportedMechanicError);
  assert.throws(() => new Environment({ decks: [[...BASELINE_DECK.slice(1), 'unknown'], BASELINE_DECK] }), UnsupportedMechanicError);
  assert.throws(() => new Environment({ decks: [BASELINE_DECK.slice(1), BASELINE_DECK] }), /60/);
  assert.throws(() => new Environment({ decks: [Array(60).fill('base1-97'), BASELINE_DECK] }), /Basic/);
  assert.throws(() => new Environment({ decks: [[...Array(5).fill('base1-52'), ...Array(55).fill('base1-97')], BASELINE_DECK] }), /four copies/);
  assert.throws(() => new Environment({ seed: -1 }), /uint32/);
  assert.throws(() => new Environment({ decks: [Array.from({ length: 60 }, () => new String('base1-52')), BASELINE_DECK] }), /primitive strings/);
});

test('deck-corpus boundary maps explicit printings and rejects name-only or unsupported data', () => {
  assert.deepEqual(importDeckCorpus({ schema: 'deck-corpus-v1', cards: [{ set: 'base1', number: '52', count: 4 }, { set: 'base1', number: '61', count: 4 }, { set: 'base1', number: '97', count: 52 }] }), BASELINE_DECK);
  assert.throws(() => importDeckCorpus({ schema: 'deck-corpus-v1', cards: [{ name: 'Machop', count: 60 }] }), UnsupportedMechanicError);
  assert.throws(() => importDeckCorpus({ schema: 'deck-corpus-v1', cards: [{ set: 'base1', number: '97', count: 61 }] }), /count/);
});

test('seeded random generator has a stable sequence and shuffle preserves cards', () => {
  const rng = new Random(1);
  assert.deepEqual(Array.from({ length: 3 }, () => rng.next()), [0.3678755429573357, 0.08161311969161034, 0.8205357783008367]);
  assert.deepEqual(new Random(11).shuffle(BASELINE_DECK).sort(), [...BASELINE_DECK].sort());
  assert.notDeepEqual(new Random(11).shuffle(BASELINE_DECK), new Random(12).shuffle(BASELINE_DECK));
});

test('setup exposes coin decision, active choice, mulligans, six prizes and first draw', () => {
  const env = new Environment({ seed: 2 });
  const initial = env.observe();
  assert.equal(initial.phase, 'choose_first');
  assert.equal(initial.players[0].handCount, 0);
  assert.equal(initial.players[1].handCount, 0);
  assert.deepEqual(initial.players[initial.player].hand, []);
  assert.ok(initial.players.every(p => p.mulligans === 0));
  choose(env, 'choose_first', a => a.player === 1);
  assert.equal(env.observe(0).firstPlayer, 1);
  assert.equal(env.observe(1).firstPlayer, 1);
  assert.ok(env.observe().players.every(p => p.handCount === 7));
  while (env.observe().turn === 0) env.step(greedyPolicy(env.observe()));
  assert.equal(env.currentPlayer, 1);
  assert.equal(env.observe().turn, 1);
  assert.deepEqual(env.observe().players.map(p => p.prizeCount), [6, 6]);
  assert.ok(env.observe().players.every(p => p.active !== null));
  assertConservation(env.observe());
});

test('post-bonus setup does not allow benching cards withheld from the original hand', () => {
  const env = new Environment({ seed: 2 });
  choose(env, 'choose_first'); choose(env, 'active'); choose(env, 'active');
  choose(env, 'ready'); choose(env, 'ready');
  choose(env, 'bonus', a => a.count === 0); choose(env, 'bonus', a => a.count === 0);
  assert.equal(env.legalActions().some(a => a.kind === 'bench'), false);
  choose(env, 'ready');
  assert.equal(env.legalActions().some(a => a.kind === 'bench'), false);
});

test('first player cannot attack, attachment and retreat each occur at most once per turn', () => {
  const env = start(2);
  assert.equal(env.legalActions().some(a => a.kind === 'attack'), false);
  choose(env, 'attach');
  assert.equal(env.legalActions().some(a => a.kind === 'attach'), false);
  const retreat = env.legalActions().find(a => a.kind === 'retreat');
  assert.ok(retreat);
  const before = env.observe().players[env.currentPlayer];
  env.step(retreat);
  const after = env.observe().players[env.currentPlayer];
  assert.equal(after.discard.length - before.discard.length, CARDS[before.active.card].retreat);
  assert.equal(env.legalActions().some(a => a.kind === 'retreat'), false);
  choose(env, 'pass');
  choose(env, 'attach');
  assert.equal(env.legalActions().some(a => a.kind === 'attack'), true);
});

test('illegal and stale actions leave the game unchanged', () => {
  const env = start();
  const stale = env.legalActions().find(a => a.kind === 'pass');
  choose(env, 'attach');
  const before = env.replayRecord();
  assert.throws(() => env.step(stale), IllegalActionError);
  assert.throws(() => env.step({ kind: 'attack', revision: env.observe().revision }), IllegalActionError);
  assert.throws(() => env.step(null), IllegalActionError);
  assert.deepEqual(env.replayRecord(), before);
});

test('observations and public logs hide private hands, prizes, deck order and setup identities', () => {
  const env = new Environment({ seed: 9 });
  choose(env, 'choose_first');
  choose(env, 'active');
  const setupView = env.observe(1);
  assert.equal(setupView.players[0].active, null);
  const view = env.observe(0);
  assert.equal(view.players[1].hand, undefined);
  assert.equal(view.players[1].deck, undefined);
  assert.equal(view.players[1].prizes, undefined);
  assert.equal(view.seed, undefined);
  assert.deepEqual(env.observe(1 - env.currentPlayer).actions, []);
  view.players[0].hand.length = 0;
  assert.equal(env.observe(0).players[0].handCount, 6);
  while (!env.terminated) env.step(greedyPolicy(env.observe()));
  const privateTypes = env.publicEvents().filter(event => ['draw', 'bonus', 'prize'].includes(event.type));
  assert.ok(privateTypes.length > 0);
  assert.ok(privateTypes.every(event => event.card === undefined && event.cards === undefined));
});

test('greedy seeded games resolve damage, KOs, prize choices, promotions and terminal outcomes', () => {
  let prizes = 0, promotions = 0, knockouts = 0;
  for (let seed = 1; seed <= 50; seed++) {
    const env = new Environment({ seed });
    let steps = 0;
    while (!env.terminated && steps++ < 2000) {
      env.step(greedyPolicy(env.observe()));
      assertConservation(env.observe());
    }
    assert.ok(env.terminated, `seed ${seed} must finish`);
    assert.ok(['prizes', 'no_pokemon', 'deckout'].includes(env.outcome.reason));
    assert.equal(env.outcome.reward.reduce((a, b) => a + b), 0);
    assert.deepEqual(env.legalActions(), []);
    assert.throws(() => env.step({ kind: 'pass' }), IllegalActionError);
    for (const event of env.publicEvents()) {
      prizes += event.type === 'prize'; promotions += event.type === 'promote'; knockouts += event.type === 'knockout';
    }
  }
  assert.ok(prizes > 0 && promotions > 0 && knockouts > 0);
});

test('passing to deckout loses only when a turn draw fails', () => {
  const env = start(3);
  while (!env.terminated) choose(env, 'pass');
  assert.equal(env.outcome.reason, 'deckout');
  const view = env.observe();
  assert.equal(view.players[1 - env.outcome.winner].deckCount, 0);
  assert.ok(view.players.every(p => p.prizeCount === 6));
});

test('the sixth prize wins a developed game with opposing Pokemon still in play', () => {
  const env = new Environment({ seed: 2 });
  const result = runEpisode(env, [delayedAttackPolicy, delayedAttackPolicy]);
  assert.equal(result.reason, 'prizes');
  assert.equal(result.winner, 1);
  assert.ok(result.turns >= 50);
  const view = env.observe();
  assert.equal(view.players[1].prizeCount, 0);
  assert.ok(view.players[0].bench.length > 0);
  assert.equal(env.publicEvents().filter(e => e.type === 'prize' && e.player === 1).length, 6);
  assert.deepEqual(replay(env.replayRecord()).outcome, env.outcome);
});

test('different hidden opponent hands can produce exactly the same player observation', () => {
  let witnessed = false;
  for (let seed = 1; seed <= 100; seed++) {
    const a = new Environment({ seed });
    const b = new Environment({ seed, decks: [BASELINE_DECK, [...BASELINE_DECK].reverse()] });
    choose(a, 'choose_first', action => action.player === 0);
    choose(b, 'choose_first', action => action.player === 0);
    if (JSON.stringify(a.observe(0)) === JSON.stringify(b.observe(0)) && JSON.stringify(a.observe(1).players[1].hand) !== JSON.stringify(b.observe(1).players[1].hand)) {
      witnessed = true;
      assert.deepEqual(a.observe(0).actions, b.observe(0).actions);
      break;
    }
  }
  assert.ok(witnessed, 'must compare distinct private worlds behind one information set');
});

test('complete records replay exactly and tampered records fail', () => {
  const env = new Environment({ seed: 71 });
  const outcome = runEpisode(env, [randomPolicy(81), greedyPolicy]);
  assert.ok(outcome.terminated && !outcome.truncated);
  const record = JSON.parse(JSON.stringify(env.replayRecord()));
  assert.deepEqual(replay(record).replayRecord(), record);
  record.events.at(-1).winner = 7;
  assert.throws(() => replay(record), /diverged/);
});

test('independent games can be stepped in an interleaved batch without RNG interference', () => {
  const envs = [new Environment({ seed: 11 }), new Environment({ seed: 12 })];
  while (envs.some(e => !e.terminated)) {
    const active = envs.filter(e => !e.terminated);
    stepBatch(active, active.map(env => greedyPolicy(env.observe())));
  }
  for (let i = 0; i < 2; i++) {
    const reference = new Environment({ seed: i + 11 });
    runEpisode(reference, [greedyPolicy, greedyPolicy]);
    assert.deepEqual(envs[i].replayRecord(), reference.replayRecord());
  }
  assert.throws(() => stepBatch(envs, []), /lengths/);
});

test('action budget is an explicit truncation and logless training retains identical behavior', () => {
  const a = new Environment({ seed: 31 }), b = new Environment({ seed: 31, record: false });
  assert.deepEqual(runEpisode(a, [greedyPolicy, greedyPolicy]), runEpisode(b, [greedyPolicy, greedyPolicy]));
  assert.deepEqual(a.observe(), b.observe());
  assert.throws(() => b.replayRecord(), /disabled/);
  const limited = runEpisode(new Environment(), [greedyPolicy, greedyPolicy], { maxActions: 1 });
  assert.equal(limited.truncated, true);
  assert.equal(limited.terminated, false);
  assert.equal(limited.winner, null);
});
