import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { Environment } from './environment.mjs';
import { PROFILE, validateDeck } from './cards.mjs';
import { Random } from './rng.mjs';
import { createPolicy, POLICY_VERSIONS } from './tactical-policy.mjs';
import { runSafeEpisode } from './policy-boundary.mjs';

export const contentHash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const text = value => typeof value === 'string' && value.length > 0;
const boundedInteger = (value, min, max) => Number.isSafeInteger(value) && value >= min && value <= max;
const unique = (values, label) => { if (new Set(values).size !== values.length) throw new TypeError(`duplicate ${label}`); };

export function validateSuite(input) {
  const suite = structuredClone(input);
  if (suite.schema !== 'offline-suite-v2' || !text(suite.id) || !text(suite.version)) throw new TypeError('invalid suite identity');
  if (suite.executionProfile !== PROFILE || suite.standardCoverageClaim !== false) throw new TypeError('Standard execution is gated; only the declared regression profile is admitted');
  if (suite.engine?.id !== 'tcg-agent-kernel' || suite.engine.version !== '0.1.0') throw new TypeError('unsupported suite engine');
  if (suite.split !== 'held-out-regression' || !text(suite.splitNote)) throw new TypeError('an explicit regression split is required');
  if (!Object.hasOwn(POLICY_VERSIONS, suite.candidate)) throw new TypeError('unknown candidate policy');
  if (!Array.isArray(suite.opponents) || suite.opponents.length === 0) throw new TypeError('opponent pool is empty');
  for (const opponent of suite.opponents) if (!Object.hasOwn(POLICY_VERSIONS, opponent.policy) || !Number.isFinite(opponent.weight) || opponent.weight <= 0) throw new TypeError('invalid opponent pool entry');
  unique(suite.opponents.map(o => o.policy), 'opponent');
  if (Math.abs(suite.opponents.reduce((sum, o) => sum + o.weight, 0) - 1) > 1e-9) throw new TypeError('opponent weights must sum to one');
  if (!Array.isArray(suite.decks) || suite.decks.length === 0) throw new TypeError('no decks');
  for (const deck of suite.decks) {
    if (!text(deck.id) || !text(deck.version) || !Array.isArray(deck.cards)) throw new TypeError('invalid deck identity');
    unique(deck.cards.map(card => card.id), 'printing in deck');
    for (const card of deck.cards) if (!boundedInteger(card.count, 1, 60)) throw new TypeError('invalid card count');
    deck.cards.sort((a, b) => String(a.id).localeCompare(String(b.id)));
    deck.expanded = validateDeck(deck.cards.flatMap(card => Array(card.count).fill(card.id)));
    deck.sha256 = contentHash({ profile: PROFILE, cards: deck.cards });
  }
  unique(suite.decks.map(deck => deck.id), 'deck');
  const deckIds = new Set(suite.decks.map(deck => deck.id));
  if (!Array.isArray(suite.matchups) || suite.matchups.length === 0) throw new TypeError('no matchups');
  for (const matchup of suite.matchups) {
    if (!text(matchup.id) || !deckIds.has(matchup.candidateDeck) || !deckIds.has(matchup.opponentDeck)) throw new TypeError('invalid matchup');
    if (!boundedInteger(matchup.seedStart, 0, 0xffffffff) || !boundedInteger(matchup.seedCount, 1, 1000) || matchup.seedStart + matchup.seedCount - 1 > 0xffffffff) throw new TypeError('invalid matchup seeds');
  }
  unique(suite.matchups.map(m => m.id), 'matchup');
  const games = suite.matchups.reduce((sum, matchup) => sum + matchup.seedCount * 2 * suite.opponents.length, 0);
  if (games > 10000) throw new RangeError('suite exceeds the 10000-game desktop budget');
  if (!boundedInteger(suite.policySeedStart, 0, 0xffffffff - games * 2) || !boundedInteger(suite.bootstrapSeed, 0, 0xffffffff) || !boundedInteger(suite.bootstrapResamples, 100, 10000) || !boundedInteger(suite.maxActions, 1, 10000)) throw new TypeError('invalid evaluation budget or independent seeds');
  return suite;
}

export function pairSummary(pairs, { seed = 1, resamples = 1000 } = {}) {
  const totals = { wins: 0, losses: 0, draws: 0, errors: 0, truncations: 0 };
  const scores = [];
  for (const pair of pairs) {
    for (const game of pair) totals[game.status]++;
    if (pair.every(game => ['wins', 'losses', 'draws'].includes(game.status))) scores.push(pair.reduce((sum, game) => sum + (game.status === 'wins' ? 1 : game.status === 'draws' ? 0.5 : 0), 0) / 2);
  }
  if (totals.errors || totals.truncations || scores.length === 0) return { totals, completedPairs: scores.length, meanPairScore: null, pairBootstrap95: null };
  const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
  const rng = new Random(seed);
  const boot = Array.from({ length: resamples }, () => mean(Array.from({ length: scores.length }, () => scores[rng.int(scores.length)]))).sort((a, b) => a - b);
  return { totals, completedPairs: scores.length, meanPairScore: mean(scores), pairBootstrap95: [boot[Math.floor(resamples * 0.025)], boot[Math.ceil(resamples * 0.975) - 1]] };
}

export function evaluateSuite(manifest, { environmentFactory = options => new Environment(options), policyFactory = createPolicy } = {}) {
  const suite = validateSuite(manifest), groups = [], games = [];
  let gameIndex = 0;
  for (const matchup of suite.matchups) for (const opponent of suite.opponents) {
    const candidateDeck = suite.decks.find(d => d.id === matchup.candidateDeck), opponentDeck = suite.decks.find(d => d.id === matchup.opponentDeck);
    const pairs = [];
    for (let pair = 0; pair < matchup.seedCount; pair++) {
      const results = [];
      for (let seat = 0; seat < 2; seat++) {
        const seed = matchup.seedStart + pair, policySeeds = [suite.policySeedStart + gameIndex * 2, suite.policySeedStart + gameIndex * 2 + 1];
        const decks = seat === 0 ? [candidateDeck, opponentDeck] : [opponentDeck, candidateDeck];
        const policyIds = seat === 0 ? [suite.candidate, opponent.policy] : [opponent.policy, suite.candidate];
        const game = { id: `${matchup.id}/${opponent.policy}/${pair}/${seat}`, matchup: matchup.id, opponentPolicy: opponent.policy, candidateSeat: seat, seed, policySeeds, decks: decks.map(d => ({ id: d.id, version: d.version, sha256: d.sha256 })), policies: policyIds.map(id => ({ id, ...POLICY_VERSIONS[id] })) };
        try {
          const env = environmentFactory({ seed, decks: decks.map(d => d.expanded), record: true });
          const policies = policyIds.map((id, index) => policyFactory(id, policySeeds[index]));
          game.outcome = runSafeEpisode(env, policies, { maxActions: suite.maxActions });
          game.status = game.outcome.truncated ? 'truncations' : game.outcome.winner === null ? 'draws' : game.outcome.winner === seat ? 'wins' : 'losses';
        } catch (error) { game.status = 'errors'; game.errorType = error.name; }
        games.push(game); results.push(game); gameIndex++;
      }
      pairs.push(results);
    }
    groups.push({ matchup: matchup.id, candidateDeck: candidateDeck.id, opponentDeck: opponentDeck.id, opponentPolicy: opponent.policy, weight: opponent.weight / suite.matchups.length, ...pairSummary(pairs, { seed: suite.bootstrapSeed, resamples: suite.bootstrapResamples }) });
  }
  return {
    schema: 'offline-evaluation-v2', suite: { id: suite.id, version: suite.version, sha256: contentHash(manifest), split: suite.split },
    sourceSha256: Object.fromEntries(['cards.mjs', 'environment.mjs', 'rng.mjs', 'policies.mjs', 'policy-boundary.mjs', 'tactical-policy.mjs', 'evaluation-suite.mjs'].map(file => [file, createHash('sha256').update(readFileSync(new URL(file, import.meta.url))).digest('hex')])),
    engine: suite.engine, executionProfile: suite.executionProfile, targetFormat: suite.targetFormat, standardCoverageClaim: false,
    uncertainty: '95% paired-seed percentile bootstrap within each matchup/opponent cell; no deck-population, rules-coverage or metagame-shift uncertainty',
    weightedScore: groups.some(g => g.meanPairScore === null) ? null : groups.reduce((sum, group) => sum + group.weight * group.meanPairScore, 0),
    groups, games,
  };
}
