import { attackDamage } from './cards.mjs';
import { Random } from './rng.mjs';

export function randomPolicy(seed = 1) {
  const rng = new Random(seed);
  return observation => observation.actions[rng.int(observation.actions.length)];
}

export function greedyPolicy(observation) {
  const actions = observation.actions, own = observation.players[observation.player], opponent = observation.players[1 - observation.player];
  return actions.find(action => action.kind === 'choose_first' && action.player === observation.player)
    ?? actions.find(action => action.kind === 'active' && own.hand[action.hand] === 'base1-52')
    ?? actions.find(action => action.kind === 'bonus' && action.count === Math.max(...actions.map(a => a.count ?? 0)))
    ?? actions.find(action => action.kind === 'bench')
    ?? actions.find(action => action.kind === 'attach' && action.target === -1 && own.active.energy === 0)
    ?? actions.find(action => action.kind === 'attack')
    ?? actions.find(action => action.kind === 'attach' && action.target >= 0 && own.bench[action.target].energy === 0)
    ?? actions.find(action => action.kind === 'promote' && own.bench[action.bench].energy > 0)
    ?? actions.find(action => action.kind === 'prize' || action.kind === 'promote' || action.kind === 'ready' || action.kind === 'active')
    ?? actions.find(action => action.kind === 'retreat' && own.bench[action.bench].energy > 0 && opponent.active && attackDamage(own.bench[action.bench].card, opponent.active.card) > attackDamage(own.active.card, opponent.active.card))
    ?? actions.find(action => action.kind === 'pass')
    ?? actions[0];
}

export function delayedAttackPolicy(observation) {
  return greedyPolicy(observation.turn < 50
    ? { ...observation, actions: observation.actions.filter(action => action.kind !== 'attack') }
    : observation);
}

export function runEpisode(env, policies, { maxActions = 2000 } = {}) {
  if (!Number.isSafeInteger(maxActions) || maxActions < 1) throw new TypeError('maxActions must be positive');
  let actions = 0;
  while (!env.terminated && actions < maxActions) {
    const player = env.currentPlayer;
    env.step(policies[player](env.observe(player))); actions++;
  }
  return { ...env.outcome, truncated: !env.terminated, actions };
}
