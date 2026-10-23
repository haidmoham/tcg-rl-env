import { CARDS, PROFILE, attackDamage } from './cards.mjs';
import { Random } from './rng.mjs';
import { greedyPolicy, randomPolicy } from './policies.mjs';

export const POLICY_VERSIONS = Object.freeze({
  'tactical-basic-v1': { version: '1', profile: PROFILE, method: 'public-observation legal-action scoring' },
  'greedy-v1': { version: '1', profile: PROFILE, method: 'frozen priority baseline' },
  'legal-random-v1': { version: '1', profile: PROFILE, method: 'uniform offered-action baseline' },
});

function score(observation, action) {
  const own = observation.players[observation.player], opponent = observation.players[1 - observation.player];
  const damage = card => opponent.active ? attackDamage(card, opponent.active.card) : CARDS[card].attack.damage;
  const readiness = p => p.energy >= CARDS[p.card].attack.cost.length;
  const strength = p => CARDS[p.card].hp - p.damage + damage(p.card) + (readiness(p) ? 60 : 0);
  switch (action.kind) {
    case 'choose_first': return action.player === observation.player ? 10 : 0;
    case 'active': { const card = CARDS[own.hand[action.hand]]; return card.hp + card.attack.damage; }
    case 'bonus': return action.count;
    case 'ready': return 0;
    case 'bench': return 50 + CARDS[own.hand[action.hand]].hp / 100;
    case 'attach': {
      const target = action.target === -1 ? own.active : own.bench[action.target];
      return !readiness(target) ? (action.target === -1 ? 160 : 40) : -5;
    }
    case 'attack': {
      const hit = damage(own.active.card), knockout = opponent.active.damage + hit >= opponent.active.hp;
      return knockout ? 1000 + 100 / own.prizeCount : 100 + hit;
    }
    case 'retreat': {
      const next = own.bench[action.bench];
      if (!readiness(next)) return -50;
      if (!readiness(own.active) || damage(next.card) > damage(own.active.card)) return 180 + damage(next.card);
      return -10;
    }
    case 'promote': return strength(own.bench[action.bench]);
    case 'prize': return 0;
    case 'pass': return -1000;
    default: throw new TypeError(`unscored action ${action.kind}`);
  }
}

export function createPolicy(id, seed) {
  if (!Object.hasOwn(POLICY_VERSIONS, id)) throw new TypeError(`unknown policy ${id}`);
  const rng = new Random(seed), random = randomPolicy(seed);
  return {
    id, ...POLICY_VERSIONS[id],
    act(input) {
      const observation = input.observation;
      if (input.schema !== 'policy-input-v1' || observation.profile !== PROFILE) throw new TypeError('policy requires its audited profile');
      if (id === 'greedy-v1') return greedyPolicy(observation);
      if (id === 'legal-random-v1') return random(observation);
      const scored = observation.actions.map(action => ({ action, score: score(observation, action) }));
      const best = Math.max(...scored.map(item => item.score));
      const ties = scored.filter(item => item.score === best);
      return ties[rng.int(ties.length)].action;
    },
  };
}
