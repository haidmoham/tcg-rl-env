import { PROFILE } from './cards.mjs';

const actionFields = {
  choose_first: ['player'], active: ['hand'], bench: ['hand'], ready: [], bonus: ['count'],
  prize: ['prize'], promote: ['bench'], pass: [], attach: ['hand', 'target'], retreat: ['bench'], attack: [],
};
const eventFields = {
  coin: ['winner'], first_player: ['player'], mulligans: ['counts'], reveal: ['players'],
  bench: ['player', 'card'], bonus: ['player', 'count'], attach: ['player', 'target', 'card'],
  retreat: ['player', 'bench', 'cost'], attack: ['player', 'card', 'damage'],
  knockout: ['player', 'card'], prize: ['player', 'remaining'], promote: ['player', 'card'],
  pass: ['player'], draw: ['player', 'count', 'turn'], terminal: ['winner', 'reason'],
};
const freeze = value => {
  if (value && typeof value === 'object') { Object.values(value).forEach(freeze); Object.freeze(value); }
  return value;
};
const strings = value => {
  if (!Array.isArray(value) || value.some(item => typeof item !== 'string')) throw new TypeError('expected card ID strings');
  return [...value];
};
const integer = value => {
  if (!Number.isSafeInteger(value)) throw new TypeError('expected integer observation field');
  return value;
};
const scalar = value => {
  if (typeof value !== 'string' && !(typeof value === 'number' && Number.isFinite(value))) throw new TypeError('expected primitive public field');
  return value;
};
const slot = value => value === null ? null : ({ card: strings([value.card])[0], hp: integer(value.hp), damage: integer(value.damage), energy: integer(value.energy) });

export function policyInput(observation, publicEvents = []) {
  if (observation?.schema !== 'observation-v1' || observation.profile !== PROFILE) throw new TypeError('an audited observation adapter is required for this profile');
  const { player, actor, firstPlayer, phase } = observation;
  const turn = integer(observation.turn), revision = integer(observation.revision);
  if (typeof phase !== 'string' || ![null, 0, 1].includes(firstPlayer)) throw new TypeError('invalid public phase or first player');
  if (![0, 1].includes(player) || player !== actor) throw new TypeError('policy must act as the current player');
  if (!Array.isArray(observation.players) || observation.players.length !== 2) throw new TypeError('two player views required');
  const players = observation.players.map((p, index) => ({
    handCount: integer(p.handCount), deckCount: integer(p.deckCount), prizeCount: integer(p.prizeCount),
    ...(index === player ? { hand: strings(p.hand) } : {}),
    active: slot(p.active), bench: p.bench.map(slot), discard: strings(p.discard), mulligans: integer(p.mulligans),
  }));
  const actions = observation.actions.map(action => {
    if (!Object.hasOwn(actionFields, action.kind)) throw new TypeError(`unadapted action kind ${action.kind}`);
    return Object.fromEntries(['kind', 'revision', ...actionFields[action.kind]].map(key => [key, key === 'kind' ? action.kind : integer(action[key])]));
  });
  const events = publicEvents.map(event => {
    if (!Object.hasOwn(eventFields, event.type)) throw new TypeError(`unadapted event ${event.type}`);
    const copy = { type: event.type };
    if (event.type === 'reveal') copy.players = event.players.map(p => ({ active: strings([p.active])[0], bench: strings(p.bench) }));
    else if (event.type === 'mulligans') copy.counts = event.counts.map(integer);
    else for (const key of eventFields[event.type]) copy[key] = scalar(event[key]);
    return copy;
  });
  return freeze({ schema: 'policy-input-v1', observation: { schema: 'observation-v1', profile: PROFILE, player, actor, firstPlayer, phase, turn, revision, players, actions }, publicEvents: events });
}

export function runSafeEpisode(env, policies, { maxActions = 2000 } = {}) {
  if (!Number.isSafeInteger(maxActions) || maxActions < 1) throw new TypeError('maxActions must be positive');
  const cursors = [0, 0];
  let actions = 0;
  while (!env.terminated && actions < maxActions) {
    const player = env.currentPlayer, events = env.publicEvents();
    const input = policyInput(env.observe(player), events.slice(cursors[player]));
    cursors[player] = events.length;
    env.step(policies[player].act(input)); actions++;
  }
  return { ...env.outcome, truncated: !env.terminated, actions };
}
