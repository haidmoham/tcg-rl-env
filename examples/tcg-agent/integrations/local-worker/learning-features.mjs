export const ordinaryFeatureContract = Object.freeze({ schema: 'native-ordinary-features/1', stateDimensions: 32, candidateDimensions: 27 });

const kinds = ['play', 'attack', 'ability', 'retreat', 'pass', 'concede'];
const categories = ['pokemon', 'trainer', 'energy'];
const oneHot = (size, index) => Array.from({ length: size }, (_, item) => Number(item === index));
const creatureFeatures = card => [(card?.hp ?? 0) / 400, (card?.damage ?? 0) / 400, (card?.providedEnergy ?? 0) / 10];

function sideFeatures(side) {
  const sum = key => side.bench.reduce((total, card) => total + card[key], 0);
  return [side.prizes / 6, side.deckCount / 60, side.handCount / 60, side.discard.length / 60,
    side.bench.length / 5, Number(side.active !== null), (side.active?.hp ?? 0) / 400,
    (side.active?.damage ?? 0) / 400, (side.active?.providedEnergy ?? 0) / 10,
    (side.active?.currentRetreat ?? 0) / 5, sum('hp') / 2000, sum('damage') / 2000, sum('providedEnergy') / 50];
}

export function encodeOrdinaryDecision(view, accepted = null) {
  if (view.source !== 'live-matchlogic' || view.status !== 'playing' || view.turn !== 'you'
      || view.prompts.length || !view.actions.length || view.actions.length > 512) throw new Error('ordinary native action state required');
  const visible = new Map();
  const add = (card, zone, creature = false) => {
    if (!card) return;
    visible.set(card.instanceId, { card: creature ? { ...card, category: 'pokemon' } : card, zone });
    if (creature) for (const cards of Object.values(card.publicCards ?? {})) cards.forEach(attached => add(attached, zone));
  };
  view.hand.forEach(card => add(card, 1));
  add(view.you.active, 2, true); view.you.bench.forEach(card => add(card, 3, true));
  add(view.opponent.active, 4, true); view.opponent.bench.forEach(card => add(card, 5, true));
  [...view.you.discard, ...view.opponent.discard].forEach(card => add(card, 0)); add(view.stadium, 0);
  const locate = id => {
    if (id === undefined) return { card: null, zone: 0 };
    const value = visible.get(id);
    if (!value) throw new Error('candidate references an unsupported visible zone');
    return value;
  };
  const state = [Number(view.turn === 'you'), view.turnNumber / 100, ...sideFeatures(view.you), ...sideFeatures(view.opponent),
    ...categories.map(category => view.hand.filter(card => card.category === category).length / 60), Number(view.stadium !== null)];
  const candidates = view.actions.map(action => {
    const kind = kinds.indexOf(action.kind);
    if (kind < 0) throw new Error('unknown native action kind');
    const source = locate(action.sourceCardId ?? action.actorId), target = locate(action.targetId);
    return [...oneHot(6, kind), ...oneHot(6, source.zone), ...oneHot(6, target.zone),
      ...creatureFeatures(source.card), ...creatureFeatures(target.card), ...oneHot(3, categories.indexOf(source.card?.category))];
  });
  if (state.length !== 32 || candidates.some(value => value.length !== 27)
      || [...state, ...candidates.flat()].some(value => !Number.isFinite(value))) throw new Error('native feature contract failed');
  const result = { state, candidates };
  if (accepted !== null) {
    const label = view.actions.findIndex(action => action.id === accepted.actionId);
    if (label < 0 || Object.keys(accepted).length !== 1) throw new Error('accepted label is outside legal candidates');
    result.label = label;
  }
  return result;
}

export function indexedOrdinaryDecision(view, index) {
  if (!Number.isSafeInteger(index) || index < 0 || index >= view.actions.length) throw new Error('learner chose outside the legal candidate mask');
  return { actionId: view.actions[index].id };
}
