import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

export const entityFeatureContract = Object.freeze(JSON.parse(readFileSync(new URL('./entity-contract.json', import.meta.url), 'utf8')));
export const entityContractSha256 = createHash('sha256').update(JSON.stringify(entityFeatureContract)).digest('hex');
const C = entityFeatureContract;
const kinds = ['play', 'attack', 'ability', 'retreat', 'pass', 'concede'];
const oneHot = (items, value) => items.map(item => Number(item === value));
const finite = value => typeof value === 'number' && Number.isFinite(value);
const number = (value, scale = 1) => finite(value) ? value / scale : 0;
const usagePair = value => typeof value === 'boolean' ? [1, Number(value)]
  : value && value.known === true && typeof value.value === 'boolean' ? [1, Number(value.value)] : [0, 0];
const countPair = value => value?.known === true && finite(value.value) && value.value >= 0 ? [1, value.value / 6] : [0, 0];
const vector = (value, size) => Array.isArray(value) && value.length === size && value.every(finite) ? value : Array(size).fill(0);

export function encodeEntityDecision(view, { catalog, decisionContext = view.decisionContext ?? {}, context } = {}) {
  if (context) decisionContext = context;
  if (view.source !== 'live-matchlogic' || view.status !== 'playing' || !catalog || catalog.schema !== 'model5-public-catalog/1'
      || catalog.contractSha256 !== entityContractSha256)
    throw new Error('bounded player-visible entity decision and public catalog required');
  const prompt = view.prompts[0];
  const offers = prompt ? prompt.choices : view.actions;
  if (!Array.isArray(offers) || !offers.length || offers.length > C.maxCandidates || (!prompt && view.turn !== 'you'))
    throw new Error('current complete legal candidate list required');
  const canonical = token => {
    const seen = new Set();
    while (catalog.printingAliases?.[token]) {
      if (seen.has(token)) throw new Error('catalog alias cycle');
      seen.add(token); token = catalog.printingAliases[token];
    }
    return token;
  };
  const cardId = card => catalog.cardIds?.[canonical(card?.printingId)] ?? catalog.cardIds?.[card?.definitionId] ?? 0;
  const rows = [], indices = new Map();
  const add = (card, owner, zone, parent = -1, position = 0) => {
    if (!card) return -1;
    if (!card.instanceId || !card.printingId) throw new Error('visible physical card identity required');
    if (indices.has(card.instanceId)) return indices.get(card.instanceId);
    if (rows.length >= C.maxEntities) throw new Error('entity bound exceeded; truncation forbidden');
    const index = rows.length;
    indices.set(card.instanceId, index); rows.push({ card, owner, zone, parent, position });
    for (const [group, attachedZone] of [['energy', 'energy'], ['tools', 'tool'], ['evolutions', 'evolution']])
      for (const attached of card.publicCards?.[group] ?? []) add(attached, owner, attachedZone, index);
    return index;
  };
  view.hand.forEach(card => add(card, 'you', 'hand'));
  for (const owner of ['you', 'opponent']) {
    const side = view[owner];
    add(side.active, owner, 'active');
    side.bench.forEach((card, index) => add(card, owner, 'bench', -1, index + 1));
    side.discard.forEach(card => add(card, owner, 'discard'));
  }
  add(view.stadium, view.stadium?.owner ?? 'unknown', 'stadium');
  for (const card of prompt?.revealedCards ?? []) add(card, card.owner ?? 'unknown', 'revealed');
  for (const choice of prompt?.choices ?? []) {
    add(choice.card, choice.card?.owner ?? 'unknown', 'revealed');
    add(choice.sourceCard, choice.sourceCard?.owner ?? 'unknown', 'revealed');
  }
  const nativeContext = prompt?.context ?? {};
  const fallbackContext = view.turn === 'you' ? decisionContext : {};
  const originCard = nativeContext.originCard ?? fallbackContext.originCard;
  if (originCard) add(originCard, originCard.owner ?? 'unknown', 'origin');
  const originAction = fallbackContext.originAction ?? {};
  const originSourceId = nativeContext.sourceCardId ?? nativeContext.originCard?.instanceId
    ?? originCard?.instanceId ?? originAction.sourceCardId ?? originAction.actorId;
  const sourceIndex = id => id == null ? -1 : indices.get(id) ?? -1;
  const originSourceIndex = sourceIndex(originSourceId);
  const actionId = (action, source) => {
    const key = action.actionKey;
    if (key && catalog.actionIds?.[key] !== undefined) return catalog.actionIds[key];
    const card = rows[source]?.card;
    const entry = catalog.cards?.[cardId(card)];
    return catalog.actionIds?.[`${entry?.key}/${action.kind ?? 'play'}`]
      ?? catalog.actionNames?.[`${entry?.key}/${action.attackName ?? ''}`] ?? 0;
  };
  const originActionId = actionId({ ...originAction, actionKey: nativeContext.actionKey ?? originAction.actionKey }, originSourceIndex);
  const entityCardIds = rows.map(row => cardId(row.card));
  const entities = rows.map(({ card, owner, zone, position }, index) => {
    const pokemon = finite(card.hp) && finite(card.damage);
    const types = card.providedEnergyTypes;
    const usage = { ...(decisionContext.entityUsage?.[card.instanceId] ?? card.usage ?? {}),
      ...(typeof card.enteredPlayThisTurn === 'boolean' ? { enteredPlay: card.enteredPlayThisTurn } : {}),
      ...(typeof card.evolvedThisTurn === 'boolean' ? { evolved: card.evolvedThisTurn } : {}) };
    return [...oneHot(['you', 'opponent', 'unknown'], owner), ...oneHot(C.zones, zone), position / 8, Number(pokemon),
      number(card.hp, 400), number(card.damage, 400), pokemon ? (card.hp - card.damage) / 400 : 0,
      number(card.providedEnergy, 10), number(card.currentRetreat, 5), number(card.printedRetreat, 5), Number(Array.isArray(types)),
      ...C.energyTypes.map(type => Array.isArray(types) ? types.filter(value => value.toLowerCase() === type).length / 10 : 0),
      Number(Array.isArray(card.conditions)), ...oneHotConditions(card.conditions), Number(card.publicCards != null),
      ...['energy', 'tools', 'evolutions'].map(group => number(card.publicCards?.[group]?.length, 60)),
      ...['abilityUsed', 'enteredPlay', 'evolved'].flatMap(key => usagePair(usage[key])),
      ...vector(catalog.cards?.[entityCardIds[index]]?.facts, C.cardFactDimensions)];
  });
  const usage = decisionContext.usage ?? decisionContext.ownUsage ?? {};
  const publicContext = view.publicContext ?? decisionContext.publicContext ?? {};
  const publicFeatures = ['you', 'opponent'].flatMap(side => {
    const lock = publicContext.itemLock?.[side];
    const expiryKnown = lock?.expiryKnown === true && finite(lock.remainingTurns) && lock.remainingTurns >= 0;
    return [...usagePair(lock), Number(expiryKnown), expiryKnown ? lock.remainingTurns / 10 : 0,
      ...countPair(publicContext.knockoutsDuringPreviousOpponentTurn?.[side])];
  });
  const sideGlobals = side => [side.prizes / 6, side.deckCount / 60, side.handCount / 60, side.discard.length / 60, side.bench.length / 8];
  const globals = [...oneHot(['you', 'opponent'], view.turn), ...oneHot(['setup', 'playing', 'finished'], view.status), view.turnNumber / 100,
    ...sideGlobals(view.you), ...sideGlobals(view.opponent), ...['energyPlayed', 'supporterPlayed', 'retreated'].flatMap(key => usagePair(usage[key])),
    Number(Boolean(prompt)), ...oneHot(C.promptKinds, prompt?.kind ?? 'unknown'), number(prompt?.min, 60), number(prompt?.max, 60),
    Number(prompt?.ordered === true), Number(prompt?.cancelable === true), Number(prompt?.requiresPreview === true), number(prompt?.allocationUnit, 10),
    ...oneHot(['none', 'add', 'final'], prompt?.allocationMode ?? 'none'), Number(originSourceIndex >= 0), Number(originActionId > 0),
    number(prompt?.groups.length, 512), ...oneHot(C.selectionKinds, C.selectionKinds.includes(nativeContext.selectionKind) ? nativeContext.selectionKind : 'unknown'),
    ...publicFeatures, ...['you', 'opponent'].flatMap(side => usagePair(publicContext.knockoutDrawEligible?.[side]))];
  const candidateSources = [], candidateTargets = [], candidateActors = [], candidateActionIds = [];
  const candidates = offers.map(offer => {
    const source = prompt ? sourceIndex(offer.sourceCardId ?? offer.sourceCard?.instanceId ?? offer.instanceId ?? offer.card?.instanceId) : sourceIndex(offer.sourceCardId ?? offer.actorId);
    const target = sourceIndex(offer.targetId);
    const actor = prompt ? originSourceIndex : sourceIndex(offer.actorId ?? offer.sourceCardId);
    const id = actionId(prompt ? { ...offer, kind: originAction.kind, actionKey: offer.actionKey ?? nativeContext.actionKey ?? originAction.actionKey } : offer,
      prompt ? originSourceIndex : source);
    candidateSources.push(source); candidateTargets.push(target); candidateActors.push(actor); candidateActionIds.push(id);
    const group = prompt?.groups.find(item => item.id === offer.group);
    return [...oneHot(kinds, prompt ? '' : offer.kind), ...oneHot(['none', 'active', 'bench', 'stadium', 'hand', 'discard'], offer.destinationZone ?? 'none'),
      ...oneHot(C.promptKinds, prompt?.kind ?? 'unknown'), number(offer.minAmount, 60), number(offer.maxAmount, 60), number(offer.initialAmount, 60),
      Number([offer.minAmount, offer.maxAmount, offer.initialAmount].some(finite)), number(group?.min, 60), number(group?.max, 60), Number(Boolean(group)),
      group ? (prompt.groups.indexOf(group) + 1) / prompt.groups.length : 0, Number(Boolean(prompt)),
      ...vector(catalog.actions?.[id]?.facts, C.actionFactDimensions)];
  });
  for (const [values, size] of [[globals, C.globalDimensions], ...entities.map(row => [row, C.entityDimensions]), ...candidates.map(row => [row, C.candidateDimensions])])
    if (values.length !== size || !values.every(finite)) throw new Error('entity feature contract failed');
  return { schema: C.schema, globals, entities, entityCardIds, entityParents: rows.map(row => row.parent), candidates,
    candidateSources, candidateTargets, candidateActors, candidateActionIds, originActionId, originSourceIndex, kind: prompt ? 'selection' : 'ordinary' };
}

function oneHotConditions(conditions) {
  return ['poisoned', 'burned', 'paralyzed', 'confused', 'asleep'].map(value => Number(conditions?.includes(value) === true));
}

export function encodeEntityChoiceDecision(view, prompt = view.prompts[0], options = {}) {
  if (!prompt || view.prompts[0]?.id !== prompt.id) throw new Error('first current prompt required');
  return encodeEntityDecision(view, options);
}

export function indexedEntityDecision(view, index) {
  const prompt = view.prompts[0], choices = prompt?.choices ?? view.actions;
  if (!Number.isSafeInteger(index) || index < 0 || index >= choices.length) throw new Error('entity decision outside legal candidate mask');
  return prompt ? { promptId: prompt.id, choiceIds: [choices[index].id] } : { actionId: choices[index].id };
}
