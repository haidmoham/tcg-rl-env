import { createHash } from 'node:crypto';
import { encodeOrdinaryDecision, indexedOrdinaryDecision } from '../integrations/local-worker/learning-features.mjs';

export const cardActionFeatureContract = Object.freeze({ schema: 'native-card-action-features/1', stateDimensions: 224,
  candidateDimensions: 191, signature: 'sha256-byte-centered/1', pooling: 'mean-card-signature/1', selection: 'uniform-chosen-card-target/1' });
export const cardActionContractSha256 = createHash('sha256').update(JSON.stringify(cardActionFeatureContract)).digest('hex');
const zero = () => Array(32).fill(0);
const digest = token => token ? [...createHash('sha256').update(token).digest()].map(byte => (byte - 127.5) / 127.5) : zero();

export function validatePrintingAliases(aliases = {}) {
  if (!aliases || typeof aliases !== 'object' || Array.isArray(aliases) || Object.keys(aliases).length > 4096
      || Object.entries(aliases).some(([key, value]) => !key || key.length > 512 || typeof value !== 'string' || !value || value.length > 512))
    throw new Error('explicit bounded printing aliases required');
  for (const start of Object.keys(aliases)) {
    const seen = new Set(); let token = start;
    while (Object.hasOwn(aliases, token)) {
      if (seen.has(token)) throw new Error('printing aliases contain a cycle');
      seen.add(token); token = aliases[token];
    }
  }
  return aliases;
}

export function encodeCardActionDecision(view, accepted = null, { printingAliases = {} } = {}) {
  const base = encodeOrdinaryDecision(view, accepted), aliases = validatePrintingAliases(printingAliases);
  const printing = card => {
    let token = card?.printingId ?? '';
    while (Object.hasOwn(aliases, token)) token = aliases[token];
    return token;
  };
  const pool = cards => {
    if (!cards.length) return zero();
    const result = zero();
    for (const token of cards.map(printing).sort()) digest(token).forEach((value, index) => { result[index] += value / cards.length; });
    return result;
  };
  const attachments = card => Object.values(card?.publicCards ?? {}).flat();
  const visible = new Map();
  const add = card => {
    if (!card) return;
    visible.set(card.instanceId, card); attachments(card).forEach(add);
  };
  const ownBoard = [view.you.active, ...view.you.bench].filter(Boolean);
  const opponentBoard = [view.opponent.active, ...view.opponent.bench].filter(Boolean);
  const discards = [...view.you.discard, ...view.opponent.discard];
  [...view.hand, ...ownBoard, ...opponentBoard, ...discards, view.stadium].filter(Boolean).forEach(add);
  const state = [...base.state, ...[view.hand, view.you.active ? [view.you.active] : [], view.you.bench,
    view.opponent.active ? [view.opponent.active] : [], view.opponent.bench, discards].flatMap(pool)];
  const candidates = view.actions.map((action, index) => {
    const source = visible.get(action.sourceCardId ?? action.actorId), target = visible.get(action.targetId);
    return [...base.candidates[index], (source?.currentRetreat ?? 0) / 5, (target?.currentRetreat ?? 0) / 5,
      (source?.conditions?.length ?? 0) / 5, (target?.conditions?.length ?? 0) / 5,
      ...digest(printing(source)), ...digest(printing(target)),
      ...digest(action.actionKey ? `action:${action.actionKey}` : action.attackName ? `attack:${action.attackName}` : ''),
      ...pool(attachments(source)), ...pool(attachments(target))];
  });
  if (state.length !== 224 || candidates.some(row => row.length !== 191)
      || [...state, ...candidates.flat()].some(value => !Number.isFinite(value))) throw new Error('card/action feature contract failed');
  return { schema: cardActionFeatureContract.schema, state, candidates, ...(accepted === null ? {} : { label: base.label }) };
}

export const indexedCardActionDecision = indexedOrdinaryDecision;

export function encodeCardChoiceDecision(view, prompt = view.prompts[0], { printingAliases = {} } = {}) {
  if (view.status !== 'playing' || !prompt || !Array.isArray(prompt.choices)
      || !prompt.choices.length || prompt.choices.length > 512 || !view.prompts.some(item => item.id === prompt.id))
    throw new Error('current bounded visible card prompt required');
  const visible = new Map();
  const add = card => { if (card) visible.set(card.instanceId, card); };
  [...view.hand, view.you.active, ...view.you.bench, view.opponent.active, ...view.opponent.bench,
    ...view.you.discard, ...view.opponent.discard].forEach(add);
  const cards = prompt.choices.map(choice => visible.get(choice.instanceId ?? choice.card?.instanceId) ?? choice.card);
  if (cards.some(card => !card?.printingId || !card.instanceId)) throw new Error('prompt contains noncard or hidden choices');
  const ownHand = new Set(view.hand.map(card => card.instanceId));
  const scope = cards.every(card => ownHand.has(card.instanceId)) ? 'own-hand' : 'visible-card';
  const semantic = `prompt:${prompt.kind}:${scope}`;
  const actions = cards.map((card, index) => ({ id: `choice-probe-${index}`, kind: 'play', sourceCardId: card.instanceId, attackName: semantic }));
  const extra = cards.filter(card => !visible.has(card.instanceId));
  const surrogate = { ...view, turn: 'you', prompts: [], hand: [...view.hand, ...extra], actions };
  const encoded = encodeCardActionDecision(surrogate, null, { printingAliases });
  const ordinaryState = encodeCardActionDecision({ ...view, turn: 'you', prompts: [], actions: [{ id: 'state-probe', kind: 'pass' }] }, null, { printingAliases }).state;
  ordinaryState[0] = Number(view.turn === 'you');
  return { ...encoded, state: ordinaryState, kind: 'selection' };
}
