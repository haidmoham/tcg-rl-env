import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { encodeEntityDecision } from './entity-features.mjs';

export const model6FeatureContract = Object.freeze(JSON.parse(readFileSync(new URL('./model6-contract.json', import.meta.url), 'utf8')));
export const model6ContractSha256 = createHash('sha256').update(JSON.stringify(model6FeatureContract)).digest('hex');
const C = model6FeatureContract;
const names = { R: 'fire', P: 'psychic', D: 'darkness', W: 'water', G: 'grass', L: 'lightning',
  F: 'fighting', M: 'metal', C: 'colorless', Y: 'fairy', N: 'dragon', A: 'rainbow' };
const zeros = n => Array(n).fill(0);
const boundedCount = value => Number.isSafeInteger(value) && value >= 0 && value <= 100 ? value : null;
function units(values) {
  const result = zeros(C.energyTypes.length);
  if (!Array.isArray(values) || values.length > 100) return { known: false, counts: result, total: 0 };
  for (const value of values) {
    if (typeof value !== 'string') return { known: false, counts: zeros(result.length), total: 0 };
    const type = names[value.replace(/^\{|\}$/g, '')];
    if (!type) return { known: false, counts: zeros(result.length), total: 0 };
    result[C.energyTypes.indexOf(type)]++;
  }
  return { known: true, counts: result, total: values.length };
}
const scaled = value => value / 10;
function attackFacts(attack) {
  const printed = units(attack.printedUnits);
  printed.known &&= attack.printedKnown === true;
  const alternatives = Array.isArray(attack.alternatives) ? attack.alternatives : [];
  if (alternatives.length > 32) throw new Error('native cost-alternative bound exceeded');
  const decoded = alternatives.map(alt => ({ cost: units(alt.costUnits), missing: units(alt.missingUnits), payable: alt.payable }));
  const effectiveKnown = attack.effectiveKnown === true && decoded.length > 0 && decoded.every(row => row.cost.known);
  const missingKnown = attack.energyKnown === true && decoded.length > 0 && decoded.every(row => row.missing.known && typeof row.payable === 'boolean');
  // Pick the least unpaid native alternative for display features; retain the full
  // alternative count and authoritative any-alternative payable result separately.
  const best = [...decoded].sort((a, b) => a.missing.total - b.missing.total || a.cost.total - b.cost.total)[0];
  const cost = effectiveKnown ? best.cost : { counts: zeros(C.energyTypes.length), total: 0 };
  const missing = missingKnown ? best.missing : { counts: zeros(C.energyTypes.length), total: 0 };
  const energyKnown = missingKnown && typeof attack.energyPayable === 'boolean';
  const legalKnown = attack.attackLegalKnown === true && typeof attack.attackLegal === 'boolean';
  const comparable = printed.known && effectiveKnown && decoded.length === 1;
  const equal = comparable && printed.counts.every((n, i) => n === cost.counts[i]);
  const row = [Number(printed.known), ...printed.counts.map(scaled), Number(effectiveKnown), ...cost.counts.map(scaled),
    Number(energyKnown), Number(energyKnown && attack.energyPayable), Number(missingKnown), ...missing.counts.map(scaled),
    Number(legalKnown), Number(legalKnown && attack.attackLegal), alternatives.length / 10,
    Number(comparable), Number(equal), Number(effectiveKnown), Number(effectiveKnown && cost.total === 0)];
  return { row, printed, cost, missing, effectiveKnown, missingKnown, energyKnown, legalKnown,
    payable: energyKnown && attack.energyPayable, legal: legalKnown && attack.attackLegal, alternatives: alternatives.length };
}
function creatureFacts(creature) {
  if (!creature) return { row: zeros(C.energyEntityDimensions), attacks: [] };
  if (!Array.isArray(creature.attacks) || creature.attacks.length > 32) throw new Error('complete printed attacks required');
  const attacks = creature.attacks.map(attackFacts), supply = units(creature.suppliedUnits);
  const supplyKnown = creature.suppliedKnown === true && supply.known;
  const retreat = creature.retreat ?? {};
  const retreatKnown = retreat.known === true && boundedCount(retreat.cost) !== null
    && boundedCount(retreat.missing) !== null && typeof retreat.payable === 'boolean';
  const all = predicate => attacks.length > 0 && attacks.every(predicate);
  const missingKnown = all(a => a.missingKnown);
  const ordered = [...attacks].sort((a, b) => a.missing.total - b.missing.total);
  const minMissing = missingKnown ? ordered[0].missing.counts : zeros(C.energyTypes.length);
  const maxMissing = missingKnown ? ordered.at(-1).missing.counts : zeros(C.energyTypes.length);
  const extrema = (known, values) => [Number(known), known ? Math.max(...values) / 10 : 0,
    Number(known), known ? Math.min(...values) / 10 : 0];
  const printedKnown = all(a => a.printed.known), effectiveKnown = all(a => a.effectiveKnown);
  const energyKnown = all(a => a.energyKnown), legalKnown = all(a => a.legalKnown);
  const row = [Number(supplyKnown), ...(supplyKnown ? supply.counts.map(scaled) : zeros(C.energyTypes.length)), supplyKnown ? supply.total / 10 : 0,
    Number(retreatKnown), retreatKnown ? retreat.cost / 10 : 0, retreatKnown ? retreat.missing / 10 : 0, Number(retreatKnown && retreat.payable),
    attacks.length / 10, Number(printedKnown), Number(effectiveKnown), Number(energyKnown), Number(energyKnown && attacks.some(a => a.payable)),
    Number(legalKnown), Number(legalKnown && attacks.some(a => a.legal)), Number(missingKnown), ...minMissing.map(scaled),
    Number(missingKnown), ...maxMissing.map(scaled), ...extrema(printedKnown, attacks.map(a => a.printed.total)),
    ...extrema(effectiveKnown, attacks.map(a => a.cost.total)), Number(effectiveKnown),
    effectiveKnown ? attacks.reduce((n, a) => n + a.alternatives, 0) / 10 : 0,
    energyKnown ? attacks.filter(a => a.payable).length / 10 : 0, legalKnown ? attacks.filter(a => a.legal).length / 10 : 0, 1];
  return { row, attacks };
}

export function encodeModel6Decision(view, options = {}) {
  const base = encodeEntityDecision(view, options), rows = [], seen = new Set();
  const add = card => {
    if (!card || seen.has(card.instanceId)) return;
    seen.add(card.instanceId); rows.push(card);
    for (const name of ['energy', 'tools', 'evolutions']) for (const child of card.publicCards?.[name] ?? []) add(child);
  };
  view.hand.forEach(add);
  for (const owner of ['you', 'opponent']) { add(view[owner].active); view[owner].bench.forEach(add); view[owner].discard.forEach(add); }
  add(view.stadium);
  const prompt = view.prompts[0];
  for (const card of prompt?.revealedCards ?? []) add(card);
  for (const choice of prompt?.choices ?? []) { add(choice.card); add(choice.sourceCard); }
  const context = options.context ?? options.decisionContext ?? view.decisionContext ?? {};
  add(prompt?.context?.originCard ?? (view.turn === 'you' ? context.originCard : null));
  if (rows.length !== base.entities.length) throw new Error('Model6 entity bindings differ from frozen base encoder');
  const facts = view.energyFacts;
  if (facts?.schema !== 'model6-energy/1' || !Array.isArray(facts.creatures) || !Array.isArray(facts.attachmentOptions))
    throw new Error('Model6 requires admitted native Energy facts');
  const indices = new Map(rows.map((card, index) => [card.instanceId, index]));
  const byId = new Map();
  for (const creature of facts.creatures) {
    if (!indices.has(creature.instanceId) || byId.has(creature.instanceId)) throw new Error('Energy facts refer to invisible or duplicate entity');
    byId.set(creature.instanceId, creature);
  }
  for (const owner of ['you', 'opponent']) for (const card of [view[owner].active, ...view[owner].bench].filter(Boolean))
    if (!byId.has(card.instanceId)) throw new Error('visible board Pokemon missing Energy facts');
  const energyAttacks = [], energyAttackEntities = [], energyAttackActionIds = [];
  const energyEntities = rows.map((card, index) => {
    const raw = byId.get(card.instanceId), encoded = creatureFacts(raw);
    encoded.attacks.forEach((attack, n) => {
      if (energyAttacks.length >= C.maxEnergyAttacks) throw new Error('complete attack table exceeds bound; truncation forbidden');
      energyAttacks.push(attack.row); energyAttackEntities.push(index);
      energyAttackActionIds.push(options.catalog.actionIds?.[raw.attacks[n].actionKey] ?? 0);
    });
    return encoded.row;
  });
  const offers = prompt?.choices ?? view.actions;
  const energyCandidateAttacks = [], energyCandidateAttackCandidates = [], energyCandidateAttackActionIds = [];
  const energyCandidates = offers.map(offer => {
    const matches = facts.attachmentOptions.filter(item => prompt
      ? item.promptId === prompt.id && item.choiceId === offer.id : item.actionId === offer.id);
    if (matches.length > 1) throw new Error('duplicate attachment counterfactual');
    const item = matches[0];
    if (!item || item.known !== true) return zeros(C.energyCandidateDimensions);
    if (!indices.has(item.sourceCardId) || !indices.has(item.targetId) || item.after?.instanceId !== item.targetId)
      throw new Error('attachment source and destination must bind visible entities');
    const candidateIndex = offers.indexOf(offer);
    if (base.candidateSources[candidateIndex] !== indices.get(item.sourceCardId)
        || base.candidateTargets[candidateIndex] !== indices.get(item.targetId)) throw new Error('counterfactual attachment binding mismatch');
    const after = creatureFacts(item.after);
    after.attacks.forEach((attack, n) => {
      if (energyCandidateAttacks.length >= C.maxEnergyCandidateAttacks) throw new Error('complete after-attachment attack table exceeds bound');
      energyCandidateAttacks.push(attack.row); energyCandidateAttackCandidates.push(candidateIndex);
      energyCandidateAttackActionIds.push(options.catalog.actionIds?.[item.after.attacks[n].actionKey] ?? 0);
    });
    return after.row;
  });
  for (const [values, dimension] of [[energyEntities, C.energyEntityDimensions], [energyCandidates, C.energyCandidateDimensions],
    [energyAttacks, C.energyAttackDimensions], [energyCandidateAttacks, C.energyAttackDimensions]])
    for (const row of values) if (row.length !== dimension || row.some(n => !Number.isFinite(n))) throw new Error('Model6 Energy tensor contract failed');
  return { ...base, schema: C.schema, energyEntities, energyCandidates, energyAttacks, energyAttackEntities, energyAttackActionIds,
    energyCandidateAttacks, energyCandidateAttackCandidates, energyCandidateAttackActionIds };
}
