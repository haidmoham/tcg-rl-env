const selectableKinds = new Set(['cards', 'pokemon', 'select']);

export function supportsCardSelection(view) {
  const prompt = view.prompts[0];
  if (!prompt || prompt.ordered || !selectableKinds.has(prompt.kind) || !prompt.choices.length || prompt.choices.length > 512
      || prompt.kind === 'allocation') return false;
  const visible = new Set([...view.hand, view.you.active, ...view.you.bench, view.opponent.active,
    ...view.opponent.bench, ...view.you.discard, ...view.opponent.discard].filter(Boolean).map(card => card.instanceId));
  return prompt.choices.every(choice => choice.card || visible.has(choice.instanceId));
}

export function* rankedPromptAnswers(view, ranking) {
  const prompt = view.prompts[0];
  if (!supportsCardSelection(view) || !Array.isArray(ranking) || ranking.length !== prompt.choices.length
      || new Set(ranking).size !== ranking.length || ranking.some(index => !Number.isSafeInteger(index) || index < 0 || index >= ranking.length))
    throw new Error('complete visible-card ranking required');
  const choices = ranking.map(index => prompt.choices[index]);
  const hand = new Set(view.hand.map(card => card.instanceId));
  const handCost = prompt.kind === 'cards' && !prompt.ordered && choices.every(choice => hand.has(choice.card?.instanceId ?? choice.instanceId));
  const groups = new Map((prompt.groups ?? []).map(group => [group.id, group]));
  const available = (start, group) => choices.slice(start).filter(choice => choice.group === group).length;
  const ungrouped = choices.filter(choice => !groups.has(choice.group)).length;
  const maximum = ungrouped + [...groups.values()].reduce((sum, group) => sum + Math.min(group.max, available(0, group.id)), 0);
  const minimum = Math.max(prompt.min, [...groups.values()].reduce((sum, group) => sum + group.min, 0));
  const preferred = handCost ? prompt.min : Math.min(prompt.max, choices.length, maximum);
  function* combinations(count, start = 0, selected = [], counts = new Map()) {
    if (choices.length - start < count) return;
    let mandatory = 0, capacity = choices.slice(start).filter(choice => !groups.has(choice.group)).length;
    for (const group of groups.values()) {
      const chosen = counts.get(group.id) ?? 0, remaining = available(start, group.id);
      if (chosen > group.max || chosen + remaining < group.min) return;
      mandatory += Math.max(0, group.min - chosen);
      capacity += Math.min(remaining, group.max - chosen);
    }
    if (mandatory > count || capacity < count) return;
    if (!count) { yield selected; return; }
    for (let index = start; index <= choices.length - count; index++) {
      const choice = choices[index], group = groups.get(choice.group), chosen = counts.get(choice.group) ?? 0;
      if (group && chosen >= group.max) continue;
      const next = new Map(counts);
      if (group) next.set(group.id, chosen + 1);
      yield* combinations(count - 1, index + 1, [...selected, choice], next);
    }
  }
  let examined = 0;
  for (let count = preferred; count >= minimum; count--) {
    for (const selected of combinations(count)) {
      if (++examined > 128) throw Object.assign(new Error('ranked card-selection search exceeded its candidate budget'), { code: 'DECISION_DOMAIN_BUDGET_EXCEEDED' });
      yield { promptId: prompt.id, choiceIds: selected.map(choice => choice.id) };
    }
  }
  if (prompt.cancelable) yield { promptId: prompt.id, choiceIds: [], cancel: true };
}

export function weakOrdinaryDecision(view) {
  const actions = view.actions.filter(action => action.kind !== 'concede');
  const selected = actions.find(action => action.kind === 'attack') ?? actions.find(action => action.kind === 'play')
    ?? actions.find(action => action.kind === 'pass') ?? actions[0];
  if (!selected) throw new Error('weak native baseline has no ordinary legal action');
  return { actionId: selected.id };
}

export function cardActionDiagnostics(views, spec = {}) {
  const initial = { ordinaryDecisions: 0, evolutionOpportunities: 0, chosenEvolutions: 0, evolutionOpportunitiesPassedByAttackOrPass: 0,
    reconOffered: 0, reconChosen: 0, phantomDiveCount: 0, firstPhantomDiveTurn: null, ordinaryModelDecisions: 0,
    selectionModelDecisions: 0, supportedCardSelections: 0, helperDecisions: 0, attacks: 0 };
  for (const entry of views) {
    const { view, decision, behaviorSource } = entry;
    if (entry.kind !== 'ordinary') {
      if (entry.kind === 'selection') initial.supportedCardSelections++;
      if (behaviorSource === 'model') initial.selectionModelDecisions++;
      else initial.helperDecisions++;
      continue;
    }
    initial.ordinaryDecisions++;
    if (behaviorSource === 'model') initial.ordinaryModelDecisions++;
    const action = view.actions.find(value => value.id === decision.actionId);
    const own = [view.you.active, ...view.you.bench].filter(Boolean);
    const evolutions = view.actions.filter(value => value.kind === 'play' && value.sourceCardId && value.targetId
      && (spec.evolutionPrintings ?? []).includes(view.hand.find(card => card.instanceId === value.sourceCardId)?.printingId)
      && own.some(card => card.instanceId === value.targetId));
    if (evolutions.length) initial.evolutionOpportunities++;
    if (evolutions.some(value => value.id === action?.id)) initial.chosenEvolutions++;
    else if (evolutions.length && ['attack', 'pass'].includes(action?.kind)) initial.evolutionOpportunitiesPassedByAttackOrPass++;
    if (view.actions.some(value => value.actionKey === spec.reconActionKey)) initial.reconOffered++;
    if (action?.actionKey === spec.reconActionKey) initial.reconChosen++;
    if (action?.kind === 'attack') initial.attacks++;
    if (action?.actionKey === spec.phantomDiveActionKey) {
      initial.phantomDiveCount++; initial.firstPhantomDiveTurn ??= view.turnNumber;
    }
  }
  return initial;
}
