import { supportsCardSelection } from '../integrations/local-worker/card-action-policy.mjs';

export const jointChoicePolicy = 'entity-ppo-choice-sets/1';
export const maximumChoiceSets = 128;

// Admit a complete domain or delegate the whole decision to the existing selector.
export function enumerateChoiceSets(view) {
  const prompt = view.prompts[0];
  if (view.status !== 'playing' || !supportsCardSelection(view) || prompt.cancelable
      || prompt.choices.some(c => c.minAmount != null || c.maxAmount != null || c.initialAmount != null))
    return { supported: false, reason: 'unsupported-choice-set-domain' };
  const n = prompt.choices.length;
  if (!Number.isSafeInteger(prompt.min) || !Number.isSafeInteger(prompt.max)
      || prompt.min < 0 || prompt.max < prompt.min || prompt.min > n)
    throw Error('invalid current choice-set cardinality');
  const sets = [];
  function combinations(count, start, chosen) {
    if (!count) { sets.push(chosen); return sets.length <= maximumChoiceSets; }
    for (let index = start; index <= n - count; index++)
      if (!combinations(count - 1, index + 1, [...chosen, index])) return false;
    return true;
  }
  for (let size = prompt.min; size <= Math.min(prompt.max, n); size++)
    if (!combinations(size, 0, [])) return { supported: false, reason: 'complete-choice-set-domain-exceeds-128' };
  return { supported: true, sets, minimum: prompt.min, maximum: Math.min(prompt.max, n) };
}

export function choiceSetDecision(view, encoded, index) {
  if (!Number.isSafeInteger(index) || !encoded.choiceSets?.[index] || !encoded.legalMask?.[index])
    throw Error('joint action outside validated complete choice sets');
  return { promptId: view.prompts[0].id,
    choiceIds: encoded.choiceSets[index].map(i => view.prompts[0].choices[i].id) };
}

export async function chooseValidatedChoiceSet(view, validate, actor, catalog, deterministic, encode) {
  const domain = enumerateChoiceSets(view);
  if (!domain.supported) return domain;
  if (typeof validate !== 'function') throw Error('current native choice-set validator required');
  const encoded = encode(view, { catalog });
  encoded.choiceSets = domain.sets;
  encoded.choiceSetMinimum = domain.minimum;
  encoded.choiceSetMaximum = domain.maximum;
  encoded.selectionCardinality = 'joint-unordered';
  encoded.legalMask = [];
  for (const indices of domain.sets)
    encoded.legalMask.push(await validate({ promptId: view.prompts[0].id,
      choiceIds: indices.map(i => view.prompts[0].choices[i].id) }) === true);
  if (!encoded.legalMask.some(Boolean)) throw Error('complete choice-set domain has no native-valid action');
  const [sample] = await actor.act([encoded], { deterministic });
  return { supported: true, encoded, sample, decision: choiceSetDecision(view, encoded, sample.actionIndex) };
}
