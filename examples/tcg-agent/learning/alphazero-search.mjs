import { enumerateChoiceSets } from './legal-choice-sets.mjs';
import { encodeModel6Decision } from './model6-features.mjs';

const unsupported = reason => Object.assign(Error(reason), { code: 'UNSUPPORTED_SEARCH_DOMAIN' });
const finite = value => typeof value === 'number' && Number.isFinite(value);

export function createSearchEvaluator({ model, catalog }) {
  return async ({ view, validate, leafOnly = false }) => {
    if (view.opponent?.hand != null || view.opponent?.deck != null) throw Error('actor-visible search observation required');
    if (view.status !== 'playing') throw unsupported('setup is delegated to the existing helper');
    const prompt = view.prompts[0];
    if (prompt && prompt.choices.length === 0 && prompt.min === 0 && prompt.max === 0) {
      const forcedDecision = { promptId: prompt.id, choiceIds: [] };
      if (typeof validate !== 'function' || await validate(forcedDecision) !== true)
        throw Error('empty forced prompt requires native validation');
      return { forcedDecision };
    }
    const encoded = encodeModel6Decision(view, { catalog });
    if (leafOnly) {
      if (encoded.kind === 'selection') encoded.selectionCardinality = 1;
      const result = await model.score([encoded]);
      return { commands: [], logits: [], value: result.values[0], encoded, leafOnly: true };
    }
    let commands;
    if (view.prompts.length) {
      const domain = enumerateChoiceSets(view);
      if (!domain.supported) throw unsupported(domain.reason);
      if (typeof validate !== 'function') throw Error('native search prompt validator required');
      encoded.choiceSets = domain.sets;
      encoded.choiceSetMinimum = domain.minimum;
      encoded.choiceSetMaximum = domain.maximum;
      encoded.selectionCardinality = 'joint-unordered';
      const proposed = domain.sets.map(indices => ({ promptId: view.prompts[0].id,
        choiceIds: indices.map(index => view.prompts[0].choices[index].id) }));
      encoded.legalMask = [];
      for (const command of proposed) encoded.legalMask.push(await validate(command) === true);
      if (!encoded.legalMask.some(Boolean)) throw Error('complete search prompt domain has no native-valid answer');
      commands = proposed;
    } else {
      commands = view.actions.map(action => ({ actionId: action.id }));
      if (!commands.length) throw unsupported('no ordinary search actions');
      encoded.legalMask = commands.map(() => true);
    }
    const result = await model.score([encoded]);
    const logits = result.logits?.[0], value = result.values?.[0];
    if (!Array.isArray(logits) || logits.length !== commands.length || !finite(value) || Math.abs(value) > 1.000001
        || logits.some((score, index) => encoded.legalMask[index] && !finite(score)))
      throw Error('search evaluator output differs from complete action domain');
    return { commands, logits: logits.map((score, index) => encoded.legalMask[index] ? score : -Infinity), value, encoded };
  };
}

function randomStream(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ state >>> 15, 1 | state);
    value ^= value + Math.imul(value ^ value >>> 7, 61 | value);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}

function gammaSample(alpha, random) {
  if (alpha < 1) return gammaSample(alpha + 1, random) * Math.max(random(), Number.MIN_VALUE) ** (1 / alpha);
  const d = alpha - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    const x = Math.sqrt(-2 * Math.log(Math.max(random(), Number.MIN_VALUE))) * Math.cos(2 * Math.PI * random());
    const t = 1 + c * x;
    if (t <= 0) continue;
    const v = t ** 3, u = Math.max(random(), Number.MIN_VALUE);
    if (u < 1 - .0331 * x ** 4 || Math.log(u) < .5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function probabilities(logits) {
  const maximum = Math.max(...logits);
  if (!finite(maximum)) throw Error('search requires at least one finite legal prior');
  const weights = logits.map(value => value === -Infinity ? 0 : Math.exp(value - maximum));
  const total = weights.reduce((sum, value) => sum + value, 0);
  return weights.map(value => value / total);
}

export async function runInformationSetSearch({ provider, rootView, publicHistory, knownDecks, rootPlayer,
  seed, evaluate, determinizations = 2, simulations = 8, maxDepth = 8, cPuct = 1.25,
  temperature = 1, rootNoise = true, signal }) {
  if (!['human', 'opponent'].includes(rootPlayer) || !Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff
      || !Number.isSafeInteger(determinizations) || determinizations < 1 || determinizations > 8
      || !Number.isSafeInteger(simulations) || simulations < 1 || simulations > 128
      || !Number.isSafeInteger(maxDepth) || maxDepth < 1 || maxDepth > 32
      || !finite(cPuct) || cPuct <= 0 || ![0, 1].includes(temperature) || typeof rootNoise !== 'boolean')
    throw Error('bounded explicit information-set search configuration required');
  const random = randomStream(seed), handles = [], informationSets = new Map();
  let commands, encoded, canonical, rootNode, expansions = 0, sharedInformationSetHits = 0, valueTotal = 0, forcedTransitions = 0;
  const check = () => { if (signal?.aborted) throw Object.assign(Error('search cancelled'), { code: 'SEARCH_CANCELLED' }); };
  const makeNode = async (handle, histories, root = false, forcedDepth = 0) => {
    check();
    const observation = await provider.observe(handle);
    const { view, toPlay, terminal } = observation;
    if (terminal != null) {
      if (!('winner' in terminal) || terminal.winner !== null && !['human', 'opponent'].includes(terminal.winner))
        throw Error('explicit natural terminal winner required');
      return { handle, histories, terminal: true, value: terminal.winner === null ? 0 : terminal.winner === rootPlayer ? 1 : -1 };
    }
    if (!['human', 'opponent'].includes(toPlay)) throw Error('search state actor missing');
    if (root && toPlay !== rootPlayer) throw Error('determinized root changed acting player');
    let evaluation;
    try {
      evaluation = await evaluate({ view, toPlay, validate: command => provider.validate(handle, command) });
    } catch (error) {
      if (root || error.code !== 'UNSUPPORTED_SEARCH_DOMAIN') throw error;
      evaluation = await evaluate({ view, toPlay, leafOnly: true });
    }
    if (evaluation.forcedDecision) {
      if (root) throw unsupported('forced empty root has no search policy target');
      if (forcedDepth >= 32) throw Error('forced native transition bound exceeded');
      const childHandle = await provider.step(handle, evaluation.forcedDecision);
      handles.push(childHandle); forcedTransitions++;
      const nextHistories = { ...histories, [toPlay]: [...histories[toPlay],
        { observation: view, forcedAction: evaluation.forcedDecision }] };
      return makeNode(childHandle, nextHistories, false, forcedDepth + 1);
    }
    if (!finite(evaluation.value) || Math.abs(evaluation.value) > 1.000001) throw Error('bounded actor-perspective search value required');
    if (evaluation.leafOnly) return { handle, histories, terminal: true, value: evaluation.value * (toPlay === rootPlayer ? 1 : -1) };
    // World handles, chance seeds and the other player's private history never key policy statistics.
    const key = JSON.stringify({ toPlay, history: histories[toPlay], encoded: evaluation.encoded, commands: evaluation.commands });
    const nextHistories = { ...histories, [toPlay]: [...histories[toPlay], { observation: evaluation.encoded }] };
    const existing = informationSets.get(key);
    if (existing) {
      sharedInformationSetHits++;
      return { handle, histories: nextHistories, node: existing, value: existing.value, fresh: false };
    }
    const priors = probabilities(evaluation.logits);
    if (root) {
      const identity = JSON.stringify({ commands: evaluation.commands, mask: evaluation.encoded.legalMask });
      if (canonical !== undefined && identity !== canonical) throw Error('hidden sample changed root observable legal domain');
      canonical = identity; commands = evaluation.commands; encoded = evaluation.encoded;
      if (rootNoise) {
        const noise = priors.map(prior => prior > 0 ? gammaSample(.3, random) : 0);
        const total = noise.reduce((sum, value) => sum + value, 0);
        for (let index = 0; index < priors.length; index++) priors[index] = .75 * priors[index] + .25 * noise[index] / total;
      }
    }
    expansions++;
    const node = { toPlay, value: evaluation.value * (toPlay === rootPlayer ? 1 : -1), visits: 0,
      edges: evaluation.commands.map((command, index) => ({ index, command, prior: priors[index], visits: 0, total: 0 })) };
    informationSets.set(key, node);
    return { handle, histories: nextHistories, node, value: node.value, fresh: true };
  };
  try {
    for (let world = 0; world < determinizations; world++) {
      check();
      const handle = await provider.begin({ rootView, publicHistory, knownDecks, seed: (seed + Math.imul(world, 0x9e3779b9)) >>> 0 });
      handles.push(handle);
      const root = await makeNode(handle, { [rootPlayer]: publicHistory, [rootPlayer === 'human' ? 'opponent' : 'human']: [] }, true);
      if (root.terminal) throw unsupported('search requested at terminal root');
      rootNode ??= root.node;
      if (root.node !== rootNode) throw Error('hidden sample changed root information set');
      const transitions = new Map([['', root]]);
      for (let simulation = 0; simulation < simulations; simulation++) {
        check();
        let current = root, value = root.value, pathKey = '';
        const path = [];
        for (let depth = 0; depth < maxDepth; depth++) {
          let best, maximum = -Infinity;
          const node = current.node;
          for (const edge of node.edges) {
            if (!edge.prior) continue;
            const q = edge.visits ? edge.total / edge.visits : node.value;
            const score = (node.toPlay === rootPlayer ? q : -q)
              + cPuct * edge.prior * Math.sqrt(node.visits + 1) / (1 + edge.visits);
            if (score > maximum) { best = edge; maximum = score; }
          }
          if (!best) throw Error('no legal search edge');
          path.push({ node, edge: best });
          pathKey += `/${best.index}`;
          let child = transitions.get(pathKey);
          if (!child) {
            const childHandle = await provider.step(current.handle, best.command); handles.push(childHandle);
            const histories = { ...current.histories,
              [node.toPlay]: [...current.histories[node.toPlay], { actionIndex: best.index }] };
            child = await makeNode(childHandle, histories);
            transitions.set(pathKey, child);
            value = child.value;
            if (child.fresh || child.terminal) break;
          }
          current = child; value = child.value;
          if (current.terminal) break;
        }
        for (const { node: visited, edge } of path) { visited.visits++; edge.visits++; edge.total += value; }
        valueTotal += value;
      }
    }
    const visits = rootNode.edges.map(edge => edge.visits);
    const total = visits.reduce((sum, value) => sum + value, 0);
    const policyTarget = visits.map(value => value / total);
    let selectedIndex = visits.indexOf(Math.max(...visits));
    if (temperature === 1) {
      let threshold = random();
      for (let index = 0; index < policyTarget.length; index++) {
        threshold -= policyTarget[index];
        if (threshold < 0) { selectedIndex = index; break; }
      }
    }
    return { commands, encoded, visits, policyTarget, selectedIndex,
      decision: commands[selectedIndex], rootValue: valueTotal / total,
      diagnostics: { determinizations, simulationsPerWorld: simulations, totalSimulations: total, expansions, maxDepth,
        informationSets: informationSets.size, sharedInformationSetHits, forcedTransitions,
        approximation: 'declared-deck-information-set-puct/1', seed, temperature, rootNoise } };
  } finally {
    const results = await Promise.allSettled(handles.reverse().map(handle => provider.release(handle)));
    if (results.some(result => result.status === 'rejected')) throw Error('search branch release failed');
  }
}
