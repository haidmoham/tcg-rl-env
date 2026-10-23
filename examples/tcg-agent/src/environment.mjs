import { CARDS, PROFILE, BASELINE_DECK, validateDeck, attackDamage, UnsupportedMechanicError } from './cards.mjs';
import { Random } from './rng.mjs';

const pokemon = id => ({ id, damage: 0, energy: 0 });
const publicPokemon = p => p === null ? null : ({ card: p.id, damage: p.damage, energy: p.energy, hp: CARDS[p.id].hp });
const uniqueHand = (hand, kind) => hand.flatMap((id, index) => CARDS[id].kind === kind && hand.indexOf(id) === index ? [{ id, index }] : []);
const removeAt = (items, index) => items.splice(index, 1)[0];

export class IllegalActionError extends Error {
  constructor(message) { super(message); this.name = 'IllegalActionError'; }
}

export class Environment {
  #players; #seed; #decks; #phase; #actor; #first; #turn; #attached; #retreated;
  #winner; #reason; #revision; #actions; #events; #record; #setupStage; #bonus;
  #rng; #bonusBasics;

  constructor({ seed = 1, decks = [BASELINE_DECK, BASELINE_DECK], profile = PROFILE, record = true } = {}) {
    if (profile !== PROFILE) throw new UnsupportedMechanicError(`unknown profile ${profile}`);
    if (!Array.isArray(decks) || decks.length !== 2) throw new TypeError('exactly two decks required');
    this.#decks = decks.map(validateDeck);
    this.#rng = new Random(seed);
    this.#seed = seed;
    this.#record = record;
    this.#events = []; this.#actions = []; this.#revision = 0;
    this.#winner = null; this.#reason = null; this.#turn = 0;
    this.#first = null; this.#bonusBasics = [{}, {}];
    this.#players = this.#decks.map(deck => ({ deck: [...deck], hand: [], active: null, bench: [], prizes: [], discard: [], mulligans: 0 }));
    this.#bonus = [0, 0];
    this.#actor = this.#rng.int(2); this.#phase = 'choose_first'; this.#setupStage = 0;
    this.#emit('coin', { winner: this.#actor });
  }

  #dealOpeningHands() {
    this.#players = this.#decks.map(deck => {
      let shuffled, hand, mulligans = -1;
      do {
        if (++mulligans >= 10000) throw new Error('mulligan safety limit exceeded');
        shuffled = this.#rng.shuffle(deck); hand = shuffled.splice(0, 7);
      } while (!hand.some(id => CARDS[id].kind === 'pokemon'));
      return { deck: shuffled, hand, active: null, bench: [], prizes: [], discard: [], mulligans };
    });
    this.#bonus = this.#players.map((p, i) => Math.max(0, this.#players[1 - i].mulligans - p.mulligans));
    this.#emit('mulligans', { counts: this.#players.map(p => p.mulligans) });
  }

  get currentPlayer() { return this.#actor; }
  get terminated() { return this.#phase === 'terminal'; }
  get outcome() { return { terminated: this.terminated, winner: this.#winner, reason: this.#reason, turns: this.#turn, reward: this.#winner === null ? [0, 0] : this.#winner === 0 ? [1, -1] : [-1, 1] }; }

  legalActions() {
    if (this.terminated) return [];
    const p = this.#players[this.#actor];
    let actions;
    switch (this.#phase) {
      case 'choose_first': actions = [0, 1].map(player => ({ kind: 'choose_first', player })); break;
      case 'active': actions = uniqueHand(p.hand, 'pokemon').map(({ index }) => ({ kind: 'active', hand: index })); break;
      case 'bench':
        actions = [{ kind: 'ready' }];
        if (p.bench.length < 5) actions.push(...uniqueHand(p.hand, 'pokemon')
          .filter(({ id }) => this.#setupStage === 0 || this.#bonusBasics[this.#actor][id] > 0)
          .map(({ index }) => ({ kind: 'bench', hand: index })));
        break;
      case 'bonus': actions = Array.from({ length: Math.min(this.#bonus[this.#actor], p.deck.length) + 1 }, (_, count) => ({ kind: 'bonus', count })); break;
      case 'prize': actions = p.prizes.map((_, prize) => ({ kind: 'prize', prize })); break;
      case 'promote': actions = p.bench.map((_, bench) => ({ kind: 'promote', bench })); break;
      case 'turn': {
        actions = [{ kind: 'pass' }];
        if (p.bench.length < 5) actions.push(...uniqueHand(p.hand, 'pokemon').map(({ index }) => ({ kind: 'bench', hand: index })));
        const energy = p.hand.indexOf('base1-97');
        if (!this.#attached && energy >= 0) {
          actions.push({ kind: 'attach', hand: energy, target: -1 });
          p.bench.forEach((_, target) => actions.push({ kind: 'attach', hand: energy, target }));
        }
        if (this.#turn > 1 && p.active.energy >= CARDS[p.active.id].attack.cost.length) actions.push({ kind: 'attack' });
        if (!this.#retreated && p.active.energy >= CARDS[p.active.id].retreat) p.bench.forEach((_, bench) => actions.push({ kind: 'retreat', bench }));
        break;
      }
      default: throw new UnsupportedMechanicError(`unhandled phase ${this.#phase}`);
    }
    return actions.map(action => ({ ...action, revision: this.#revision }));
  }

  observe(player = this.#actor) {
    if (player !== 0 && player !== 1) throw new TypeError('player must be 0 or 1');
    const visibleBoard = this.#turn > 0;
    return {
      schema: 'observation-v1', profile: PROFILE, player, actor: this.#actor, firstPlayer: this.#first, phase: this.#phase, turn: this.#turn, revision: this.#revision,
      players: this.#players.map((p, i) => ({
        handCount: p.hand.length, deckCount: p.deck.length, prizeCount: p.prizes.length,
        hand: i === player ? [...p.hand] : undefined,
        active: i === player || visibleBoard ? publicPokemon(p.active) : null,
        bench: i === player || visibleBoard ? p.bench.map(publicPokemon) : [],
        discard: [...p.discard], mulligans: p.mulligans,
      })),
      actions: player === this.#actor ? this.legalActions() : [],
      outcome: this.outcome,
    };
  }

  step(action) {
    if (this.terminated) throw new IllegalActionError('game is terminal');
    if (action === null || typeof action !== 'object' || Array.isArray(action)) throw new IllegalActionError('action must be an object');
    const legal = this.legalActions().find(candidate => {
      const keys = Object.keys(candidate);
      return keys.length === Object.keys(action).length && keys.every(key => candidate[key] === action[key]);
    });
    if (!legal) throw new IllegalActionError('action is illegal or has a stale revision');
    const actor = this.#actor, p = this.#players[actor], other = this.#players[1 - actor];
    if (this.#record) this.#actions.push({ ...legal });
    this.#revision++;
    switch (legal.kind) {
      case 'choose_first':
        this.#first = legal.player; this.#emit('first_player', { player: legal.player });
        this.#dealOpeningHands(); this.#actor = 0; this.#phase = 'active';
        break;
      case 'active':
        p.active = pokemon(removeAt(p.hand, legal.hand));
        if (actor === 0) this.#actor = 1;
        else { this.#actor = 0; this.#phase = 'bench'; }
        break;
      case 'bench': {
        const id = removeAt(p.hand, legal.hand); p.bench.push(pokemon(id));
        if (this.#turn === 0 && this.#setupStage === 1) this.#bonusBasics[actor][id]--;
        if (this.#turn > 0) this.#emit('bench', { player: actor, card: id });
        break;
      }
      case 'ready':
        if (actor === 0) this.#actor = 1;
        else if (this.#setupStage === 0) {
          for (const player of this.#players) player.prizes = player.deck.splice(0, 6);
          this.#actor = 0; this.#phase = 'bonus';
        } else {
          this.#emit('reveal', { players: this.#players.map(player => ({ active: player.active.id, bench: player.bench.map(slot => slot.id) })) });
          this.#beginTurn(this.#first);
        }
        break;
      case 'bonus': {
        const drawn = p.deck.splice(0, legal.count);
        p.hand.push(...drawn);
        for (const id of drawn) if (CARDS[id].kind === 'pokemon') this.#bonusBasics[actor][id] = (this.#bonusBasics[actor][id] ?? 0) + 1;
        this.#emit('bonus', { player: actor, count: legal.count });
        if (actor === 0) this.#actor = 1;
        else { this.#actor = 0; this.#phase = 'bench'; this.#setupStage = 1; }
        break;
      }
      case 'attach': {
        removeAt(p.hand, legal.hand);
        const slot = legal.target === -1 ? p.active : p.bench[legal.target];
        slot.energy++; this.#attached = true;
        this.#emit('attach', { player: actor, target: legal.target, card: 'base1-97' });
        break;
      }
      case 'retreat': {
        const cost = CARDS[p.active.id].retreat;
        p.active.energy -= cost; p.discard.push(...Array(cost).fill('base1-97'));
        [p.active, p.bench[legal.bench]] = [p.bench[legal.bench], p.active]; this.#retreated = true;
        this.#emit('retreat', { player: actor, bench: legal.bench, cost });
        break;
      }
      case 'attack': {
        const damage = attackDamage(p.active.id, other.active.id);
        other.active.damage += damage;
        this.#emit('attack', { player: actor, card: p.active.id, damage });
        if (other.active.damage >= CARDS[other.active.id].hp) {
          other.discard.push(other.active.id, ...Array(other.active.energy).fill('base1-97'));
          this.#emit('knockout', { player: 1 - actor, card: other.active.id });
          other.active = null; this.#phase = 'prize';
        } else this.#beginTurn(1 - actor);
        break;
      }
      case 'prize':
        p.hand.push(removeAt(p.prizes, legal.prize)); this.#emit('prize', { player: actor, remaining: p.prizes.length });
        if (p.prizes.length === 0) this.#finish(actor, 'prizes');
        else if (other.bench.length === 0) this.#finish(actor, 'no_pokemon');
        else { this.#actor = 1 - actor; this.#phase = 'promote'; }
        break;
      case 'promote': p.active = removeAt(p.bench, legal.bench); this.#emit('promote', { player: actor, card: p.active.id }); this.#beginTurn(actor); break;
      case 'pass': this.#emit('pass', { player: actor }); this.#beginTurn(1 - actor); break;
      default: throw new UnsupportedMechanicError(`unhandled action ${legal.kind}`);
    }
    return this.outcome;
  }

  #beginTurn(player) {
    this.#actor = player; this.#turn++; this.#phase = 'turn'; this.#attached = false; this.#retreated = false;
    const p = this.#players[player];
    if (p.deck.length === 0) { this.#finish(1 - player, 'deckout'); return; }
    p.hand.push(p.deck.shift());
    this.#emit('draw', { player, count: 1, turn: this.#turn });
  }

  #finish(winner, reason) { this.#winner = winner; this.#reason = reason; this.#phase = 'terminal'; this.#emit('terminal', { winner, reason }); }
  #emit(type, fields) { if (this.#record) this.#events.push({ type, ...fields }); }

  publicEvents() { return structuredClone(this.#events); }

  replayRecord() {
    if (!this.#record) throw new Error('recording disabled');
    return { schema: 'replay-v1', engine: '0.1.0', profile: PROFILE, seed: this.#seed, decks: this.#decks.map(deck => [...deck]), actions: this.#actions.map(action => ({ ...action })), events: this.publicEvents(), outcome: this.outcome };
  }
}

export function replay(record) {
  if (record?.schema !== 'replay-v1' || record.engine !== '0.1.0') throw new TypeError('unsupported replay version');
  const env = new Environment({ seed: record.seed, decks: record.decks, profile: record.profile });
  for (const action of record.actions) env.step(action);
  if (JSON.stringify(env.replayRecord()) !== JSON.stringify(record)) throw new Error('replay diverged');
  return env;
}

export function stepBatch(environments, actions) {
  if (environments.length !== actions.length) throw new TypeError('batch lengths differ');
  return environments.map((env, index) => env.step(actions[index]));
}
