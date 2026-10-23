export const PROFILE = 'basic-duel-v1';

export class UnsupportedMechanicError extends Error {
  constructor(message) { super(message); this.name = 'UnsupportedMechanicError'; }
}

export const CARDS = Object.freeze({
  'base1-52': Object.freeze({ name: 'Machop', kind: 'pokemon', hp: 50, type: 'fighting', weakness: 'psychic', resistance: null, retreat: 1, attack: Object.freeze({ name: 'Low Kick', cost: Object.freeze(['fighting']), damage: 20 }) }),
  'base1-61': Object.freeze({ name: 'Rattata', kind: 'pokemon', hp: 30, type: 'colorless', weakness: 'fighting', resistance: 'psychic', retreat: 0, attack: Object.freeze({ name: 'Bite', cost: Object.freeze(['colorless']), damage: 20 }) }),
  'base1-97': Object.freeze({ name: 'Fighting Energy', kind: 'energy', type: 'fighting' }),
});

export const BASELINE_DECK = Object.freeze([
  ...Array(4).fill('base1-52'), ...Array(4).fill('base1-61'), ...Array(52).fill('base1-97'),
]);

export function validateDeck(deck) {
  if (!Array.isArray(deck) || deck.length !== 60) throw new TypeError('a deck must contain exactly 60 card IDs');
  const counts = new Map();
  for (const id of deck) {
    if (typeof id !== 'string') throw new TypeError('card IDs must be primitive strings');
    if (!Object.hasOwn(CARDS, id)) throw new UnsupportedMechanicError(`card ${String(id)} is outside ${PROFILE}`);
    counts.set(id, (counts.get(id) ?? 0) + 1);
    if (CARDS[id].kind !== 'energy' && counts.get(id) > 4) throw new TypeError(`more than four copies of ${id}`);
  }
  if (!deck.some(id => CARDS[id].kind === 'pokemon')) throw new TypeError('a deck needs a Basic Pokemon');
  return [...deck];
}

export function importDeckCorpus(record) {
  if (record?.schema !== 'deck-corpus-v1' || !Array.isArray(record.cards)) throw new TypeError('expected deck-corpus-v1');
  const deck = [];
  for (const card of record.cards) {
    if (!Number.isSafeInteger(card.count) || card.count < 1 || card.count > 60) throw new TypeError('invalid card count');
    const id = `${card.set}-${card.number}`;
    if (!Object.hasOwn(CARDS, id)) throw new UnsupportedMechanicError(`unmapped printing ${id}`);
    deck.push(...Array(card.count).fill(id));
  }
  return validateDeck(deck);
}

export function attackDamage(attackerId, defenderId) {
  const attacker = CARDS[attackerId], defender = CARDS[defenderId];
  if (attacker?.kind !== 'pokemon' || defender?.kind !== 'pokemon') throw new TypeError('damage requires two admitted Pokemon');
  let damage = attacker.attack.damage;
  if (defender.weakness === attacker.type) damage *= 2;
  if (defender.resistance === attacker.type) damage = Math.max(0, damage - 30);
  return damage;
}
