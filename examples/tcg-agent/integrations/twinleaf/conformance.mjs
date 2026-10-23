import assert from 'node:assert/strict';
import { loadPublicRules, twinleafPin } from './public-engine.mjs';

export function runTwinleafConformance() {
  const engine = loadPublicRules();
  try {
    const { Store } = engine.load('game/store/store.ts');
    const { State, GamePhase, GameWinner } = engine.load('game/store/state/state.ts');
    const { Player } = engine.load('game/store/state/player.ts');
    const { CardList } = engine.load('game/store/state/card-list.ts');
    const { PokemonCardList } = engine.load('game/store/state/pokemon-card-list.ts');
    const { Bulbasaur } = engine.load('sets/11-mega-evolution/set-mega-evolution/bulbasaur.ts');
    const { Ivysaur } = engine.load('sets/11-mega-evolution/set-mega-evolution/ivysaur.ts');
    const { Scorbunny } = engine.load('sets/11-mega-evolution/set-mega-evolution/scorbunny.ts');
    const { GrassEnergy, FireEnergy } = engine.load('sets/11-mega-evolution/set-mega-evolution-energy/basic-energies.ts');
    const { PlayCardAction, PlayerType, SlotType } = engine.load('game/store/actions/play-card-action.ts');
    const { AttackAction } = engine.load('game/store/actions/game-actions.ts');
    const { ResolvePromptAction } = engine.load('game/store/actions/resolve-prompt-action.ts');
    let cardId = 0;
    const card = CardClass => Object.assign(new CardClass(), { id: cardId++ });
    const fixture = (first = Bulbasaur, second = Bulbasaur) => {
      const state = new State();
      state.phase = GamePhase.PLAYER_TURN;
      state.turn = 3;
      state.players = [first, second].map((CardClass, index) => {
        const player = new Player();
        player.id = index + 1;
        player.active.cards = [card(CardClass)];
        player.active.hp = player.active.getPokemonCard().hp;
        player.active.pokemonPlayedTurn = 0;
        player.active.isPublic = true;
        player.bench = Array.from({ length: 5 }, () => new PokemonCardList());
        player.deck.cards = Array.from({ length: 20 }, () => card(GrassEnergy));
        player.prizes = Array.from({ length: 6 }, () => {
          const prize = new CardList();
          prize.cards.push(card(GrassEnergy));
          prize.isSecret = true;
          return prize;
        });
        return player;
      });
      const store = new Store({ onStateChange() {} });
      store.state = state;
      return store;
    };
    const target = { player: PlayerType.BOTTOM_PLAYER, slot: SlotType.ACTIVE, index: 0 };
    const settle = (store, coin = true) => {
      for (let count = 0; count < 30; count++) {
        const prompt = store.state.prompts.find(item => item.result === undefined);
        if (!prompt) return;
        let raw;
        if (prompt.type === 'WaitPrompt') raw = true;
        else if (prompt.type === 'Coin flip') raw = coin;
        else if (prompt.type === 'Choose prize') raw = [0];
        else throw new Error(`UNSUPPORTED_CONFORMANCE_PROMPT: ${prompt.type}`);
        const decoded = prompt.decode(raw, store.state);
        assert.equal(prompt.validate(decoded, store.state), true);
        store.dispatch(new ResolvePromptAction(prompt.id, decoded));
      }
      throw new Error('CONFORMANCE_PROMPT_BUDGET_EXCEEDED');
    };
    const passed = [];
    const check = (name, run) => { run(); passed.push(name); };
    const withControlledCoin = (heads, run) => {
      const random = Math.random;
      Math.random = () => heads ? 0.25 : 0.75;
      try { run(); } finally { Math.random = random; }
    };
    check('MEG 1 to MEG 2 evolves, retains damage and attached energy, removes hand card', () => {
      const store = fixture();
      const player = store.state.players[0];
      player.active.damage = 20;
      player.active.cards.push(card(GrassEnergy));
      player.hand.cards.push(card(Ivysaur));
      store.dispatch(new PlayCardAction(1, 0, target));
      assert.equal(store.state.players[0].active.getPokemonCard().name, 'Ivysaur');
      assert.equal(store.state.players[0].active.damage, 20);
      assert.equal(store.state.players[0].active.cards.length, 3);
      assert.equal(store.state.players[0].hand.cards.length, 0);
    });
    check('same-turn evolution is rejected', () => {
      const store = fixture();
      store.state.players[0].active.pokemonPlayedTurn = 3;
      store.state.players[0].hand.cards.push(card(Ivysaur));
      assert.throws(() => store.dispatch(new PlayCardAction(1, 0, target)), error => error.message === 'POKEMON_CANT_EVOLVE_THIS_TURN');
      assert.equal(store.state.players[0].active.getPokemonCard().name, 'Bulbasaur');
    });
    check('Ivysaur Razor Leaf rejects two wrong-type Energy', () => {
      const store = fixture(Ivysaur);
      store.state.players[0].active.cards.push(card(FireEnergy), card(FireEnergy));
      assert.throws(() => store.dispatch(new AttackAction(1, 'Razor Leaf')), error => error.message === 'NOT_ENOUGH_ENERGY');
      assert.equal(store.state.players[1].active.damage, 0);
    });
    check('Ivysaur Razor Leaf pays Grass Grass, deals 60, then passes turn', () => {
      const store = fixture(Ivysaur);
      store.state.players[0].active.cards.push(card(GrassEnergy), card(GrassEnergy));
      store.dispatch(new AttackAction(1, 'Razor Leaf'));
      settle(store);
      assert.equal(store.state.players[1].active.damage, 60);
      assert.equal(store.state.activePlayer, 1);
    });
    check('Scorbunny Wild Kick controlled heads applies Fire weakness to Bulbasaur', () => {
      const store = fixture(Scorbunny);
      store.state.players[0].active.cards.push(card(FireEnergy));
      withControlledCoin(true, () => {
        store.dispatch(new AttackAction(1, 'Wild Kick'));
        settle(store, true);
      });
      assert.equal(store.state.players[1].active.damage, 60);
    });
    check('Scorbunny Wild Kick controlled tails deals no damage', () => {
      const store = fixture(Scorbunny);
      store.state.players[0].active.cards.push(card(FireEnergy));
      withControlledCoin(false, () => {
        store.dispatch(new AttackAction(1, 'Wild Kick'));
        settle(store, false);
      });
      assert.equal(store.state.players[1].active.damage, 0);
    });
    check('knockout of last opposing Pokemon ends game with correct winner', () => {
      const store = fixture(Ivysaur);
      store.state.players[0].active.cards.push(card(GrassEnergy), card(GrassEnergy));
      store.state.players[1].active.damage = 30;
      store.dispatch(new AttackAction(1, 'Razor Leaf'));
      settle(store);
      assert.equal(store.state.phase, GamePhase.FINISHED);
      assert.equal(store.state.winner, GameWinner.PLAYER_1);
      assert.equal(store.state.players[1].active.cards.length, 0);
    });
    return { schemaVersion: 1, engineCommit: twinleafPin.commit, mode: 'privileged-rules-conformance',
      completeStandard: false, seededGames: false, tests: passed, count: passed.length };
  } finally {
    engine.close();
  }
}
