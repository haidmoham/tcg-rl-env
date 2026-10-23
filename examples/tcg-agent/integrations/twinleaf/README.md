# Public Twinleaf rules conformance

This integration executes real printed-card transitions from the public Twinleaf engine. It does not admit full Standard games into training. `assertStandardAdmission()` always rejects that request until card coverage, legal decisions, deterministic randomness and safe observations have independent evidence.

## Run

From the repository root with Git, Node 22 or 24, and npm available:

```sh
node scripts/twinleaf-prepare.mjs
node scripts/twinleaf-smoke.mjs
node scripts/twinleaf-audit.mjs
node --test tests/twinleaf.test.mjs
```

Preparation fetches only recognized public sources. It pins Twinleaf at `d89efe580127a05aff8b0456fd2ad587edbbee2d` and installs two exact npm packages with integrity hashes and lifecycle scripts disabled. It starts no server, database, client or training process. The main simulator requires none of these optional packages. Tests skip the external conformance case when this cache has not been prepared. The explicit smoke command fails if preparation is missing or the public source has changed.

The loader transpiles imported public TypeScript without typechecking. This is an isolated conformance tool, not a production build. It checks the origin URL, exact commit and clean source tree before loading. The engine's raw state and object-based prompts stay inside this privileged diagnostic process. No `info` object or engine reference is passed to a policy.

## Verified transitions

Seven targeted assertions use actual MEG Bulbasaur 1, Ivysaur 2, Scorbunny 26 and MEE basic Energy source classes:

- Bulbasaur evolves into Ivysaur, retains damage and Energy, and leaves the hand.
- Evolution during the turn the Basic was played is rejected.
- Two Fire Energy cannot pay Ivysaur's Grass Grass attack.
- Two Grass Energy pay Razor Leaf, which deals 60 and ends the turn.
- A controlled heads result lets Scorbunny deal 60 after Fire weakness.
- A controlled tails result makes Scorbunny deal zero.
- Knocking out the last opposing Pokémon ends the game with the correct winner.

These are constructed midgame states, not completed Standard deck games. Presentation-only `WaitPrompt` gates resolve immediately. Unknown prompts fail explicitly. The two coin tests temporarily replace `Math.random` within a synchronous diagnostic and restore it immediately. They do not establish seeded engine determinism, and must not run concurrently in one JavaScript context. Log timestamps also use wall-clock time. No throughput claim is made for this integration.

## Coverage evidence

`source-inventory.json` comes from the audit command. At this pin it finds 14,764 exported class declarations and 14,152 declarations with literal `set` and `setNumber` metadata. These are not unique verified card counts. Helpers, alternate printings and reprint subclasses can contribute entries. Inherited or computed metadata remains null. Static extraction is deliberately incomplete and cannot establish registry reachability or card correctness.

Each row records the public source path and SHA-256 hash, class, base class, literal card identity and regulation mark where present. Every row starts `script-present-unverified`. A source card's internal `fullName` can use a different suffix from its printed set metadata. For example, Bulbasaur uses set `MEG`, number `1`, but full name `Bulbasaur M1L`. Join catalog identities only through reviewed set aliases plus set number.

## Architecture decision

The candidate designs were to expand our three-card reducer by hand or to keep our verified small environment and use a pinned external engine for the broad rules transition boundary. The second design avoids copying thousands of private or unreviewed rules. It lets current learning experiments retain deterministic safe observations while public engine conformance grows independently. The public Twinleaf source is the only engine loaded here. No private Indigo Circuit source is copied or imported.

Twinleaf has a much broader historical and modern card corpus than the separate [ptcg-engine](https://github.com/gemelom/ptcg-engine) at `92c3cc4fe85a26f102d7bb6b3e8be7678512d2e5`. That Python engine has 104 card implementation modules plus `__init__.py` and five deck fixtures at the inspected pin. It offers a more direct seeded environment API and detached observations, but its raw actions still contain engine references. Its native Windows registry module path construction replaces `/` only and can skip card modules with backslashes. Neither engine is evidence of complete current Standard coverage.

Twinleaf's `Simulator` is useful prior art but its defaults force heads and preserve deck order. `clone()` does not preserve arbiter options. Randomness also appears outside its arbiter in coin-reflip and shuffle helpers. Its Store retains callback closures for pending prompts, so a serialized board alone cannot restore an in-progress decision. The adapter work still needs a complete decision algebra, safe detached views, injected per-game RNG, replayable chance events, prompt closure handling and tested deck admission.

## Licenses and contribution boundary

[Twinleaf's README](https://github.com/twinleafgg/twinleafgg/blob/d89efe580127a05aff8b0456fd2ad587edbbee2d/README.md) and server package metadata declare MIT. There is no standalone LICENSE in the inspected Git tree. The cache retains its upstream files. This repository redistributes inventory facts and original integration code, not the upstream engine source or artwork. Before redistributing a bundled engine, obtain and retain the full upstream copyright and license notice.

TypeScript 4.3.5 uses Apache-2.0. The installed `@progress/pako-esm` 1.0.1 `LICENSE.md` grants the MIT-style permission and retains Progress, Vitaly Puzrin and Andrei Tuputcyn notices. Both remain in their npm packages. Twinleaf explicitly rejects LLM-generated upstream contributions. This work does not submit patches there.

The next engine milestone is a seeded, serializable public decision protocol around this pin, followed by per-card interaction tests and versioned catalog admission. Unverified cards and prompt types must remain unavailable to training rather than produce plausible but incorrect games.
