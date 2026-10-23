# Run and reuse the exported work

Use Python 3.10+ for the main package, Node 22 or 24 for the public example.
The cloned source tree includes the JavaScript example; the Python wheel
provides the `tcg` package. The source distribution also includes the recipes,
manifests, examples and attribution. Read [PROVENANCE.md](PROVENANCE.md) before
collecting or redistributing external inputs.

## Offline package and complete-game regression kernel

From the repository root:

```sh
python scripts/verify_template.py
python -m pip install .
cd examples/tcg-agent
node --test tests/*.test.mjs
node scripts/demo.mjs
```

Without optional preparation, the Twinleaf external transition test skips.
All other JavaScript tests require only Node. The original kernel covers three
card definitions, seeded complete games, replay, legal actions, actor visibility,
censoring and paired evaluation. It does not admit arbitrary cards.

## Retrieve all pinned public card source and run existing conformance checks

Inside `examples/tcg-agent`, with Git, Node and npm available:

```sh
node scripts/twinleaf-prepare.mjs
node scripts/twinleaf-smoke.mjs
node --test tests/twinleaf.test.mjs
node scripts/twinleaf-audit.mjs .cache/source-inventory.json
```

Preparation retrieves the pinned public Twinleaf source tree into ignored
`.cache/twinleaf` and installs two locked packages with lifecycle scripts
disabled. It preserves upstream files, checks the exact origin/revision/clean
tree, and starts no server, database or game client. Review the upstream terms
linked in PROVENANCE.md; retrieval does not grant additional rights.

The collected inventory contains **14,764 class declarations**, of which
**14,152 have literal set/number fields**. These are source entries, including
helpers and alternate printings, not unique cards or parity-verified cards.
Inherited/computed fields can be null. Every row remains
`script-present-unverified`, even for classes exercised by the small conformance
tool. This avoids promoting partial behavior evidence to complete card parity.

The unchanged conformance tool makes **seven targeted assertions** for MEG
Bulbasaur, Ivysaur, Scorbunny and MEE basic Grass/Fire Energy: evolution and its
timing restriction, typed attack costs, damage/turn end, controlled coin outcomes
and final-Pokemon knockout. These are constructed transitions, not full deck
games, exhaustive interactions, a native-to-Live oracle comparison or a seeded
Twinleaf environment. Unverified Standard admission deliberately throws.
See the retained [integration guide](examples/tcg-agent/integrations/twinleaf/README.md).

## Reuse learning components

The example's `learning/` directory includes original candidate and entity
scorers, behavioral cloning, masked/entity PPO, GAE, joint choice sets,
information-set search, search-policy/value learning, checkpoint and runtime
utilities. Feature schemas and source-hash checks remain intact. No historical
checkpoint or native game corpus is supplied.

With an independently installed compatible PyTorch runtime, run the existing
bounded CPU mathematical/behavioral checks from `examples/tcg-agent`:

```sh
python -m unittest discover -s learning -p 'test_*.py'
```

These tests create synthetic data and temporary checkpoints. They do not train
a playing-strength model or use the restricted engine. Choose a writable temp
directory when running in a Windows sandbox. This export does not prescribe an
old historical Torch pin; the verification receipt records the tested runtime.

The main `tcg/` package also includes canonical observation/decision, rollout,
PPO, checkpoint/population and native adaptation components. Existing
`scripts/qualify_*.py`, `check_live_parity.py` and `tcg/*parity.py` preserve the
native parity tools. Inspect CLI help before use. They need exact user-supplied
licensed inputs and fixture binaries. No lawful access to those inputs is
promised; do not download or run competition-only artifacts outside their terms.
The generic Python package imports without loading an engine.

## Interpret coverage

| Evidence | What it establishes | What remains unverified |
| --- | --- | --- |
| Kernel's unchanged tests | Exercised three-card regression behavior, replay and visible policy contracts | Broader card rules and current Standard |
| Seven Twinleaf assertions | Those constructed public-engine transitions | Complete card behavior, arbitrary prompts, determinism, full games and Live parity |
| 14,764 static source declarations | Presence and source identity | Registry reachability, uniqueness, correctness, legality and gameplay parity |
| Python learning tests | Exercised mathematical/runtime/storage contracts on synthetic inputs | Playing strength, native rollout parity and training readiness |
| `coverage/current.json` | Historical retained aggregate: 8,010 of 21,920 declared cases matched | 13,909 unobserved and one source-missing case; complete Standard |

No restricted native parity test was rerun for this export. No public engine
card has been newly designated as fully parity-verified. The full-Standard gate
must still exit **2**:

```sh
python scripts/verify_template.py --require-full-standard
```

## Refresh provenance

`scripts/sync_template.py` refreshes only the canonical Python allowlist from
an explicitly accepted immutable commit. Use its dry-run before `--write`.
It does not refresh the separate example export.

For `reuse-manifest.json`, retrieve each `sourcePath` using `git show` at its
recorded commit, preserve bytes and relative layout under `examples/tcg-agent/`,
then verify its SHA-256. Review any proposed new path and its dependency closure
before changing the manifest. A source update is not a coverage admission.
Run both unchanged test suites, the template verifier and package smoke checks
before publication. Never add downloaded source/cache contents to either manifest.
