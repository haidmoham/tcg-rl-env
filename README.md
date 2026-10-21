# TCG RL environment template

Reuse actor-visible observations, exact legal-action selections, rollout
accounting, policy code, and training and evaluation recipes for Pokemon TCG
research. The target is the full dated Standard card pool.

**Full Standard training is not ready.** This release contains reusable code
and an aggregate coverage snapshot. Native games require a separately obtained,
authorized engine. The repository does not supply card programs, model weights,
raw game data, or a complete public simulator.

## Inspect the template

Use Python 3.10 or later. These commands use the standard library and start no
games, models, training, or network requests.

```sh
git clone https://github.com/haidmoham/tcg-rl-env.git
cd tcg-rl-env
python scripts/verify_template.py
python scripts/plan_campaign.py
python scripts/check_readiness.py
```

The verifier must pass. The historical reference readiness command must exit
with status 2 while its campaign remains unqualified. A passing template check
establishes source integrity and the exercised contracts. It establishes no
native gameplay parity or model improvement.

Inspect the fixed historical evaluation without executing it.

```sh
python scripts/evaluate_native_checkpoint_288.py \
  --protocol experiments/mew-sustained-v1.json \
  --amendment experiments/mew-evaluation-288-v1.json --plan-only
```

The plan contains 288 games and 144 paired clusters. Its historical six-deck
scope is separate from current Standard qualification.

## Reuse legal decisions

The `tcg` package preserves option IDs, cardinality, selection order, repeated
allocations, and explicit submission. A reader must supply only information
visible to the acting player. Unknown card identities or unsupported prompt
semantics must reject instead of silently truncating the observation or options.

```python
from tcg.decision import ActorDecision, LegalPrompt, Option, Token

decision = ActorDecision(
    tokens=(),
    prompt=LegalPrompt(
        options=(Option("first", Token()), Option("second", Token())),
        minimum=2,
        maximum=2,
        ordered=True,
    ),
)
decision = decision.pick("second").pick("first")
assert decision.response() == ("second", "first")
```

`tcg.rollout` distinguishes natural terminal outcomes, administrative
truncation, and infrastructure failure. Truncations remain censored. An
infrastructure failure invalidates the rollout. Preserve these distinctions
when connecting another trainer.

`tcg.cabt.CabtBattle` accepts an explicit engine path and SHA-256 identity.
Importing it loads no native library. Its optional seed API declaration does
not itself prove deterministic replay. `tcg.cabt_observation` projects the
supported actor-visible wire observations and rejects unsupported prompts.

## Supply a training runtime

The generic package has no required third-party dependency. Install the
historical policy prototype separately when you need its PyTorch components.

```sh
python -m pip install .
python -m pip install -r requirements-policy.txt
```

Use Python 3.10 through 3.12 for that historical policy prototype. Its
requirements do not recreate the separately qualified native
Mew runtime. The optional native recipes require a compatible user-supplied Mew
source tree, an authorized CABT engine, exact deck manifests, and, for teacher
transfer, an authorized checkpoint. Obtain redistribution and usage rights for
those inputs separately. This repository's MIT license covers its project code.

The exported recipes preserve their original input and protocol checks.
They are starting points for a declared experiment. They do not grant execution
authorization or qualify new artifacts by themselves.

| Path | Reusable work |
| --- | --- |
| `tcg/decision.py`, `tcg/rollout.py` | Legal selections and complete-game learning targets. |
| `tcg/policy.py`, `tcg/ppo_update.py` | Policy and PPO components for the historical prototype. |
| `tcg/collection.py`, `tcg/population.py` | Assignment identities and opponent sampling. |
| `tcg/native_policy.py`, `tcg/native_distillation.py` | Explicit boundary to a separately supplied Mew implementation. |
| `tcg/native_actorcritic.py`, `tcg/native_sustained.py` | Critic calibration and teacher-free improvement stages. |
| `scripts/prepare_native_pilot_bindings.py` | Hash and bind authorized external inputs. |
| `scripts/train_native_sustained.py` | Bounded historical training recipe. |
| `scripts/evaluate_native_checkpoint_288.py` | Frozen, paired historical evaluation. |
| `scripts/audit_native_campaign.py` | Audit saved campaign receipts. |

Keep engine libraries, copied vendor code, checkpoints, local bindings, and raw
traces outside the repository. Match a recipe's exact source and artifact pins
before execution. Keep a frozen external evaluation reference and retain faults
and censoring. Teacher loss and self-play win rate do not establish improvement.

The default `check_readiness.py` command needs no deck data. Its optional
`--evidence` path also expects the omitted historical deck manifest. Restore
that exact artifact only after establishing permission to use it. The public
template does not include third-party deck lists.

## Check Standard coverage

[`coverage/current.json`](coverage/current.json) records a dated aggregate from
the retained qualification work. Its requirement counts describe declared
scenarios against a frozen pool. Complete declared scenarios do not prove every
possible behavior of a card. The aggregate includes source receipt hashes.
Readers without the restricted inputs cannot independently replay that evidence.

```sh
python scripts/verify_template.py --require-full-standard
```

This release exits with status 2 for full Standard readiness. It cannot enable
training by editing a summary boolean. Full readiness requires a separately
reviewed gate over exact engine, pool, registry, reader, vocabulary, adapter,
deck, and evidence identities. Missing printings, unresolved identities,
unsupported prompts, privacy gaps, deferred required rulings, and uncertain
current-server legality remain blockers. Effects, legal prompts, information
boundaries, interactions, and mechanics need qualified evidence on the same
build. A future gate must validate those inputs before enabling full-pool runs.

## Refresh accepted work

One maintainer owns the public export. Engine, training, and evaluation owners
retain their separate source and admission responsibilities. Export only an
explicitly accepted immutable source commit. Review the diff for restricted
content before publishing it.

[`export-manifest.json`](export-manifest.json) lists every approved source file
and its exact hash. `scripts/sync_template.py --help` describes the dry-run
check and explicit write operation. The exporter reads committed Git blobs.
It does not copy dirty files, untracked candidates, source history, or newly
added files outside the allowlist. Source refresh and coverage admission are
separate decisions. A newer source commit does not advance coverage by itself.

Preserve exported source bytes. Add a source file only after reviewing its
content, dependency closure, attribution, and public redistribution scope.
Publish sanitized aggregate receipts separately. Never import raw oracles,
card text dumps, or private paths as coverage evidence.

The maintainer runs the verifier, confirms the expected full-Standard rejection,
and checks a clean clone before pushing a refresh. CI repeats the public checks
on Linux and Windows. Restricted native qualification remains external.
