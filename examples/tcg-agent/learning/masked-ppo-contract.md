# Masked PPO integration contract

`masked_ppo.py` owns the candidate actor, state-only critic, GAE targets, PPO updates, and private checkpoints. The native collector, reward attribution, frozen selection actor, evaluation schedule, and training budget belong to the main trainer.

## Actor and critic

`ModelConfig` defaults to `native-card-action-features/1`, 224 state floats, 191 floats per candidate, and hidden widths 128 and 64. Dimensions and schema are configurable. The actor uses the existing `CandidateScorer.network` structure. `warm_start(saved_bc_payload)` loads the actor strictly after checking the existing feature contract and hash. The critic is independent and receives only actor-visible state.

`MaskedActorCritic.forward(states, candidates, mask)` accepts float tensors `[B,S]` and `[B,K,A]`, plus a boolean legal mask `[B,K]`. It returns a categorical distribution over candidates and a value tensor `[B]`. Padding has zero probability. Empty legal sets fail. Candidate order is preserved, and no candidate list is truncated. `pad_records` derives the mask from each complete legal list.

The frozen BC selection policy is a separate model instance. Reusing the updating PPO actor for selections would make the selection policy change during training.

## Rollout records

Records are flat, grouped by episode, and chronological within each episode. Each record contains these fields:

| Field | Meaning |
| --- | --- |
| `schema` | Feature schema, normally `native-card-action-features/1` |
| `kind` | `ordinary`; selection decisions are excluded |
| `state` | Complete actor-visible state feature vector |
| `candidates` | Complete legal candidate feature rows in offered order |
| `actionIndex` | Sampled index into this exact candidate list |
| `oldLogProb` | Log probability under the frozen collection policy |
| `oldValue` | Critic value at this state under the collection checkpoint |
| `reward` | Reward accrued before the next learned ordinary decision or boundary |
| `terminated` | Natural game termination |
| `truncated` | Environment or episode-budget cutoff with no terminal result |
| `nextValue` | Collection critic value at the next or final pre-reset actor-visible state |
| `episodeId` | Unique game plus controlled-seat identifier |
| `stepIndex` | Contiguous index counting learned ordinary decisions for this seat |
| `segmentEnd` | Stops GAE recursion at termination, truncation, or a collector batch cut |

All flags are booleans. Every final record in the batch and every terminal or truncated record marks `segmentEnd=true`. A segment can start at a nonzero `stepIndex`. The collector must not concatenate different controlled seats into one episode. Missing bootstrap values, unmarked episode changes, step gaps, and selection records fail admission. Within an unbroken segment, `nextValue` must match the following `oldValue`.

The PPO collector samples from `PPOTrainer.act(..., deterministic=False)` and stores its returned `actionIndex`, `oldLogProb`, and `oldValue` verbatim. Evaluation uses `deterministic=True`. Teacher argmax, epsilon mixing, temperature changes, and candidate reordering cannot be used with unchanged PPO log probabilities. `update` checks collection log probabilities and values against the current frozen checkpoint before any optimizer step. Collect a fresh rollout after every update.

## Reward and bootstrap

The baseline reward is +1 for a natural learner win and -1 for a natural learner loss. A natural draw is 0 if the native engine supplies one. Intermediate reward is 0. Timeout, budget exhaustion, and unavailable terminal outcomes are truncations, never implicit losses. The collector is responsible for the actual outcome and for assigning a terminal result that arrives after opponent or frozen-selection activity to the preceding learned ordinary transition.

Gamma is per learned ordinary-action decision. Opponent turns and frozen selections are part of the transition between those decisions. This baseline does not claim fixed-time or per-native-command discounting.

The learner computes:

```text
delta[t] = reward[t] + gamma * (not terminated[t]) * nextValue[t] - oldValue[t]
advantage[t] = delta[t] + gamma * lambda * (not boundary[t]) * advantage[t+1]
boundary[t] = terminated[t] or truncated[t] or segmentEnd[t]
return[t] = advantage[t] + oldValue[t]
```

Natural termination removes bootstrap. Truncation and collector cuts bootstrap from the actual pre-reset state and stop recursion. A final next ordinary decision is unnecessary for value inference, because the critic consumes state alone. If no valid final actor-visible state can be encoded, exclude that segment with an explicit reason. Never substitute a reset state or invent a zero bootstrap for an unknown truncation value.

## Python API

```python
from learning.masked_ppo import ModelConfig, PPOConfig, PPOTrainer

trainer = PPOTrainer(ModelConfig(), PPOConfig(), seed=1729, device="cpu")
trainer.warm_start(saved_bc_payload)
actions = trainer.act(observation_records, deterministic=False)
next_values = trainer.values(actor_visible_state_vectors)
metrics = trainer.update(completed_rollout_records)
trainer.save(absolute_new_path_under_runs, metadata={"sourceCommit": source_commit})
resumed = PPOTrainer.load(absolute_checkpoint_path, device="cpu")
```

`act` accepts feature records and returns one dictionary per record with `actionIndex`, `oldLogProb`, and `oldValue`. `values` returns one float per state. `update` admits at most `max_records` and pads only a minibatch at a time. It splits shuffled minibatches to respect `max_padded_candidates`. A single decision that exceeds that bound fails without dropping candidates. `act` and `pad_records` reject an excessive complete padded batch. Keep collector and learner bounds consistent.

`PPOConfig` fixes gamma, GAE lambda, policy clipping, value clipping, entropy coefficient, value coefficient, learning rate, gradient norm, epochs, minibatch size, target KL, advantage normalization, and memory bounds. Defaults are serialized with every checkpoint. Metrics include policy and value loss, entropy, approximate KL, clip fraction, explained variance, gradient norm, weight change, actual optimizer steps, and stop reason. Explained variance is `None` for constant targets.

The policy objective uses the minimum of the ordinary and clipped importance-weighted advantages. Value loss is half the maximum of squared unclipped and clipped errors. Advantage normalization uses population standard deviation over the frozen rollout and handles singleton minibatches. Approximate KL is `mean(exp(log_ratio) - 1 - log_ratio)`. A full-rollout diagnostic after each epoch controls target-KL stopping. `grad_norm` is the largest norm before clipping in this update.

Admission and stale-policy checks finish before weight changes. A numerical failure during optimization raises immediately and can leave earlier minibatch steps applied. Discard that trainer instance and reload the last accepted checkpoint. An update is accepted only after it returns finite diagnostics. The caller owns its wall-time budget and checkpoints between accepted updates.

Checkpoint writes require a fresh absolute `.pt` path under the repository's ignored `runs` directory. Checkpoints preserve model and optimizer state, model and PPO configs, seed, generator state, update count, feature metadata, and learner source hash. Loading uses the restricted weights-only PyTorch loader. Caller metadata records collection seeds, source commit, BC parent hash, frozen selection hash, opponent, and native source pins. Weights and rollout records stay private.

## Frozen JSONL inference process

`python learning/masked_ppo_service.py --checkpoint <absolute-runs-path.pt> --device cpu` loads a PPO checkpoint for ordinary decisions. `--sampling-seed <integer>`, also accepted as `--seed`, optionally resets its sampling generator to a recorded per-worker seed. Without that option, sampling resumes from the saved generator state. The service performs inference only and does not train or invoke a teacher. The Node trainer supplies its own adapter instead of launching the BC-only `card_action_torch.py` entry point.

Every request and response has an integer `id`. Responses contain `ok`. Failed requests contain `error` and do not kill the service. Requests are newline-delimited JSON and responses flush one JSON line. The operations are:

```json
{"id":1,"op":"ping"}
{"id":2,"op":"infer","deterministic":false,"schema":"native-card-action-features/1","kind":"ordinary","state":["224 floats"],"candidates":[["191 floats per legal candidate"]]}
{"id":3,"op":"infer_batch","deterministic":true,"items":["ordinary feature records"]}
{"id":4,"op":"value","state":["224 floats"]}
{"id":5,"op":"values","states":[["224 floats per state"]]}
{"id":6,"op":"close"}
```

Quoted feature rows illustrate shapes. Actual requests use finite numbers. Both inference operations require an explicit boolean `deterministic`. Training collection uses `false`; frozen argmax evaluation uses `true`. `infer` returns `index`, `actionIndex`, `oldLogProb`, `oldValue`, and `selectionMode`. `index` equals `actionIndex` and maps directly into the still-current offered legal list. The log probability always corresponds to that returned action under the masked stochastic policy, including when argmax selects it.

`infer_batch` returns parallel `indices`, `oldLogProbs`, and `oldValues` arrays, plus `decisions` containing the same per-item dictionaries as the Python API. `selectionMode` is `sampled` or `deterministic`. `value` and `values` return `value` and `values` without requiring a next legal action set. `ping` reports feature dimensions, contract hash, printing aliases, checkpoint path and SHA256, checkpoint schema `native-masked-ppo/1`, update count, sampling seed, device, and objective. Requests with `kind=selection` fail. The separate frozen BC selection process handles those requests, with its checkpoint attributed by the collector.

## Reference and validation

The reference is [Spider policy_training.py at b75c0c7f](https://github.com/haidmoham/spider/blob/b75c0c7f4d198be794d340948244a7c884e91e17/spider/policy_training.py), especially `gae_targets`, `clipped_policy_loss`, and `approximate_kl`. The adaptation uses a masked discrete categorical policy. It bootstraps genuine truncations instead of copying Spider's experiment-specific horizon-as-terminal choice. No Spider continuous-action distribution or robot reward terms are used.

`python -m unittest learning.test_masked_ppo -v` checks hand-calculated GAE and clipping, masks, candidate order, state-only values, exact warm-start scores, admission failures, stale-policy rejection, finite updates, checkpoint continuation, and terminal-reward learning on a synthetic two-action task. These checks establish the mathematical and software behavior of the component. They do not establish TCG strength. Frozen teacher, Model2, and random-opponent native evaluation remains required for the main experiment.
