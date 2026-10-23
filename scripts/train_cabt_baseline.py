"""Run only the explicitly authorized, bounded historical CABT baseline."""

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENGINE_SHA256 = "a9792857abafb44709bb9062a6d7cb5aa9cd1148c5168ea3214b20a91b62c0f7"
WATCHDOG_SECONDS = 7200 + 900 + 120
_COLLECTION = {
    "learner_seat": "uniform_independent_per_game", "training_deck_indices": [0, 1, 2, 3],
    "deck_sampling": "independent_uniform_with_replacement", "minimum_learner_micro_decisions": 8192,
    "include_forced_learner_micro_decisions": True, "max_engine_submissions_per_game": 4096,
    "max_engine_submissions_per_turn": 512,
    "native_response_semantics": "ordered_unique_index_vector",
    "administrative_truncation": "censor_whole_game_no_synthetic_outcome",
    "truncation_stop_fraction": 0.01, "truncation_stop_minimum_started_games": 100,
}
_LIMITS = {
    "accepted_ppo_updates": 12, "learner_micro_decisions_soft_cap": 100000,
    "training_wall_seconds": 7200, "finish_only_current_complete_update_at_cap": True,
    "qualification_updates_before_continuing": 2, "cpu_collection_workers": 1,
    "cuda_fit_memory_fraction": 0.5, "microbatch_size": 8, "precision": "float32_no_amp",
}
_CHECKPOINTING = {
    "complete_update_boundary_every_updates": 1, "population_snapshot_interval_updates": 10,
    "initial_checkpoint_retained": True, "checkpoint_selection": "last_complete_verified_update_at_fixed_cap",
}
_EVALUATION = {
    "games": 32, "opponent": "retained_initial_random_checkpoint", "seat_assignment": "balanced",
    "deck_strata": ["training/training", "training/heldout", "heldout/training", "heldout/heldout"],
    "games_per_stratum": 8, "native_pairing": False,
    "interpretation": "descriptive_small_sample_no_competitive_or_formal_reproduction_claim",
    "checkpoint_selection": False,
}
_SELFPLAY = {
    "sampling_mode": "current_pfsp", "current_policy_probability": 0.5,
    "history_pool": "initial_plus_latest_31_archived_snapshots_excluding_current_weights",
    "history_weight": "(1-mean(last_256_completed_game_scores))**2", "unseen_score_prior": 0.5,
    "all_zero_weights": "uniform_historical", "snapshot_interval_accepted_updates": 10,
    "pfsp_exponent_and_window": "local_reconstruction_not_author_disclosed",
}


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    paths = [*sorted((ROOT / "tcg").glob("*.py")), Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in paths}


def checkout_identity():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT,
                            check=True, capture_output=True, text=True)
    return {"head": head.stdout.strip(), "dirty": bool(status.stdout), "status": status.stdout.splitlines()}


def exclusive_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encode(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_progress(directory, value):
    with (Path(directory) / "progress.jsonl").open("ab") as stream:
        stream.write(encode(value) + b"\n")
        stream.flush()


def write_failure(directory, error, **context):
    try:
        exclusive_json(Path(directory) / "failure.json", {
            "status": "failed", "error_type": type(error).__name__, "reason": str(error), **context,
        })
    except FileExistsError:
        pass


def _hash(value):
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("required lowercase SHA-256 pin is missing or invalid")


@dataclass(frozen=True)
class Prepared:
    protocol: dict
    protocol_path: Path
    protocol_sha256: str
    binary: Path
    initial_checkpoint: Path
    reference_sha256: str
    decks_sha256: str
    decks: tuple[tuple[int, ...], ...]
    vocabulary: object
    policy: object


def validate_inputs(protocol_path, binary, initial_checkpoint, *, device="cuda"):
    """Read and validate every launch pin before creating a directory or using CUDA."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be explicitly cpu or cuda")
    protocol_path = Path(protocol_path).resolve(strict=True)
    protocol = json.loads(protocol_path.read_bytes())
    required = {"schema", "status", "baseline_training_authorized", "ranked_lineage_training_authorized",
                "scope", "reference_proposal", "reference_proposal_sha256", "model", "engine",
                "deck_manifest", "deck_manifest_sha256", "projection_sha256", "algorithm", "collection",
                "limits", "checkpointing", "evaluation", "holds", "known_limitations", "selfplay"}
    if type(protocol) is not dict or set(protocol) != required:
        raise ValueError("unsupported baseline protocol fields")
    if (protocol["schema"] != "86-tcg-agent/cabt-small-baseline/1"
            or protocol["status"] != "authorized_pending_execution_checks"
            or protocol["baseline_training_authorized"] is not True
            or protocol["ranked_lineage_training_authorized"] is not False
            or protocol["scope"] != "small_reconstructed_cabt_kaggle_baseline_not_formal_confirmation_or_ranked_player"
            or protocol["algorithm"] != "reference_ppo_hyperparameters_with_source_informed_current_pfsp_league"):
        raise ValueError("small baseline authorization or algorithm scope mismatch")
    for name, expected in (("collection", _COLLECTION), ("limits", _LIMITS),
                           ("checkpointing", _CHECKPOINTING), ("evaluation", _EVALUATION), ("selfplay", _SELFPLAY)):
        if encode(protocol[name]) != encode(expected):
            raise ValueError(f"baseline {name} differs from the authorized recipe")
    for name in ("reference_proposal_sha256", "deck_manifest_sha256", "projection_sha256"):
        _hash(protocol[name])
    model = protocol["model"]
    if (type(model) is not dict or set(model) != {"initialization", "seed", "parameter_count", "initial_weights_sha256"}
            or model["initialization"] != "random" or type(model["seed"]) is not int or model["seed"] != 860001
            or type(model["parameter_count"]) is not int or model["parameter_count"] != 972802):
        raise ValueError("baseline random model specification mismatch")
    _hash(model["initial_weights_sha256"])
    engine = protocol["engine"]
    engine_fields = {"version", "match_format", "sha256", "native_seed_control", "chance_replay_qualified",
                     "rules_target", "official_release_sha256", "seed_api", "native_seed_bits", "replay_scope",
                     "seed_semantics", "card_pool_count", "card_ports"}
    if (type(engine) is not dict or set(engine) != engine_fields or engine.get("sha256") != ENGINE_SHA256
            or encode({key: engine.get(key) for key in ("version", "match_format", "native_seed_control",
                       "chance_replay_qualified", "seed_api", "native_seed_bits", "card_pool_count", "card_ports")})
            != encode({"version": 1, "match_format": "best_of_one", "native_seed_control": True,
                       "chance_replay_qualified": True, "seed_api": "BattleStartSeeded64", "native_seed_bits": 64,
                       "card_pool_count": 1267, "card_ports": False})
            or engine.get("rules_target") != "historical_cabt_pool_not_full_current_standard"
            or engine.get("official_release_sha256") != "eae88634e26dc31d94150a4d8202fc9d32596b8c688ef67e14cb4088cd4d5771"
            or engine.get("seed_semantics") != "per_battle_mt19937_seed_seq_low32_then_high32_before_setup"
            or engine.get("replay_scope") != "current_windows_build_actor_json_and_responses_only"):
        raise ValueError("qualified historical seeded engine contract mismatch")
    required_holds = {"ranked_lineage_training_until_full_target_coverage", "live_client_and_ranked_play",
                      "cloud_spending", "old_experiment_modification", "private_engine_source_or_checkpoint_publication"}
    if (type(protocol["holds"]) is not list or set(protocol["holds"]) != required_holds
            or type(protocol["known_limitations"]) is not list or not protocol["known_limitations"]
            or any(type(value) is not str or not value for value in protocol["known_limitations"])):
        raise ValueError("baseline holds and limitations must remain explicit")
    if protocol["reference_proposal"] != "experiments/r0-reference.json" or protocol["deck_manifest"] != "experiments/cabt-v1-candidate-decks.json":
        raise ValueError("reference and deck manifest must use the canonical project files")
    reference_path, decks_path = ROOT / protocol["reference_proposal"], ROOT / protocol["deck_manifest"]
    binary = Path(binary).resolve(strict=True)
    initial_checkpoint = Path(initial_checkpoint).resolve(strict=True)
    if (sha256(binary) != engine["sha256"] or sha256(reference_path) != protocol["reference_proposal_sha256"]
            or sha256(decks_path) != protocol["deck_manifest_sha256"]
            or sha256(ROOT / "tcg/cabt_observation.py") != protocol["projection_sha256"]
            or sha256(initial_checkpoint / "weights.pt") != model["initial_weights_sha256"]):
        raise ValueError("binary, reference, decks, projection or initial weight hash mismatch")
    reference = json.loads(reference_path.read_bytes())
    learning = reference["learning"]
    expected_learning = {"algorithm": "ppo", "behavior_cloning": False, "reward_shaping": False,
                         "teacher_kl_coefficient": 0.0, "teacher_value_coefficient": 0.0,
                         "auxiliary_losses": False, "search_annotations": False, "train_search": False,
                         "evaluation_search": False, "win_reward": 1.0, "loss_reward": -1.0,
                         "draw_reward": 0.0, "gamma": 1.0, "gae_lambda": 0.95,
                         "trajectory": "learner_seat_only_complete_games", "rollout_minimum_learner_decisions": 8192,
                         "update_epochs": 4, "minibatch_size": 256, "optimizer": "adamw", "learning_rate": 1e-4,
                         "adam_betas": [0.9, 0.999], "adam_epsilon": 1e-5, "weight_decay": 0.0,
                         "learning_rate_schedule": "constant", "clip_ratio": 0.2, "value_coefficient": 0.5,
                         "value_loss": "mean_squared_error", "value_clipping": False, "entropy_coefficient": 0.01,
                         "advantage_normalization": "once_per_complete_rollout", "gradient_norm_cap": 0.5,
                         "target_kl": 0.02, "kl_stop": "stop_remaining_epochs_after_full_rollout_KL_check"}
    if encode(learning) != encode(expected_learning):
        raise ValueError("reference PPO loss or optimizer differs from the declared baseline")
    deck_manifest = json.loads(decks_path.read_bytes())
    if deck_manifest.get("engine_version") != 1 or len(deck_manifest.get("decks", [])) != 6:
        raise ValueError("baseline requires six pinned historical v1 deck lists")
    decks = tuple(tuple(row["card_ids"]) for row in deck_manifest["decks"])
    if any(len(deck) != 60 or any(type(card) is not int or card <= 0 for card in deck) for deck in decks):
        raise ValueError("each deck must contain exactly 60 positive integer engine IDs")
    from tcg.cabt_observation import CabtObservationAdapter
    from tcg.checkpoint import load_checkpoint
    from tcg.policy import PolicyConfig
    import torch
    torch.set_num_threads(1)
    manifest = json.loads((initial_checkpoint / "manifest.json").read_bytes())
    card_ids = manifest["vocabulary"]["card_ids"]
    if (type(card_ids) is not list or len(card_ids) != 1267
            or any(type(card) is not str or not card.isdecimal() or int(card) <= 0 for card in card_ids)
            or len(set(card_ids)) != len(card_ids)):
        raise ValueError("initial checkpoint must pin the complete v1 card vocabulary")
    if not set(str(card) for deck in decks for card in deck) <= set(card_ids):
        raise ValueError("deck IDs are absent from the initial vocabulary")
    vocabulary = CabtObservationAdapter(tuple(card_ids)).vocabulary()
    policy = load_checkpoint(initial_checkpoint, vocabulary, PolicyConfig())
    if sum(parameter.numel() for parameter in policy.parameters()) != model["parameter_count"]:
        raise ValueError("initial model parameter count mismatch")
    if sha256(initial_checkpoint / "weights.pt") != model["initial_weights_sha256"]:
        raise ValueError("initial checkpoint changed during validation")
    return Prepared(protocol, protocol_path, sha256(protocol_path), binary, initial_checkpoint,
                    protocol["reference_proposal_sha256"], protocol["deck_manifest_sha256"], decks, vocabulary, policy)


def domain_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")


@dataclass(frozen=True)
class DeckDraw:
    learner_seat: int
    learner_deck_index: int
    opponent_deck_index: int
    engine_seed: int


def training_draw(generator):
    return DeckDraw(generator.randrange(2), generator.randrange(4), generator.randrange(4), generator.getrandbits(64))


def evaluation_plan(seed):
    generator = random.Random(domain_seed(seed, "evaluation-assignments"))
    pools = ((range(4), range(4)), (range(4), range(4, 6)),
             (range(4, 6), range(4)), (range(4, 6), range(4, 6)))
    result = []
    for stratum, (learner_pool, opponent_pool) in enumerate(pools):
        for index in range(8):
            result.append((stratum, DeckDraw(index % 2, generator.choice(learner_pool),
                                            generator.choice(opponent_pool), generator.getrandbits(64))))
    return tuple(result)


@dataclass
class CampaignCounts:
    started_at: float
    accepted_updates: int = 0
    optimizer_steps: int = 0
    games_started: int = 0
    games_finished: int = 0
    admitted_decisions: int = 0
    trained_decisions: int = 0
    raw_decisions: int = 0
    truncated_games: int = 0

    def cap_reason(self, now):
        if self.accepted_updates >= 12:
            return "accepted_update_cap"
        if self.admitted_decisions >= 100000:
            return "admitted_decision_soft_cap"
        if now - self.started_at >= 7200:
            return "training_wall_cap"
        return None


def configure_fit(policy, device):
    import torch
    from tcg.ppo_update import create_optimizer
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise ValueError("unsupported deterministic CUBLAS workspace configuration")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("explicit CUDA fit requested but CUDA is unavailable")
        torch.cuda.set_per_process_memory_fraction(0.5)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    policy.to(device=device, dtype=torch.float32).requires_grad_(True)
    policy.train()
    optimizer = create_optimizer(policy, baseline_authorized=True)
    return optimizer


def write_rollout(directory, index, batch):
    path = Path(directory) / "rollouts" / f"update-{index:04d}.jsonl.gz"
    digest = hashlib.sha256()
    with gzip.open(path, "xt", encoding="utf-8") as stream:
        def emit(value):
            line = encode(value) + b"\n"
            digest.update(line)
            stream.write(line.decode())
        emit({"schema": "86-tcg-agent/private-loss-inputs/1", "identity": asdict(batch.identity),
              "assignments": [asdict(value) for value in batch.assignments],
              "counts": asdict(batch.cumulative_counts), "censored_episode_ids": batch.targets.censored_episode_ids})
        for index, step in enumerate(batch.targets.steps):
            emit({"step": asdict(step), "return": batch.targets.returns[index],
                  "advantage": batch.targets.advantages[index],
                  "normalized_advantage": batch.targets.normalized_advantages[index]})
        stream.write((encode({"footer": True, "rows": len(batch.targets.steps), "sha256": digest.hexdigest()}) + b"\n").decode())
    return {"path": str(path.relative_to(directory)), "file_sha256": sha256(path), "rows": len(batch.targets.steps),
            "content_sha256": digest.hexdigest()}


def record_episode(directory, phase, assignment, draw, result):
    from tcg.rollout import EpisodeEnd
    path = Path(directory) / "episodes.jsonl.gz"
    with gzip.open(path, "at", encoding="utf-8") as stream:
        stream.write((encode({"schema": "86-tcg-agent/private-episode/1", "phase": phase,
                              "assignment": asdict(assignment), "draw": asdict(draw)}) + b"\n").decode())
        for step in result.episode.steps:
            stream.write((encode({"step": asdict(step)}) + b"\n").decode())
        stream.write((encode({"end": result.episode.end.value, "reason": result.reason,
                              "terminal_result": result.terminal_result}) + b"\n").decode())
    progress = {"event": "game", "phase": phase, "episode_id": assignment.episode_id,
                "identity": asdict(assignment.identity), "opponent": asdict(assignment.opponent),
                "learner_deck_sha256": assignment.learner_deck_sha256,
                "opponent_deck_sha256": assignment.opponent_deck_sha256,
                "draw": asdict(draw), "end": result.episode.end.value, "reason": result.reason,
                "wall_seconds": result.wall_seconds, "learner_micro_decisions": len(result.episode.steps),
                "submissions_attempted": result.submissions_attempted, "submissions_accepted": result.submissions_accepted}
    progress.update(ordered_response_checks=result.ordered_response_checks,
                    multi_selection_submissions=result.multi_selection_submissions)
    if result.episode.end is EpisodeEnd.TERMINAL:
        progress["terminal_reward"] = result.episode.steps[-1].reward
    append_progress(directory, progress)


def _absolute_decks(prepared, draw):
    pair = [None, None]
    pair[draw.learner_seat] = prepared.decks[draw.learner_deck_index]
    pair[1 - draw.learner_seat] = prepared.decks[draw.opponent_deck_index]
    return tuple(pair)


def run_campaign(prepared, directory, device):
    import torch
    from tcg.cabt import CabtBattle
    from tcg.cabt_observation import CabtObservationAdapter
    from tcg.collection import BatchCollector, CollectionCounts, CollectionIdentity, GameAssignment
    from tcg.episode_runner import EpisodeLimits, FrozenPolicy, deck_sha256, run_episode
    from tcg.population import Population, Snapshot
    from tcg.ppo_update import update_policy
    from tcg.rollout import EpisodeEnd
    from tcg.training_checkpoint import TrainingCounters, TrainingIdentity, policy_weights_sha256, save_training_checkpoint

    if "engine_seed" not in inspect.signature(run_episode).parameters:
        raise ValueError("seeded episode runner support is required; unseeded fallback is forbidden")
    directory = Path(directory)
    for child in ("rollouts", "checkpoints"):
        (directory / child).mkdir(exist_ok=False)
    seed = prepared.protocol["model"]["seed"]
    initial_hash = prepared.protocol["model"]["initial_weights_sha256"]
    identity = TrainingIdentity(prepared.protocol_sha256, prepared.reference_sha256,
                                prepared.protocol["engine"]["sha256"], prepared.decks_sha256,
                                prepared.protocol["projection_sha256"], initial_hash)
    population = Population(Snapshot(0, initial_hash), seed=domain_seed(seed, "population"), sampling_mode="current_pfsp")
    deck_rng = random.Random(domain_seed(seed, "training-assignments"))
    rngs = {name: torch.Generator(device="cpu").manual_seed(domain_seed(seed, name))
            for name in ("learner", "opponent", "shuffle")}
    paths = {initial_hash: prepared.initial_checkpoint}
    frozen = {initial_hash: FrozenPolicy.from_initial_checkpoint(prepared.initial_checkpoint, initial_hash, prepared.vocabulary)}
    current_hash, current_path = initial_hash, prepared.initial_checkpoint
    counts = CampaignCounts(time.monotonic())
    stop_reason = None

    def resolve(snapshot):
        if snapshot.weights_sha256 not in paths:
            raise ValueError("sampled opponent has no retained immutable artifact")
        if snapshot.weights_sha256 not in frozen:
            frozen[snapshot.weights_sha256] = FrozenPolicy.from_trained_checkpoint(
                paths[snapshot.weights_sha256], snapshot.weights_sha256, prepared.vocabulary, expected_identity=identity)
        return frozen[snapshot.weights_sha256]

    def play(assignment, draw, seconds):
        with CabtBattle(prepared.binary, identity.engine_sha256, seed_api="BattleStartSeeded64") as battle:
            return run_episode(battle, assignment, resolve(Snapshot(counts.accepted_updates, current_hash)),
                               resolve(assignment.opponent), CabtObservationAdapter(prepared.vocabulary.card_ids),
                               _absolute_decks(prepared, draw), rngs["learner"], rngs["opponent"],
                               protocol_path=prepared.protocol_path, engine_seed=draw.engine_seed,
                               limits=EpisodeLimits(4096, 512, min(120.0, seconds)))

    try:
        optimizer = configure_fit(prepared.policy, device)
        while counts.cap_reason(time.monotonic()) is None:
            collection_identity = CollectionIdentity(current_hash, counts.accepted_updates, identity.engine_sha256,
                                                     identity.projection_sha256, identity.protocol_sha256)
            collector = BatchCollector(collection_identity, population, CollectionCounts(counts.games_started, counts.truncated_games))
            while not collector.ready and counts.cap_reason(time.monotonic()) is None:
                draw = training_draw(deck_rng)
                assignment = collector.start_game(f"train-{counts.games_started:08d}", draw.learner_seat,
                    deck_sha256(prepared.decks[draw.learner_deck_index]), deck_sha256(prepared.decks[draw.opponent_deck_index]))
                counts.games_started += 1
                result = play(assignment, draw, 7200 - (time.monotonic() - counts.started_at))
                counts.games_finished += 1
                counts.raw_decisions += len(result.episode.steps)
                if result.episode.end is EpisodeEnd.TERMINAL:
                    counts.admitted_decisions += len(result.episode.steps)
                elif result.episode.end is EpisodeEnd.TRUNCATION:
                    counts.truncated_games += 1
                record_episode(directory, "training", assignment, draw, result)
                collector.complete_game(assignment, result.episode.steps, result.episode.end)
                append_progress(directory, {"event": "collection_counts", **asdict(counts), "elapsed_seconds": time.monotonic() - counts.started_at})
            stop_reason = counts.cap_reason(time.monotonic())
            if stop_reason is not None:
                break
            batch = collector.freeze()
            rollout_receipt = write_rollout(directory, counts.accepted_updates + 1, batch)
            before = {name: tensor.detach().cpu().clone() for name, tensor in prepared.policy.state_dict().items()}
            prepared.policy.train()
            fit_started = time.monotonic()
            append_progress(directory, {"event": "fit_started", "policy_version": counts.accepted_updates,
                                        "learner_decisions": len(batch.targets.steps), "device": device,
                                        "rollout": rollout_receipt})
            report = update_policy(prepared.policy, optimizer, batch, expected_identity=collection_identity,
                                   shuffle_generator=rngs["shuffle"], weights_path=current_path / "weights.pt",
                                   baseline_authorized=True, microbatch_size=8,
                                   deadline_monotonic=counts.started_at + 7200,
                                   progress_callback=lambda measured: append_progress(
                                       directory, {"event": "fit_minibatch", **measured}))
            movement = math.sqrt(sum(float((tensor.detach().cpu() - before[name]).double().square().sum())
                                     for name, tensor in prepared.policy.state_dict().items()))
            if not math.isfinite(movement) or movement <= 0:
                raise ValueError("accepted update requires finite nonzero parameter movement")
            version = counts.accepted_updates + 1
            prospective = policy_weights_sha256(prepared.policy)
            population.accept_update(Snapshot(version, prospective) if version % 10 == 0 else None)
            saved_counters = TrainingCounters(version, counts.optimizer_steps + report.optimizer_steps,
                                              counts.games_started, counts.games_finished,
                                              counts.trained_decisions + report.learner_decisions)
            checkpoint = directory / "checkpoints" / f"update-{version:04d}"
            saved_hash = save_training_checkpoint(checkpoint, policy=prepared.policy, optimizer=optimizer,
                identity=identity, counters=saved_counters, population=population, deck_rng=deck_rng,
                policy_rngs=rngs, in_flight_games=0, partial_minibatch=False)
            if saved_hash != prospective:
                raise ValueError("saved checkpoint differs from the prospective learned artifact")
            accepted = FrozenPolicy.from_trained_checkpoint(checkpoint, saved_hash, prepared.vocabulary, expected_identity=identity)
            counts.accepted_updates = version
            counts.optimizer_steps += report.optimizer_steps
            counts.trained_decisions += report.learner_decisions
            current_hash, current_path = saved_hash, checkpoint
            paths[saved_hash], frozen[saved_hash] = checkpoint, accepted
            receipt = {"event": "accepted_update", "counts": asdict(counts), "checkpoint": str(checkpoint.relative_to(directory)),
                       "weights_sha256": saved_hash, "parameter_l2_movement": movement, "update": asdict(report),
                       "fit_and_checkpoint_wall_seconds": time.monotonic() - fit_started,
                       "rollout": rollout_receipt, "health_gate": "passed" if version == 2 else "finite_update"}
            exclusive_json(directory / f"accepted-update-{version:04d}.json", receipt)
            append_progress(directory, receipt)
            print(json.dumps({"accepted_update": version, "trained_decisions": counts.trained_decisions,
                              "checkpoint": str(checkpoint)}, separators=(",", ":")), flush=True)
        final = {"status": "training_complete", "stop_reason": stop_reason or counts.cap_reason(time.monotonic()),
                 "counts": asdict(counts), "last_good_weights_sha256": current_hash,
                 "last_good_checkpoint": str(current_path), "unused_admitted_decisions": counts.admitted_decisions - counts.trained_decisions,
                 "native_active_game_resume_verified": False, "cuda_optimizer_resume_verified": False}
        exclusive_json(directory / "training-result.json", final)
        if counts.accepted_updates == 0:
            raise ValueError("training budget ended before a complete verified update")
        evaluation = []
        evaluation_deadline = time.monotonic() + 900
        evaluation_rngs = {name: torch.Generator(device="cpu").manual_seed(domain_seed(seed, "evaluation-" + name))
                           for name in ("learner", "opponent")}
        for index, (stratum, draw) in enumerate(evaluation_plan(seed)):
            remaining = evaluation_deadline - time.monotonic()
            if remaining <= 0:
                break
            assignment = GameAssignment(f"evaluation-{index:04d}", draw.learner_seat,
                deck_sha256(prepared.decks[draw.learner_deck_index]), deck_sha256(prepared.decks[draw.opponent_deck_index]),
                Snapshot(0, initial_hash), CollectionIdentity(current_hash, counts.accepted_updates,
                    identity.engine_sha256, identity.projection_sha256, identity.protocol_sha256))
            with CabtBattle(prepared.binary, identity.engine_sha256, seed_api="BattleStartSeeded64") as battle:
                result = run_episode(battle, assignment, frozen[current_hash], frozen[initial_hash],
                    CabtObservationAdapter(prepared.vocabulary.card_ids), _absolute_decks(prepared, draw),
                    evaluation_rngs["learner"], evaluation_rngs["opponent"], protocol_path=prepared.protocol_path,
                    engine_seed=draw.engine_seed, limits=EpisodeLimits(4096, 512, min(120.0, remaining)))
            record_episode(directory, "evaluation", assignment, draw, result)
            row = {"stratum": stratum, "end": result.episode.end.value, "reason": result.reason,
                   "score": ((result.episode.steps[-1].reward + 1) / 2 if result.episode.end is EpisodeEnd.TERMINAL else None)}
            evaluation.append(row)
            if result.episode.end is EpisodeEnd.INFRASTRUCTURE:
                raise RuntimeError("evaluation infrastructure failure; fixed sample is incomplete")
        scores = [row["score"] for row in evaluation if row["score"] is not None]
        final.update(status="complete", evaluation={"planned_games": 32, "started_games": len(evaluation),
                     "natural_terminal_games": len(scores), "complete_fixed_sample": len(scores) == 32,
                     "observed_terminal_mean_score": sum(scores) / len(scores) if scores else None,
                     "missing_score_bounds": [sum(scores) / 32, (sum(scores) + 32 - len(scores)) / 32],
                     "games": evaluation, "interpretation": _EVALUATION["interpretation"], "native_pairing": False})
        exclusive_json(directory / "result.json", final)
        return final
    except BaseException as error:
        write_failure(directory, error, counts=asdict(counts), last_good_checkpoint=str(current_path),
                      last_good_weights_sha256=current_hash, phase=getattr(error, "phase", "campaign"),
                      affected_update_checkpoint_allowed=False)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiments/cabt-small-baseline.json")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--_child-token", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    prepared = validate_inputs(args.protocol, args.binary, args.initial_checkpoint, device=args.device)
    directory = args.run_directory.resolve()
    if args._child_token is not None:
        launch = json.loads((directory / "launch.json").read_bytes())
        if (launch["token"] != args._child_token or launch["protocol_sha256"] != prepared.protocol_sha256
                or launch["binary"] != str(prepared.binary) or launch["initial_checkpoint"] != str(prepared.initial_checkpoint)
                or launch["device"] != args.device or launch["source_sha256"] != source_manifest()):
            raise ValueError("child launch differs from the validated exclusive parent request")
        exclusive_json(directory / "child-started.json", {"pid": os.getpid(), "protocol_sha256": prepared.protocol_sha256})
        run_campaign(prepared, directory, args.device)
        return 0
    token = uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    exclusive_json(directory / "launch.json", {"token": token, "protocol_sha256": prepared.protocol_sha256,
                   "binary": str(prepared.binary), "initial_checkpoint": str(prepared.initial_checkpoint),
                   "device": args.device, "watchdog_seconds": WATCHDOG_SECONDS,
                   "source_sha256": source_manifest(), "checkout": checkout_identity()})
    with (directory / "protocol.json").open("xb") as stream:
        stream.write(prepared.protocol_path.read_bytes())
    command = [sys.executable, str(Path(__file__).resolve()), "--protocol", str(prepared.protocol_path),
               "--binary", str(prepared.binary), "--initial-checkpoint", str(prepared.initial_checkpoint),
               "--run-directory", str(directory), "--device", args.device, "--execute", "--_child-token", token]
    with (directory / "child.log").open("xb") as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except BaseException as error:
            write_failure(directory, error, phase="child_spawn")
            raise
        try:
            code = process.wait(timeout=WATCHDOG_SECONDS)
        except BaseException as error:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=30)
            finally:
                write_failure(directory, error, phase="parent_watchdog", accepted_update_receipts=sorted(
                    path.name for path in directory.glob("accepted-update-*.json")))
            raise
    if code:
        write_failure(directory, RuntimeError(f"baseline child exited with code {code}"), phase="child_exit")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
