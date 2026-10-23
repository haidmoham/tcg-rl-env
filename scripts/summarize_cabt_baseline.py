"""Summarize recorded CABT baseline experiments without loading models or engines."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import gzip
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tcg.decision import ActorDecision, LegalPrompt, Option, SUBMIT, Token
from tcg.rollout import Episode, EpisodeEnd, EpisodeProvenance, LearnerStep, admit_rollout


def read_object(path):
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def progress_snapshot(path):
    data = path.read_bytes() if path.exists() else b""
    lines = data.splitlines(keepends=True)
    partial = bool(lines and not lines[-1].endswith(b"\n"))
    if partial:
        lines.pop()
    events = []
    for index, line in enumerate(lines, 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object in {path}, line {index}")
        events.append(value)
    return events, partial


def minibatch_summary(rows):
    sample_uses = sum(row["learner_decisions"] for row in rows)
    metrics = ("loss", "policy_loss", "value_loss", "entropy", "approximate_kl", "clip_fraction")
    return {
        "recorded_optimizer_steps": len(rows),
        "learner_sample_uses_across_epochs": sample_uses,
        "sample_weighted_means": {
            key: sum(row[key] * row["learner_decisions"] for row in rows) / sample_uses
            if sample_uses else None for key in metrics
        },
        "gradient_norm_before_clip_mean_per_optimizer_step":
            sum(row["gradient_norm_before_clip"] for row in rows) / len(rows) if rows else None,
        "gradient_norm_before_clip_max":
            max((row["gradient_norm_before_clip"] for row in rows), default=None),
        "summed_minibatch_wall_seconds": sum(row["wall_seconds"] for row in rows),
    }


def distribution(values):
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "population_sd": statistics.pstdev(values) if values else None}


def retained_rollout(directory, receipt, launch):
    artifact = receipt["rollout"]
    path = (directory / artifact["path"]).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("retained rollout is outside the run directory")
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["file_sha256"]:
        raise ValueError("retained rollout compressed SHA-256 mismatch")
    source_hashes = {}
    for name in ("tcg/decision.py", "tcg/rollout.py"):
        source_hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if source_hashes[name] != launch["source_sha256"][name]:
            raise ValueError("diagnostic decision or target semantics differ from launch")
    digest, rows, footer = hashlib.sha256(), [], None
    with gzip.open(path, "rb") as stream:
        first = stream.readline()
        header = json.loads(first)
        if header["schema"] != "86-tcg-agent/private-loss-inputs/1":
            raise ValueError("unsupported retained rollout schema")
        digest.update(first.replace(b"\r\n", b"\n"))
        for line in stream:
            value = json.loads(line)
            if footer is not None:
                raise ValueError("data after retained rollout footer")
            if value.get("footer") is True:
                footer = value
            else:
                digest.update(line.replace(b"\r\n", b"\n"))
                rows.append(value)
    if (footer is None or footer["rows"] != len(rows) or len(rows) != artifact["rows"]
            or len(rows) != receipt["update"]["learner_decisions"]
            or footer["sha256"] != digest.hexdigest()
            or artifact["content_sha256"] != digest.hexdigest()):
        raise ValueError("retained rollout footer or content SHA-256 mismatch")
    if (header["identity"]["protocol_sha256"] != launch["protocol_sha256"]
            or header["identity"]["projection_sha256"] != launch["source_sha256"]["tcg/cabt_observation.py"]):
        raise ValueError("retained rollout identity differs from launch")
    episodes, current, forced, genuine = [], [], Counter(), []
    legal_choice_counts, native_prompt_frames = Counter(), Counter()
    frame_option_count_max = 0
    for index, row in enumerate(rows):
        raw = row["step"]
        provenance = EpisodeProvenance(**raw["provenance"])
        if (provenance.policy_version != header["identity"]["learner_policy_version"]
                or provenance.policy_version + 1 != receipt["counts"]["accepted_updates"]):
            raise ValueError("retained rollout learner policy version mismatch")
        decision, prompt = raw["decision"], dict(raw["decision"]["prompt"])
        # Token contents do not affect choices; native options keep their recorded IDs.
        prompt["options"] = tuple(Option(option["option_id"], Token()) for option in prompt["options"])
        actor = ActorDecision((), LegalPrompt(**prompt), tuple(decision["selected"]))
        step = LearnerStep(provenance, raw["sequence"], actor, raw["action_id"],
                           raw["old_log_probability"], raw["old_value"], raw["reward"])
        choices = actor.choices()
        legal_choice_counts[len(choices)] += 1
        if not actor.selected:
            native_fields = {}
            for role in ("prompt.type", "prompt.context"):
                fields = [token["features"] for token in decision["tokens"] if token["role"] == role]
                if (len(fields) != 1 or len(fields[0]) != 1
                        or type(fields[0][0]) is not int or fields[0][0] < 0):
                    raise ValueError("initial learner frame lacks one exact native prompt type/context value")
                native_fields[role] = fields[0][0]
            native_prompt_frames[(native_fields["prompt.type"], native_fields["prompt.context"])] += 1
            frame_option_count_max = max(frame_option_count_max, len(actor.prompt.options))
        if len(choices) == 1:
            if step.old_log_probability != 0:
                raise ValueError("forced action has nonzero recorded log probability")
            forced["submit" if choices[0] == SUBMIT else "pick"] += 1
        else:
            genuine.append(index)
        if current and current[0].provenance != provenance:
            episodes.append(Episode(current[0].provenance, tuple(current), EpisodeEnd.TERMINAL))
            current = []
        current.append(step)
    if current:
        episodes.append(Episode(current[0].provenance, tuple(current), EpisodeEnd.TERMINAL))
    targets = admit_rollout(tuple(episodes))
    agreement = {}
    for name, recomputed in (("advantage", targets.advantages), ("return", targets.returns),
                              ("normalized_advantage", targets.normalized_advantages)):
        recorded = [row[name] for row in rows]
        if not all(type(value) in (int, float) and math.isfinite(value) for value in recorded):
            raise ValueError("nonfinite retained target")
        agreement[name + "_max_absolute_error"] = max(abs(a - b) for a, b in zip(recorded, recomputed))
    if any(error > 1e-12 for error in agreement.values()):
        raise ValueError("retained GAE targets disagree with canonical recurrence")

    def value_metrics(indices):
        errors = [rows[index]["return"] - rows[index]["step"]["old_value"] for index in indices]
        returns = [rows[index]["return"] for index in indices]
        variance = statistics.pvariance(returns) if returns else 0
        return {"learner_microsteps": len(indices),
                "mse": statistics.mean(error ** 2 for error in errors) if errors else None,
                "explained_variance": 1 - statistics.pvariance(errors) / variance if variance else None,
                "target_population_variance": variance}

    lengths = [len(episode.steps) for episode in episodes]
    outcomes = Counter({"wins": 0, "losses": 0, "draws": 0})
    for episode in episodes:
        outcomes[{1: "wins", -1: "losses", 0: "draws"}[episode.steps[-1].reward]] += 1
    return {
        "artifact": artifact, "compressed_sha256_verified": True,
        "normalized_lf_footer_sha256_verified": True, "semantics_source_sha256": source_hashes,
        "learner_microsteps": len(rows), "forced_microsteps": sum(forced.values()),
        "forced_submit_microsteps": forced["submit"], "forced_pick_microsteps": forced["pick"],
        "genuine_choice_microsteps": len(genuine), "genuine_choice_fraction": len(genuine) / len(rows),
        "observed_learner_prompt_workload": {
            "legal_choice_count_microstep_histogram": dict(sorted(legal_choice_counts.items())),
            "legal_choice_count_max": max(legal_choice_counts, default=None),
            "initial_native_prompt_frames": sum(native_prompt_frames.values()),
            "native_type_context_frame_histogram": {
                f"{kind}/{context}": count for (kind, context), count in sorted(native_prompt_frames.items())},
            "native_frame_option_count_max": frame_option_count_max,
            "scope": "Observed admitted learner frames only; excludes opponent frames and is not exhaustive coverage.",
        },
        "episode_lengths_in_learner_microsteps": {**distribution(lengths), "median": statistics.median(lengths),
                                                  "minimum": min(lengths), "maximum": max(lengths)},
        "terminal_learner_outcomes": dict(outcomes), "censored_episode_ids": header["censored_episode_ids"],
        "preupdate_value_against_frozen_gae_returns": {
            "all_microsteps": value_metrics(range(len(rows))), "genuine_choice_microsteps": value_metrics(genuine)},
        "raw_advantages": distribution([row["advantage"] for row in rows]),
        "normalized_advantages": distribution([row["normalized_advantage"] for row in rows]),
        "canonical_gae_recurrence_agreement": agreement,
    }


def summarize(directory):
    launch = read_object(directory / "launch.json")
    protocol_bytes = (directory / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    if hashlib.sha256(protocol_bytes).hexdigest() != launch["protocol_sha256"]:
        raise ValueError("retained protocol bytes differ from the launch identity")
    events, partial = progress_snapshot(directory / "progress.jsonl")
    counts = [event for event in events if event.get("event") == "collection_counts"]
    latest_counts = counts[-1] if counts else {}
    games = [event for event in events if event.get("event") == "game"]
    updates = []
    for path in sorted(directory.glob("accepted-update-*.json")):
        receipt = read_object(path)
        report = receipt["update"]
        rollout = retained_rollout(directory, receipt, launch)
        minibatches = minibatch_summary(report["minibatch_metrics"])
        choice_fraction = rollout["genuine_choice_fraction"]
        choice_metrics = {
            key: minibatches["sample_weighted_means"][key] / choice_fraction if choice_fraction else None
            for key in ("entropy", "approximate_kl", "clip_fraction")
        }
        choice_metrics["final_epoch_mean_old_minus_current_logp"] = (
            report["epoch_reports"][-1]["mean_old_minus_current_logp"] / choice_fraction
            if choice_fraction else None)
        updates.append({
            "receipt": path.name, "policy_version": receipt["counts"]["accepted_updates"],
            "checkpoint": receipt["checkpoint"], "weights_sha256": receipt["weights_sha256"],
            "learner_decisions": report["learner_decisions"], "optimizer_steps": report["optimizer_steps"],
            "fit_and_checkpoint_wall_seconds": receipt["fit_and_checkpoint_wall_seconds"],
            "parameter_l2_movement": receipt["parameter_l2_movement"],
            "epoch_reports": report["epoch_reports"],
            "minibatches": minibatches,
            "ppo_means_over_genuine_choices": choice_metrics,
            "retained_rollout_diagnostics": rollout,
        })
    phases = {}
    for phase in ("training", "evaluation"):
        records = [game for game in games if game["phase"] == phase]
        endings = Counter(game["end"] for game in records)
        admitted = sum(game["learner_micro_decisions"] for game in records if game["end"] == "terminal")
        wall = sum(game["wall_seconds"] for game in records)
        phases[phase] = {
            "recorded_finished_games": len(records), "natural_terminal_games": endings["terminal"],
            "censored_games": endings["administrative_truncation"],
            "infrastructure_fault_games": endings["infrastructure_failure"], "end_counts": dict(endings),
            "terminal_eligible_learner_decisions": admitted,
            "raw_learner_decisions": sum(game["learner_micro_decisions"] for game in records),
            "summed_game_wall_seconds": wall,
            "admitted_decisions_per_summed_game_wall_second": admitted / wall if wall else None,
            "finished_games_per_summed_game_wall_second": len(records) / wall if wall else None,
            "ordered_response_checks": sum(game.get("ordered_response_checks", 0) for game in records),
            "multi_selection_submissions": sum(game.get("multi_selection_submissions", 0) for game in records),
        }
    outcome_fields = ("status", "stop_reason", "counts", "last_good_weights_sha256", "unused_admitted_decisions",
                      "native_active_game_resume_verified", "cuda_optimizer_resume_verified", "evaluation",
                      "error_type", "phase", "affected_update_checkpoint_allowed")
    outcomes = {}
    for name in ("training-result.json", "result.json", "failure.json", "interruption.json"):
        if (directory / name).exists():
            value = read_object(directory / name)
            outcomes[name] = {key: value[key] for key in outcome_fields if key in value}
            if name == "interruption.json":
                outcomes[name].update({key: value[key] for key in (
                    "classification", "accepted_updates", "completed_optimizer_minibatches",
                    "affected_terminal_prompt_count", "changed_selection_order_count",
                    "native_counterfactual_qualified", "resume_partial_update_allowed") if key in value})
    checkpoints = []
    for manifest in sorted((directory / "checkpoints").glob("*/manifest.json")):
        retained = read_object(manifest)
        payload = retained["payload"]
        checkpoints.append({"manifest": str(manifest.relative_to(directory)),
                            **{key: payload[key] for key in ("identity", "counters", "weights_sha256", "files")}})
    return {
        "schema": "86-tcg-agent/cabt-baseline-summary/1",
        "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": directory.name,
        "status": "interrupted_invalid_experiment" if "interruption.json" in outcomes else
                  "failed" if "failure.json" in outcomes else
                  "complete" if "result.json" in outcomes else "in_progress_or_stopped_without_result",
        "progress_snapshot": {"complete_events": len(events), "ignored_unterminated_last_line": partial},
        "identity": {"checkout": launch["checkout"], "source_sha256": launch["source_sha256"],
                     "protocol_sha256": launch["protocol_sha256"], "engine": protocol["engine"],
                     "initial_weights_sha256": protocol["model"]["initial_weights_sha256"],
                     "deck_manifest_sha256": protocol["deck_manifest_sha256"],
                     "projection_sha256": protocol["projection_sha256"], "device": launch["device"]},
        "collection": phases,
        "latest_recorded_collection_counts": latest_counts,
        "last_recorded_end_to_end_elapsed_seconds": latest_counts.get("elapsed_seconds"),
        "accepted_updates": updates,
        "all_recorded_fit_minibatches": minibatch_summary(
            [event for event in events if event.get("event") == "fit_minibatch"]),
        "retained_checkpoints": checkpoints, "outcomes": outcomes,
        "interpretation": [
            "Collection rates use summed completed-game wall time and exclude fitting and between-game overhead.",
            "End-to-end elapsed is the last recorded collection-count timestamp, not live elapsed or completed fit time.",
            "Minibatch losses repeat samples across epochs; gradient norms are per optimizer step before clipping.",
            "Self-play outcomes do not establish learning or strength; fixed evaluation is descriptive only.",
            "Retained rollout diagnostics count each admitted learner microstep once, including forced actions.",
            "Genuine choices have more than one legal action, including optional submit; forced picks and submits are separate.",
            "Native prompt histogram keys are type/context values from recorded tokens; only empty-selected initial learner microsteps count a frame, including optional empty submit. Later picks and submissions do not count again.",
            "Choice-only entropy, approximate KL and clipping divide full-pass means by the genuine-choice fraction; forced-action contributions are zero. These are descriptive denominators and do not change the PPO objective or stopping gate.",
            "Minibatch means span changing parameters during fitting; only the final-epoch full-rollout KL describes the accepted final parameters.",
            "Preupdate value MSE and explained variance use frozen GAE returns, not Monte Carlo terminal rewards.",
            "Explained variance is 1 minus population variance of return-minus-value divided by return variance; zero target variance yields null.",
            "GAE agreement recomputes canonical complete-episode gamma=1, lambda=0.95 targets with zero terminal bootstrap and whole-rollout normalization.",
            "An in-progress snapshot can omit the active game or update and does not establish process liveness.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, help="write an exclusive new JSON receipt as well as stdout")
    args = parser.parse_args()
    output = json.dumps(summarize(args.run_directory.resolve()), indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
