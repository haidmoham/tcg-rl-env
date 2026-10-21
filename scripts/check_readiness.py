"""Report pretraining evidence without loading weights, an engine, or a trainer."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from plan_campaign import summarize


ROOT = Path(__file__).resolve().parents[1]


def natural_game(game, deck_names):
    return (type(game) is dict and game.get("status") == "natural_terminal"
            and type(game.get("result")) is int and game["result"] in (0, 1, 2)
            and type(game.get("submissions")) is int and game["submissions"] > 0
            and type(game.get("wall_seconds")) in (int, float)
            and math.isfinite(game["wall_seconds"]) and game["wall_seconds"] >= 0
            and type(game.get("deck0")) is str and game["deck0"] in deck_names
            and type(game.get("deck1")) is str and game["deck1"] in deck_names)


def sha256_digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def evidence_bindings(record, root, required):
    root = Path(root)
    if type(record) is not dict or type(record.get("implementation_sha256")) is not dict:
        raise ValueError("evidence must contain implementation hashes")
    mismatches = []
    for name, expected in record.get("implementation_sha256", {}).items():
        if type(name) is not str:
            raise ValueError("invalid evidence source path")
        source = (root / name.replace("\\", "/")).resolve()
        if not source.is_relative_to(root.resolve()):
            raise ValueError("evidence source escapes repository")
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            mismatches.append(name)
    decks = root / "experiments/cabt-v1-candidate-decks.json"
    if record.get("deck_manifest_sha256") != hashlib.sha256(decks.read_bytes()).hexdigest():
        mismatches.append("deck_manifest")
    recorded = {name.replace("\\", "/") for name in record.get("implementation_sha256", {})}
    if not required <= recorded:
        mismatches.append("missing_implementation_hashes")
    return mismatches


def inspect_evidence(path, root=ROOT):
    root = Path(root)
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(record) is not dict:
        raise ValueError("evidence must be an object")
    required = {"tcg/cabt.py", "scripts/qualify_cabt.py"}
    if record.get("policy_inference"):
        required |= {"tcg/decision.py", "tcg/policy.py", "tcg/checkpoint.py", "tcg/cabt_observation.py"}
    mismatches = evidence_bindings(record, root, required)
    decks = root / "experiments/cabt-v1-candidate-decks.json"
    games = record.get("games", [])
    deck_names = {deck["name"] for deck in json.loads(decks.read_text(encoding="utf-8"))["decks"]}
    if type(games) is not list:
        games = []
        mismatches.append("malformed_games")
    well_formed = [game for game in games if natural_game(game, deck_names)]
    natural = len(well_formed)
    if not games or natural != len(games):
        mismatches.append("missing_or_invalid_natural_games")
    if not sha256_digest(record.get("engine_sha256")):
        mismatches.append("missing_engine_hash")
    valid = (record.get("schema") == "cabt-untrained-engineering/1"
             and record.get("status") == "diagnostic_passed"
             and type(record.get("optimizer_updates")) is int and record["optimizer_updates"] == 0
             and record.get("errors") == [] and not mismatches
             and type(record.get("policy_inference")) is bool
             and type(record.get("natural_games")) is int
             and natural == record["natural_games"] and natural > 0)
    if record.get("policy_inference"):
        valid = (valid and record.get("parameters_unchanged") is True
                 and record.get("initialization") == "random"
                 and sha256_digest(record.get("checkpoint_manifest_sha256")))
    return record, {"valid_current_evidence": valid, "stale_or_missing": mismatches,
                    "natural_games": natural,
                    "ordered_deck_pairs": len({(game["deck0"], game["deck1"]) for game in well_formed})}


def inspect_hp_evidence(path, expected_artifact, root=ROOT):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"tcg/cabt.py", "tcg/decision.py", "tcg/policy.py", "tcg/checkpoint.py",
                "tcg/cabt_observation.py", "scripts/qualify_cabt_hp.py"}
    mismatches = evidence_bindings(record, root, required)
    if not expected_artifact or record.get("engine_sha256") != expected_artifact:
        mismatches.append("engine_artifact_identity")
    observed = record.get("observed")
    samples = observed.get("negative_hp") if type(observed) is dict else None
    negative = (type(samples) is list and bool(samples) and all(
        type(sample) is dict and type(sample.get("hp")) is int and sample["hp"] < 0
        and type(sample.get("max_hp")) is int and sample["max_hp"] >= 0
        and type(sample.get("seat")) is int and sample["seat"] in (0, 1)
        and type(sample.get("area")) is int and sample["area"] in (4, 5)
        and type(sample.get("index")) is int and sample["index"] >= 0 for sample in samples))
    valid = (not mismatches and record.get("schema") == "cabt-hp-regression/1"
             and record.get("scope") == "DIRECTED_RESPONSE_DIAGNOSTIC_NOT_POLICY_PLAY"
             and record.get("status") == "reproduced_and_projected" and record.get("errors") == []
             and type(record.get("optimizer_updates")) is int and record["optimizer_updates"] == 0
             and type(record.get("policy_forward_calls")) is int and record["policy_forward_calls"] == 1
             and all(record.get(name) is True for name in ("projection_passed", "finite_legal_logits",
                                                          "finite_value", "parameters_unchanged"))
             and record.get("ordinary_policy_play_verified") is False
             and record.get("initialization") == "random" and record.get("device") == "cpu"
             and sha256_digest(record.get("checkpoint_manifest_sha256"))
             and sha256_digest(record.get("checkpoint_weights_sha256")) and negative
             and type(observed.get("prompt_context")) is int and observed["prompt_context"] == 14
             and type(observed.get("prompt_type")) is int and observed["prompt_type"] == 1
             and type(observed.get("remain_damage_counter")) is int and observed["remain_damage_counter"] > 0)
    return {"valid_current_hp_evidence": bool(valid), "stale_or_missing": mismatches,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def readiness(plan, evidence_paths, root=ROOT, hp_evidence_paths=()):
    budget = summarize(plan)
    records = [inspect_evidence(path, root) for path in evidence_paths]
    expected_artifact = plan["environment"]["cabt_candidate"]["binary_sha256"]
    hp_records = [inspect_hp_evidence(path, expected_artifact, root) for path in hp_evidence_paths]
    for record, summary in records:
        if not expected_artifact or record.get("engine_sha256") != expected_artifact:
            summary["valid_current_evidence"] = False
            summary["stale_or_missing"].append("engine_artifact_identity")
    native = any(summary["valid_current_evidence"] and summary["natural_games"] >= 64
                 and summary["ordered_deck_pairs"] == 36 for _, summary in records)
    policy = any(summary["valid_current_evidence"] and record.get("policy_inference")
                 for record, summary in records)
    return {
        "schema": "86-tcg-agent/pretraining-readiness/1",
        "scope": "formal_reference_campaign_not_small_baseline_authorization",
        "training_authorized": False,
        "ready_to_train": False,
        "optimizer_updates": 0,
        "checks": {
            "reference_plan": "passed",
            "native_64_game_matrix": "passed" if native else "not_verified",
            "unchanged_random_policy_native_game": "passed" if policy else "not_verified",
            "authoritative_seeded_replay": "formal_official_artifact_has_no_declared_seed_api",
            "native_hidden_world_invariance": "unqualified_for_formal_artifact_bounded_source_evidence_is_separate",
            "complete_prompt_effect_coverage": "not_verified_required_deck_effect_fixtures_incomplete",
            "signed_hp_pending_knockout": ("passed_directed_native_projection" if any(
                record["valid_current_hp_evidence"] for record in hp_records) else "not_verified_current_directed_fixture"),
            "deck_rules_and_printing_provenance": "not_verified_candidate_lists_only",
            "full_tcg_live_standard_coverage": "not_verified_complete_dated_legal_pool_and_rules_required",
            "training_update_and_resume": "formal_campaign_update_and_resume_unqualified",
            "whole_training_campaign_throughput": "formal_campaign_complete_loop_rate_unqualified",
        },
        "evidence": [{"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), **summary}
                     for path, (_, summary) in zip(evidence_paths, records)],
        "hp_evidence": hp_records,
        "budget": budget,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--hp-evidence", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("preserve existing evidence; choose a fresh output")
    plan = json.loads((ROOT / "experiments/r0-reference.json").read_text(encoding="utf-8"))
    result = readiness(plan, args.evidence, hp_evidence_paths=args.hp_evidence)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if result["ready_to_train"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
