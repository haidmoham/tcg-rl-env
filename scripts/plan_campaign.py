"""Check R0's statistical and time budget without executing the experiment."""

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def positive_rate(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("rate must be finite and greater than zero")
    return number


def summarize(plan, learner_rate=None, evaluation_rate=None):
    if plan["status"] != "proposal_not_runnable" or plan["execution_authorized"]:
        raise ValueError("R0 must remain a non-executing proposal")
    if learner_rate is not None:
        learner_rate = positive_rate(learner_rate)
    if evaluation_rate is not None:
        evaluation_rate = positive_rate(evaluation_rate)
    learning = plan["learning"]
    if any(learning[key] for key in (
        "behavior_cloning", "reward_shaping", "auxiliary_losses",
        "search_annotations", "train_search", "evaluation_search",
        "teacher_kl_coefficient", "teacher_value_coefficient",
    )):
        raise ValueError("reference proposal contains an undeclared learning-method change")
    if learning["gamma"] != 1.0 or plan["model"]["weights_from_prior_agents"]:
        raise ValueError("reference initialization/discount contract changed")
    confirm = plan["confirmation"]
    n, alpha = confirm["seed_blocks_per_matchup"], confirm["alpha"]
    if type(n) is not int or n <= 0 or not 0 < alpha < 1 or confirm["games_per_block"] != 2:
        raise ValueError("invalid fixed paired-block design")
    matchups = [(item["policy"], item["opponent"]) for item in confirm["matchups"]]
    required_matchups = {
        ("final_checkpoint", "initial_checkpoint"),
        ("initial_checkpoint", "frozen_heuristic"),
        ("final_checkpoint", "frozen_heuristic"),
    }
    if len(matchups) != 3 or set(matchups) != required_matchups:
        raise ValueError("confirmation requires the primary and both independent-reference matchups")
    improvement = confirm["independent_improvement"]
    if (improvement["interval"] != "difference_of_two_hoeffding_intervals_with_union_coverage"
            or not 0 < improvement["alpha"] < 1
            or improvement["claim"] != "descriptive_only_no_power_or_confirmatory_improvement_claim"
            or improvement["opponent"] != "frozen_heuristic"
            or not improvement["opponent_frozen_before_training"]
            or improvement["opponent_in_training_pool"]):
        raise ValueError("invalid independent-improvement interval design")
    difference_radius = 2 * math.sqrt(math.log(4 / improvement["alpha"]) / (2 * n))
    if (not 0 <= confirm["primary_score_threshold"] <= 1
            or not 0 <= confirm["power_alternative_score"] <= 1
            or not 0 < confirm["requested_minimum_power_bound"] < 1):
        raise ValueError("invalid primary score or power target")
    radius = math.sqrt(math.log(2 / alpha) / (2 * n))
    gap = confirm["power_alternative_score"] - confirm["primary_score_threshold"] - radius
    power_bound = 0.0 if gap <= 0 else 1 - math.exp(-2 * n * gap * gap)
    if power_bound < confirm["requested_minimum_power_bound"]:
        raise ValueError("sample plan does not meet its declared conservative power bound")
    campaign = plan["campaign"]
    for key in ("training_wall_hours_cap", "confirmation_wall_hours_reserved", "total_wall_hours_cap"):
        if not math.isfinite(campaign[key]) or campaign[key] <= 0:
            raise ValueError("wall-hour caps must be finite and greater than zero")
    if type(campaign["learner_decisions_cap"]) is not int or campaign["learner_decisions_cap"] <= 0:
        raise ValueError("learner decision cap must be a positive integer")
    if campaign["training_wall_hours_cap"] + campaign["confirmation_wall_hours_reserved"] > campaign["total_wall_hours_cap"]:
        raise ValueError("time reservations exceed the total campaign cap")
    games = n * confirm["games_per_block"] * len(matchups)
    evaluation_seconds = campaign["confirmation_wall_hours_reserved"] * 3600
    training_seconds = campaign["training_wall_hours_cap"] * 3600
    result = {
        "status": plan["status"],
        "pretraining_authorized": plan["pretraining_authorized"],
        "execution_authorized": plan["execution_authorized"],
        "scope": "arithmetic_only_no_environment_or_learning_execution",
        "independent_blocks_per_matchup": n,
        "two_sided_hoeffding_radius": radius,
        "primary_observed_score_must_exceed": confirm["primary_score_threshold"] + radius,
        "power_lower_bound_at_alternative": power_bound,
        "primary_power_scope": confirm["primary_interpretation"],
        "independent_improvement_difference_interval_radius": difference_radius,
        "independent_improvement_interval_range": [-1, 1],
        "independent_improvement_claim": improvement["claim"],
        "confirmation_games_total": games,
        "minimum_confirmation_games_per_second": games / evaluation_seconds,
        "learner_decisions_per_second_for_full_cap": campaign["learner_decisions_cap"] / training_seconds,
        "rates_are": "caller_supplied_scenarios_not_measurements",
        "blockers": plan["blockers"],
    }
    if learner_rate is not None:
        result["scenario_learner_decisions_upper_bound"] = min(
            campaign["learner_decisions_cap"], math.floor(learner_rate * training_seconds)
        )
        result["scenario_training_note"] = "excludes update, development-eval and checkpoint overhead unless caller rate includes them"
    if evaluation_rate is not None:
        hours = games / evaluation_rate / 3600
        result["scenario_confirmation_hours"] = hours
        result["scenario_confirmation_fits_reserved_budget"] = hours <= campaign["confirmation_wall_hours_reserved"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learner-decisions-per-second", type=positive_rate)
    parser.add_argument("--evaluation-games-per-second", type=positive_rate)
    args = parser.parse_args()
    plan = json.loads((ROOT / "experiments/r0-reference.json").read_text())
    print(json.dumps(summarize(plan, args.learner_decisions_per_second, args.evaluation_games_per_second), indent=2))


if __name__ == "__main__":
    main()
