"""Prepare blocked confirmation assignments without loading policies or an engine."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import re


MATCHUPS = (
    ("final_checkpoint", "initial_checkpoint"),
    ("initial_checkpoint", "frozen_heuristic"),
    ("final_checkpoint", "frozen_heuristic"),
)
POOL_ASSIGNMENT = "first_four_training_last_two_heldout_lists_not_heldout_archetypes"
STRATA = tuple(itertools.product(("training", "heldout"), repeat=2))


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stream_seed(domain, *components):
    return int.from_bytes(hashlib.sha256(encode([domain, *components])).digest()[:8], "big")


def prepare(plan_bytes, deck_manifest_bytes, frozen_heuristic_sha256):
    if type(frozen_heuristic_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", frozen_heuristic_sha256) is None:
        raise ValueError("an explicit frozen heuristic SHA-256 is required")
    plan = json.loads(plan_bytes)
    decks_manifest = json.loads(deck_manifest_bytes)
    if (type(plan) is not dict or plan.get("schema") != "86-tcg-agent/reference-proposal/1"
            or type(decks_manifest) is not dict or decks_manifest.get("schema") != "cabt-candidate-decks/1"):
        raise ValueError("unsupported plan or deck manifest schema")
    if (plan.get("pretraining_authorized") is not True or plan.get("execution_authorized") is not False
            or plan.get("authorization_scope") != "pretraining_implementation_and_offline_validation_only_no_training_or_live_play"):
        raise ValueError("confirmation preparation requires pretraining-only authorization")
    confirmation = plan["confirmation"]
    if (type(confirmation["seed_blocks_per_matchup"]) is not int
            or confirmation["seed_blocks_per_matchup"] != 800
            or type(confirmation["games_per_block"]) is not int or confirmation["games_per_block"] != 2
            or tuple((item["policy"], item["opponent"]) for item in confirmation["matchups"]) != MATCHUPS
            or confirmation["seed_assignment"] != "disjoint_domains_per_matchup_same_fixed_deck_strata"):
        raise ValueError("unsupported confirmation design")
    domain = confirmation["seed_domain"]
    assignment_seed = plan["model"]["training_seed"]
    if type(domain) is not str or not domain or type(assignment_seed) is not int or assignment_seed < 0:
        raise ValueError("invalid confirmation seed domain or assignment seed")
    environment = plan["environment"]
    if (type(environment["engine_seed_control_verified"]) is not bool
            or type(decks_manifest["engine_version"]) is not int
            or type(environment["cabt_candidate"]["engine_version"]) is not int
            or environment["training_decks"] != 4 or environment["confirmation_heldout_decks"] != 2
            or decks_manifest["assignment"] != POOL_ASSIGNMENT
            or decks_manifest["engine_version"] != environment["cabt_candidate"]["engine_version"]):
        raise ValueError("unsupported deck-pool assignment or engine version")
    decks = decks_manifest["decks"]
    if type(decks) is not list or len(decks) != 6:
        raise ValueError("exactly four training and two heldout deck lists are required")
    deck_records = []
    names, card_lists = set(), set()
    for index, deck in enumerate(decks):
        name, cards = deck["name"], deck["card_ids"]
        if type(name) is not str or not name or name in names:
            raise ValueError("deck names must be nonempty and unique")
        if (type(cards) is not list or len(cards) != 60 or deck["count"] != 60
                or any(type(card) is not int or card <= 0 for card in cards)
                or tuple(sorted(cards)) in card_lists):
            raise ValueError("deck lists must contain 60 exact card IDs and distinct multisets")
        names.add(name)
        card_lists.add(tuple(sorted(cards)))
        deck_records.append({"name": name, "pool": "training" if index < 4 else "heldout",
                             "deck_sha256": digest(encode(cards)), "card_ids": cards})
    pools = {pool: [deck["deck_sha256"] for deck in deck_records if deck["pool"] == pool]
             for pool in ("training", "heldout")}
    schedule = []
    for policy_pool, opponent_pool in STRATA:
        pairs = list(itertools.product(pools[policy_pool], pools[opponent_pool]))
        random.Random(stream_seed(domain, assignment_seed, "pair_order", policy_pool, opponent_pool)).shuffle(pairs)
        schedule.extend({"stratum": [policy_pool, opponent_pool], "policy_deck_sha256": pairs[index % len(pairs)][0],
                         "opponent_deck_sha256": pairs[index % len(pairs)][1]} for index in range(200))
    random.Random(stream_seed(domain, assignment_seed, "block_order")).shuffle(schedule)
    matchups, seeds = [], set()
    for policy, opponent in MATCHUPS:
        matchup_id = f"{policy}__vs__{opponent}"
        matchup_domain = f"{domain}/{matchup_id}"
        blocks = []
        for index, assignment in enumerate(schedule):
            block_id = f"{matchup_id}/block-{index:04d}"
            native_seed = stream_seed(matchup_domain, index, "native_engine")
            if native_seed in seeds:
                raise ValueError("assigned seed collision")
            seeds.add(native_seed)
            games = []
            for leg in (0, 1):
                roles = []
                for role, identity, deck_key, seat in (
                    ("policy", policy, "policy_deck_sha256", leg),
                    ("opponent", opponent, "opponent_deck_sha256", 1 - leg),
                ):
                    policy_seed = stream_seed(matchup_domain, index, leg, role, "policy_sampling")
                    if policy_seed in seeds:
                        raise ValueError("assigned seed collision")
                    seeds.add(policy_seed)
                    roles.append({"seat": seat, "role": role, "policy_identity": identity,
                                  "deck_sha256": assignment[deck_key], "policy_sampling_seed": policy_seed})
                games.append({"game_id": f"{block_id}/leg-{leg}", "leg": leg, "policy_seat": leg,
                              "native_engine_seed": native_seed, "seats": sorted(roles, key=lambda role: role["seat"])})
            blocks.append({"block_id": block_id, "block_index": index, **assignment,
                           "native_engine_seed": native_seed, "games": games})
        matchups.append({"matchup_id": matchup_id, "seed_domain": matchup_domain,
                         "policy": policy, "opponent": opponent, "blocks": blocks})
    manifest = {
        "schema": "86-tcg-agent/confirmation-assignments/1",
        "status": "blocked_pretraining_assignments_only",
        "execution_authorized": False,
        "pretraining_authorized": True,
        "plan_sha256": digest(plan_bytes),
        "deck_manifest_sha256": digest(deck_manifest_bytes),
        "policy_bindings": {
            "frozen_heuristic": {"sha256": frozen_heuristic_sha256, "status": "caller_supplied_frozen_identity_not_loaded"},
            "initial_checkpoint": {"sha256": None, "status": "initial_checkpoint_binding_required"},
            "final_checkpoint": {"sha256": None, "status": "bind_last_complete_verified_update_at_fixed_cap"},
        },
        "blocks_per_matchup": 800,
        "games_per_block": 2,
        "games_total": 4800,
        "deck_scope": decks_manifest["status"],
        "deck_records": deck_records,
        "design": {
            "status": "reconstructed_exact_choice_requires_canonical_protocol_update",
            "strata": [{"policy_pool": a, "opponent_pool": b, "blocks_per_matchup": 200} for a, b in STRATA],
            "ordered_pairs_within_stratum": "counts_differ_by_at_most_one_including_self_deck_pairs",
            "assignment_shuffle": "Python_random_MT19937_fixed_seed_derived_by_sha256",
            "assignment_seed": assignment_seed,
            "same_exact_deck_schedule_for_all_matchups": True,
            "heldout_involving_blocks_per_matchup": 600,
            "seat_pairing": "policy_seat_0_then_1_roles_keep_their_decks_and_swap_seats",
        },
        "randomness": {
            "seed_domain": domain,
            "derivation": "first_8_bytes_big_endian_of_sha256_canonical_JSON_component_array",
            "native_seed_assignment": "same_seed_for_both_legs_of_block_disjoint_between_blocks_and_matchups",
            "policy_seed_assignment": "distinct_stream_per_matchup_block_leg_and_policy_role",
            "engine_seed_control_verified": environment["engine_seed_control_verified"],
            "policy_sampling_pairing_verified": False,
            "seed_api_binding_verified": False,
            "qualification": "assigned_integers_are_not_engine_seed_control_or_independent_native_chance_evidence",
            "paired_assignment_limit": "reset_and_policy_streams_require_qualification_divergent_play_need_not_share_later_chance",
        },
        "execution_blockers": ["training_and_confirmation_execution_not_authorized", "initial_and_final_checkpoint_bindings_required",
                               "native_seed_API_binding_and_replay_qualification_required", "policy_sampling_stream_qualification_required",
                               "deck_and_engine_qualification_required"],
        "matchups": matchups,
    }
    manifest["manifest_sha256"] = digest(encode(manifest))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--deck-manifest", type=Path, required=True)
    parser.add_argument("--frozen-heuristic-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = prepare(args.plan.read_bytes(), args.deck_manifest.read_bytes(), args.frozen_heuristic_sha256)
    with args.output.open("xb") as output:
        output.write(encode(manifest) + b"\n")
    print(f"prepared {manifest['games_total']} blocked assignments; sha256={manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
