"""Exact categorical policy over complete native-validated unordered choice sets."""

import itertools
import torch

BEHAVIOR_POLICY = "entity-ppo-choice-sets/1"
MAXIMUM_CHOICE_SETS = 128


def action_count(record):
    return len(record.get("choiceSets", record["candidates"]))


def checked_choice_sets(record):
    sets = record.get("choiceSets")
    if sets is None:
        return None
    minimum, maximum = record.get("choiceSetMinimum"), record.get("choiceSetMaximum")
    count = len(record["candidates"])
    if (record["kind"] != "selection" or record.get("selectionCardinality") != "joint-unordered"
            or type(minimum) is not int or type(maximum) is not int or not 0 <= minimum <= maximum <= count
            or not isinstance(sets, list) or not 1 <= len(sets) <= MAXIMUM_CHOICE_SETS):
        raise ValueError("bounded complete unordered choice-set domain required")
    expected = []
    for size in range(minimum, maximum + 1):
        for indices in itertools.combinations(range(count), size):
            expected.append(list(indices))
            if len(expected) > MAXIMUM_CHOICE_SETS:
                raise ValueError("complete choice-set domain exceeds 128")
    if sets != expected or any(type(i) is not int for group in sets for i in group):
        raise ValueError("choice sets must enumerate the complete domain exactly once in canonical order")
    return sets


def attach_choice_sets(batch, records, device):
    domains = [checked_choice_sets(record) for record in records]
    if not any(domain is not None for domain in domains):
        return batch
    domains = [domain if domain is not None else [[i] for i in range(len(record["candidates"]))]
               for record, domain in zip(records, domains)]
    maximum = max(map(len, domains))
    members = max(1, max(len(group) for domain in domains for group in domain))
    indices = torch.zeros((len(records), maximum, members), dtype=torch.long, device=device)
    present = torch.zeros_like(indices, dtype=torch.bool)
    mask = torch.zeros((len(records), maximum), dtype=torch.bool, device=device)
    for row, (record, domain) in enumerate(zip(records, domains)):
        mask[row, :len(domain)] = torch.tensor(record.get("legalMask", [True] * len(domain)), device=device)
        for option, group in enumerate(domain):
            if group:
                indices[row, option, :len(group)] = torch.tensor(group, device=device)
                present[row, option, :len(group)] = True
    batch["choice_set_indices"], batch["choice_set_members"], batch["choice_set_mask"] = indices, present, mask
    return batch


def aggregate_choice_logits(logits, indices, members, mask):
    selected = logits[:, None, :].expand(-1, indices.shape[1], -1).gather(2, indices)
    summed = torch.where(members, selected, torch.zeros_like(selected)).sum(-1)
    return summed.masked_fill(~mask, float("-inf"))
