"""Randomly initialized ragged transformer actor-critic. No simulator imports."""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .decision import ActorDecision, SUBMIT, Token, identifier, integer


@dataclass(frozen=True)
class Vocabulary:
    card_ids: tuple[str, ...]
    zones: tuple[str, ...] = ("global",)
    roles: tuple[str, ...] = ("state", "option")

    def __post_init__(self) -> None:
        for name in ("card_ids", "zones", "roles"):
            values = getattr(self, name)
            if type(values) is not tuple or len(set(values)) != len(values):
                raise ValueError(f"{name} must be an immutable unique tuple")
            for value in values:
                identifier(value, name)
        if not self.zones or not self.roles:
            raise ValueError("zone and role vocabularies must not be empty")


@dataclass(frozen=True)
class PolicyConfig:
    feature_width: int = 8
    max_event_positions: int = 32
    d_model: int = 128
    layers: int = 4
    heads: int = 4
    feedforward_width: int = 512
    dropout: float = 0.0
    activation: str = "relu"

    def __post_init__(self) -> None:
        integer(self.feature_width, "feature width", 1)
        integer(self.max_event_positions, "event positions", 1)
        for name in ("d_model", "layers", "heads", "feedforward_width"):
            integer(getattr(self, name), name, 1)
        if type(self.dropout) is not float:
            raise ValueError("dropout must be a float")
        if (self.d_model, self.layers, self.heads, self.feedforward_width, self.dropout) != (128, 4, 4, 512, 0.0):
            raise ValueError("architecture must match the R0 reference")
        if self.activation != "relu":
            raise ValueError("activation must match the R0 reference")


@dataclass(frozen=True)
class PolicyOutput:
    logits: Tensor
    values: Tensor
    legal_mask: Tensor
    option_ids: tuple[tuple[str, ...], ...]


class Policy(nn.Module):
    def __init__(self, vocabulary: Vocabulary, config: PolicyConfig = PolicyConfig()) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.config = config
        width = config.d_model
        self.card_embedding = nn.Embedding(len(vocabulary.card_ids) + 1, width)
        self.zone_embedding = nn.Embedding(len(vocabulary.zones), width)
        self.role_embedding = nn.Embedding(len(vocabulary.roles), width)
        self.owner_embedding = nn.Embedding(3, width)
        self.position_embedding = nn.Embedding(config.max_event_positions, width)
        self.numeric = nn.Linear(config.feature_width, width, bias=False)
        self.selection = nn.Linear(2, width, bias=False)
        self.prompt = nn.Linear(5, width, bias=False)
        self.submit_query = nn.Parameter(torch.randn(width) * 0.02)
        self.value_query = nn.Parameter(torch.randn(width) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, config.heads, config.feedforward_width,
                                       config.dropout, activation=config.activation,
                                       batch_first=True, norm_first=True)
            for _ in range(config.layers)
        ])
        self.final_norm = nn.LayerNorm(width)
        self.option_score = nn.Linear(width, 1)
        self.value_score = nn.Linear(width, 1)

    def _tokens(self, tokens: Sequence[Token]) -> Tensor:
        if not tokens:
            return self.value_query.new_empty((0, self.config.d_model))
        vocabulary = self.vocabulary
        cards = {value: index + 1 for index, value in enumerate(vocabulary.card_ids)}
        zones = {value: index for index, value in enumerate(vocabulary.zones)}
        roles = {value: index for index, value in enumerate(vocabulary.roles)}
        identities, features, positions, positioned = [], [], [], []
        for token in tokens:
            try:
                identities.append((0 if token.card_id is None else cards[token.card_id],
                                   zones[token.zone], roles[token.role], token.owner))
            except KeyError as error:
                raise ValueError("unknown card, zone, or role ID") from error
            if len(token.features) > self.config.feature_width:
                raise ValueError("numeric features exceed declared width")
            if token.position is not None and token.position >= self.config.max_event_positions:
                raise ValueError("event position exceeds declared history")
            features.append(token.features + (0.0,) * (self.config.feature_width - len(token.features)))
            positions.append(token.position or 0)
            positioned.append(token.position is not None)
        device = self.value_query.device
        indices = torch.tensor(identities, device=device, dtype=torch.long).reshape(-1, 4)
        numbers = self.value_query.new_tensor(features).reshape(-1, self.config.feature_width)
        result = (self.card_embedding(indices[:, 0]) + self.zone_embedding(indices[:, 1])
                  + self.role_embedding(indices[:, 2]) + self.owner_embedding(indices[:, 3])
                  + self.numeric(numbers))
        if any(positioned):
            position_indices = [index for index, present in enumerate(positioned) if present]
            values = self.position_embedding(torch.tensor(
                [positions[index] for index in position_indices], device=device, dtype=torch.long))
            result = result + torch.zeros_like(result).index_copy(
                0, torch.tensor(position_indices, device=device), values)
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite encoded token")
        return result

    def forward(self, decisions: Sequence[ActorDecision]) -> PolicyOutput:
        if not decisions:
            raise ValueError("empty decision batch")
        tokens, references, selections, metadata = [], {}, [], []
        lengths, option_ids, masks, offsets, spans = [], [], [], [], []
        for decision in decisions:
            if type(decision) is not ActorDecision:
                raise ValueError("Policy accepts only validated ActorDecision values")
            prompt = decision.prompt
            metadata.append([prompt.minimum, prompt.maximum, prompt.ordered,
                             prompt.allow_repeats, prompt.allocation_total or 0])
            start = len(tokens)
            tokens.extend(decision.tokens)
            selections.extend([(0, 0)] * len(decision.tokens))
            by_id = {option.option_id: option for option in prompt.options}
            occurrences = []
            if prompt.ordered:
                occurrences.extend((by_id[selected], 1, ordinal + 1)
                                   for ordinal, selected in enumerate(decision.selected))
            for option in prompt.options:
                count = decision.selected.count(option.option_id)
                ordinal = decision.selected.index(option.option_id) + 1 if count and prompt.ordered else 0
                occurrences.append((option, count, ordinal))
            for option, count, ordinal in occurrences:
                if option.state_indices:
                    references.setdefault(len(option.state_indices), []).append(
                        (len(tokens), tuple(start + index for index in option.state_indices)))
                tokens.append(option.token)
                selections.append((count, ordinal))
            history_length = len(decision.selected) if prompt.ordered else 0
            offset = 1 + len(decision.tokens) + history_length
            ids = tuple(option.option_id for option in prompt.options) + (SUBMIT,)
            legal = decision.choices()
            if not legal:
                raise ValueError("all-invalid decision")
            spans.append((start, len(tokens)))
            lengths.append(2 + len(tokens) - start)
            offsets.append(offset)
            option_ids.append(ids)
            masks.append([option_id in legal for option_id in ids])
        embedded = self._tokens(tokens)
        reference_vectors = torch.zeros_like(embedded)
        for group in references.values():
            destinations, sources = zip(*group)
            indices = torch.tensor(sources, device=embedded.device)
            means = embedded[indices].mean(1)
            reference_vectors = reference_vectors.index_copy(
                0, torch.tensor(destinations, device=embedded.device), means)
        embedded = embedded + reference_vectors
        if any(decision.prompt.options for decision in decisions):
            embedded = embedded + self.selection(self.value_query.new_tensor(selections).reshape(-1, 2))
        prompt_vectors = self.value_query + self.prompt(self.value_query.new_tensor(metadata))
        rows = [torch.cat((prompt_vectors[index:index + 1], embedded[start:end], self.submit_query[None]), 0)
                for index, (start, end) in enumerate(spans)]
        padded = nn.utils.rnn.pad_sequence(rows, batch_first=True)
        padding = torch.arange(padded.shape[1], device=padded.device)[None, :] >= torch.tensor(
            lengths, device=padded.device)[:, None]
        for block in self.blocks:
            padded = block(padded, src_key_padding_mask=padding)
        encoded = self.final_norm(padded)
        option_vectors = [encoded[i, offset:length] for i, (offset, length) in enumerate(zip(offsets, lengths))]
        logits = self.option_score(nn.utils.rnn.pad_sequence(option_vectors, batch_first=True)).squeeze(-1)
        legal_mask = torch.zeros_like(logits, dtype=torch.bool)
        for i, mask in enumerate(masks):
            legal_mask[i, :len(mask)] = torch.tensor(mask, device=logits.device)
        values = self.value_score(encoded[:, 0]).squeeze(-1).tanh()
        if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
            raise ValueError("nonfinite policy output")
        return PolicyOutput(logits.masked_fill(~legal_mask, -torch.inf), values,
                            legal_mask, tuple(option_ids))
