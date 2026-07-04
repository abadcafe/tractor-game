"""Torch Transformer policy/value model for Tractor self-play."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tensorize import (
    OBSERVATION_COMPONENT_COUNT,
    ArgumentPrefixTensorBatch,
    ObservationTensorBatch,
    observation_component_tensors,
)
from server.training.vocab_schema import VOCAB_SCHEMA


@dataclass(frozen=True, slots=True)
class ArgumentPrefixScores:
    """Next-argument logits for one batch of prefixes."""

    argument_logits: Tensor


@dataclass(frozen=True, slots=True)
class ObservationEncoding:
    """Reusable encoded observation memory for policy/value heads."""

    memory: Tensor
    memory_padding_mask: Tensor
    observation_context: Tensor


class TractorPolicyModel(nn.Module):
    """Shared observation encoder with semantic argument decoder."""

    def __init__(
        self,
        *,
        d_model: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self._token_type_embedding = _embedding(
            VOCAB_SCHEMA.token_type_vocab_size, d_model
        )
        self._segment_embedding = _embedding(
            VOCAB_SCHEMA.segment_vocab_size, d_model
        )
        self._field_embedding = _embedding(
            VOCAB_SCHEMA.field_vocab_size, d_model
        )
        self._value_embedding = _embedding(
            VOCAB_SCHEMA.value_vocab_size, d_model
        )
        self._suit_embedding = _embedding(
            VOCAB_SCHEMA.suit_vocab_size, d_model
        )
        self._rank_embedding = _embedding(
            VOCAB_SCHEMA.rank_vocab_size, d_model
        )
        self._points_embedding = _embedding(
            VOCAB_SCHEMA.points_vocab_size, d_model
        )
        self._color_embedding = _embedding(
            VOCAB_SCHEMA.color_vocab_size, d_model
        )
        self._role_embedding = _embedding(
            VOCAB_SCHEMA.role_vocab_size, d_model
        )
        self._trick_age_embedding = _embedding(
            VOCAB_SCHEMA.trick_age_vocab_size, d_model
        )
        self._trick_state_embedding = _embedding(
            VOCAB_SCHEMA.trick_state_vocab_size, d_model
        )
        self._play_order_embedding = _embedding(
            VOCAB_SCHEMA.play_order_vocab_size, d_model
        )
        self._count_embedding = _embedding(
            VOCAB_SCHEMA.count_vocab_size, d_model
        )
        self._play_width_embedding = _embedding(
            VOCAB_SCHEMA.play_width_vocab_size, d_model
        )
        self._event_age_embedding = _embedding(
            VOCAB_SCHEMA.event_age_vocab_size, d_model
        )
        self._categorical_projection = nn.Linear(
            OBSERVATION_COMPONENT_COUNT * d_model,
            d_model,
            bias=False,
        )
        self._numeric_projection = nn.Linear(
            NUMERIC_FEATURE_COUNT * 2,
            d_model,
            bias=False,
        )
        observation_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self._observation_encoder = nn.TransformerEncoder(
            observation_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self._argument_embedding = _embedding(
            SEMANTIC_CODEC.argument_vocab_size, d_model
        )
        self._argument_position_embedding = nn.Embedding(
            SEMANTIC_CODEC.max_argument_tokens, d_model
        )
        argument_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self._argument_decoder = nn.TransformerDecoder(
            argument_layer,
            num_layers=1,
        )
        self._decision_projection = nn.Linear(d_model * 2, d_model)
        self._argument_head = nn.Linear(
            d_model, SEMANTIC_CODEC.argument_vocab_size
        )
        self._value_head = nn.Linear(d_model, 1)

    def encode_observations(
        self,
        observation: ObservationTensorBatch,
    ) -> ObservationEncoding:
        """Encode observations once for value and argument decoding."""
        components = observation_component_tensors(observation)
        memory_padding_mask = components.token_type_ids.eq(
            VOCAB_SCHEMA.obs_pad_id
        )
        memory = self._encode_observation(
            observation,
            memory_padding_mask=memory_padding_mask,
        )
        query_mask = components.segment_ids.eq(
            VOCAB_SCHEMA.segment_action_query_id
        ) & (~memory_padding_mask)
        observation_context = _query_or_all_mean(
            memory,
            padding_mask=memory_padding_mask,
            query_mask=query_mask,
        )
        return ObservationEncoding(
            memory=memory,
            memory_padding_mask=memory_padding_mask,
            observation_context=observation_context,
        )

    def select_observation_encoding(
        self,
        encoding: ObservationEncoding,
        *,
        active_indices: tuple[int, ...],
    ) -> ObservationEncoding:
        """Select rows from a reusable observation encoding."""
        assert active_indices
        if active_indices == tuple(
            range(int(encoding.memory.shape[0]))
        ):
            return encoding
        index = torch.tensor(
            active_indices,
            dtype=torch.long,
            device=encoding.memory.device,
        )
        return ObservationEncoding(
            memory=encoding.memory.index_select(0, index),
            memory_padding_mask=encoding.memory_padding_mask.index_select(
                0, index
            ),
            observation_context=encoding.observation_context.index_select(
                0, index
            ),
        )

    def value_estimates(self, encoding: ObservationEncoding) -> Tensor:
        """Return value estimates from an encoded observation batch."""
        return self._value_head(encoding.observation_context).squeeze(
            -1
        )

    def score_argument_prefixes(
        self,
        encoding: ObservationEncoding,
        prefix: ArgumentPrefixTensorBatch,
    ) -> ArgumentPrefixScores:
        """Return next semantic-argument logits for encoded prefixes."""
        prefix_context = self._decode_argument_prefix(
            prefix,
            memory=encoding.memory,
            memory_padding_mask=encoding.memory_padding_mask,
        )
        decision_context = torch.tanh(
            self._decision_projection(
                torch.cat(
                    (encoding.observation_context, prefix_context),
                    dim=-1,
                )
            )
        )
        return ArgumentPrefixScores(
            argument_logits=self._argument_head(decision_context),
        )

    def _encode_observation(
        self,
        observation: ObservationTensorBatch,
        *,
        memory_padding_mask: Tensor,
    ) -> Tensor:
        obs_embedded = self._embed_observation(observation)
        return self._observation_encoder(
            obs_embedded,
            src_key_padding_mask=memory_padding_mask,
        )

    def _embed_observation(
        self,
        observation: ObservationTensorBatch,
    ) -> Tensor:
        components = observation_component_tensors(observation)
        categorical_input = torch.cat(
            (
                self._token_type_embedding(components.token_type_ids),
                self._segment_embedding(components.segment_ids),
                self._field_embedding(components.field_ids),
                self._value_embedding(components.value_ids),
                self._suit_embedding(components.suit_ids),
                self._rank_embedding(components.rank_ids),
                self._points_embedding(components.points_ids),
                self._color_embedding(components.color_ids),
                self._role_embedding(components.role_ids),
                self._trick_age_embedding(components.trick_age_ids),
                self._trick_state_embedding(components.trick_state_ids),
                self._play_order_embedding(components.play_order_ids),
                self._count_embedding(components.count_ids),
                self._play_width_embedding(components.play_width_ids),
                self._event_age_embedding(components.event_age_ids),
            ),
            dim=-1,
        )
        numeric_values = observation.numeric_values * (
            observation.numeric_masks
        )
        numeric_input = torch.cat(
            (numeric_values, observation.numeric_masks),
            dim=-1,
        )
        return self._categorical_projection(
            categorical_input
        ) + self._numeric_projection(numeric_input)

    def _decode_argument_prefix(
        self,
        prefix: ArgumentPrefixTensorBatch,
        *,
        memory: Tensor,
        memory_padding_mask: Tensor,
    ) -> Tensor:
        positions = torch.arange(
            prefix.argument_ids.shape[1],
            dtype=torch.long,
            device=prefix.argument_ids.device,
        ).unsqueeze(0)
        embedded = self._argument_embedding(
            prefix.argument_ids
        ) + self._argument_position_embedding(positions)
        decoded = self._argument_decoder(
            embedded,
            memory,
            tgt_key_padding_mask=~prefix.argument_masks,
            memory_key_padding_mask=memory_padding_mask,
        )
        lengths = prefix.argument_masks.sum(dim=1).clamp_min(1)
        gather_index = (
            (lengths - 1)
            .view(-1, 1, 1)
            .expand(-1, 1, decoded.shape[-1])
        )
        return decoded.gather(dim=1, index=gather_index).squeeze(1)


def _embedding(vocab_size: int, d_model: int) -> nn.Embedding:
    return nn.Embedding(
        vocab_size, d_model, padding_idx=VOCAB_SCHEMA.obs_pad_id
    )


def _masked_mean(values: Tensor, padding_mask: Tensor) -> Tensor:
    keep_mask = (~padding_mask).unsqueeze(-1).to(dtype=values.dtype)
    summed = (values * keep_mask).sum(dim=-2)
    counts = keep_mask.sum(dim=-2).clamp_min(1.0)
    return summed / counts


def _query_or_all_mean(
    values: Tensor,
    *,
    padding_mask: Tensor,
    query_mask: Tensor,
) -> Tensor:
    query_keep = query_mask.unsqueeze(-1).to(dtype=values.dtype)
    query_summed = (values * query_keep).sum(dim=-2)
    query_counts = query_keep.sum(dim=-2)
    query_mean = query_summed / query_counts.clamp_min(1.0)
    all_mean = _masked_mean(values, padding_mask)
    return torch.where(query_counts.gt(0), query_mean, all_mean)
