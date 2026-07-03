"""Torch Transformer policy/value model for Tractor self-play."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.semantic_actions import (
    ARGUMENT_VOCAB_SIZE,
    MAX_ARGUMENT_TOKENS,
)
from server.training.tensorize import (
    ArgumentPrefixTensorBatch,
    ObservationTensorBatch,
)
from server.training.vocab import (
    COLOR_VOCAB_SIZE,
    COUNT_VOCAB_SIZE,
    EVENT_AGE_VOCAB_SIZE,
    FIELD_VOCAB_SIZE,
    OBS_PAD_ID,
    PLAY_ORDER_VOCAB_SIZE,
    PLAY_WIDTH_VOCAB_SIZE,
    POINTS_VOCAB_SIZE,
    RANK_VOCAB_SIZE,
    ROLE_VOCAB_SIZE,
    SEGMENT_ACTION_QUERY_ID,
    SEGMENT_VOCAB_SIZE,
    SUIT_VOCAB_SIZE,
    TOKEN_TYPE_VOCAB_SIZE,
    TRICK_AGE_VOCAB_SIZE,
    TRICK_STATE_VOCAB_SIZE,
    VALUE_VOCAB_SIZE,
)

OBSERVATION_COMPONENT_COUNT: int = 15


@dataclass(frozen=True, slots=True)
class ArgumentHeadOutput:
    """Next-argument logits plus the shared value estimate."""

    argument_logits: Tensor
    values: Tensor


class TractorPolicyModel(nn.Module):
    """Shared observation encoder with semantic argument decoder."""

    def __init__(
        self,
        *,
        d_model: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self._token_type_embedding = _embedding(
            TOKEN_TYPE_VOCAB_SIZE, d_model
        )
        self._segment_embedding = _embedding(
            SEGMENT_VOCAB_SIZE, d_model
        )
        self._field_embedding = _embedding(FIELD_VOCAB_SIZE, d_model)
        self._value_embedding = _embedding(VALUE_VOCAB_SIZE, d_model)
        self._suit_embedding = _embedding(SUIT_VOCAB_SIZE, d_model)
        self._rank_embedding = _embedding(RANK_VOCAB_SIZE, d_model)
        self._points_embedding = _embedding(POINTS_VOCAB_SIZE, d_model)
        self._color_embedding = _embedding(COLOR_VOCAB_SIZE, d_model)
        self._role_embedding = _embedding(ROLE_VOCAB_SIZE, d_model)
        self._trick_age_embedding = _embedding(
            TRICK_AGE_VOCAB_SIZE, d_model
        )
        self._trick_state_embedding = _embedding(
            TRICK_STATE_VOCAB_SIZE, d_model
        )
        self._play_order_embedding = _embedding(
            PLAY_ORDER_VOCAB_SIZE, d_model
        )
        self._count_embedding = _embedding(COUNT_VOCAB_SIZE, d_model)
        self._play_width_embedding = _embedding(
            PLAY_WIDTH_VOCAB_SIZE, d_model
        )
        self._event_age_embedding = _embedding(
            EVENT_AGE_VOCAB_SIZE, d_model
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
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self._observation_encoder = nn.TransformerEncoder(
            observation_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self._argument_embedding = _embedding(
            ARGUMENT_VOCAB_SIZE, d_model
        )
        self._argument_position_embedding = nn.Embedding(
            MAX_ARGUMENT_TOKENS, d_model
        )
        argument_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self._argument_decoder = nn.TransformerDecoder(
            argument_layer,
            num_layers=1,
        )
        self._decision_projection = nn.Linear(d_model * 2, d_model)
        self._argument_head = nn.Linear(d_model, ARGUMENT_VOCAB_SIZE)
        self._value_head = nn.Linear(d_model, 1)

    def forward_argument(
        self,
        observation: ObservationTensorBatch,
        prefix: ArgumentPrefixTensorBatch,
    ) -> ArgumentHeadOutput:
        """Return next semantic-argument logits for a prefix."""
        encoded = self._encode_observation(observation)
        obs_padding = observation.token_type_ids.eq(OBS_PAD_ID)
        query_mask = observation.segment_ids.eq(
            SEGMENT_ACTION_QUERY_ID
        ) & (~obs_padding)
        obs_context = _query_or_all_mean(
            encoded,
            padding_mask=obs_padding,
            query_mask=query_mask,
        )
        prefix_context = self._decode_argument_prefix(
            prefix,
            memory=encoded,
            memory_padding_mask=obs_padding,
        )
        decision_context = torch.tanh(
            self._decision_projection(
                torch.cat((obs_context, prefix_context), dim=-1)
            )
        )
        return ArgumentHeadOutput(
            argument_logits=self._argument_head(decision_context),
            values=self._value_head(obs_context).squeeze(-1),
        )

    def _encode_observation(
        self,
        observation: ObservationTensorBatch,
    ) -> Tensor:
        obs_padding = observation.token_type_ids.eq(OBS_PAD_ID)
        obs_embedded = self._embed_observation(observation)
        return self._observation_encoder(
            obs_embedded,
            src_key_padding_mask=obs_padding,
        )

    def _embed_observation(
        self,
        observation: ObservationTensorBatch,
    ) -> Tensor:
        categorical_input = torch.cat(
            (
                self._token_type_embedding(observation.token_type_ids),
                self._segment_embedding(observation.segment_ids),
                self._field_embedding(observation.field_ids),
                self._value_embedding(observation.value_ids),
                self._suit_embedding(observation.suit_ids),
                self._rank_embedding(observation.rank_ids),
                self._points_embedding(observation.points_ids),
                self._color_embedding(observation.color_ids),
                self._role_embedding(observation.role_ids),
                self._trick_age_embedding(observation.trick_age_ids),
                self._trick_state_embedding(
                    observation.trick_state_ids
                ),
                self._play_order_embedding(observation.play_order_ids),
                self._count_embedding(observation.count_ids),
                self._play_width_embedding(observation.play_width_ids),
                self._event_age_embedding(observation.event_age_ids),
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
    return nn.Embedding(vocab_size, d_model, padding_idx=OBS_PAD_ID)


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
