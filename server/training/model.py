"""Torch Transformer policy/value model for action-token training."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from server.training.action_tokens import (
    ACTION_TOKEN_VOCAB_SIZE,
    MAX_ACTION_TOKENS,
    PAD_TOKEN_ID,
)
from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.tensorize import ObservationTensorBatch
from server.training.vocab import (
    CARD_ORDER_VOCAB_SIZE,
    COLOR_VOCAB_SIZE,
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


class UpgradePolicyModel(nn.Module):
    """Transformer encoder with autoregressive action-token head."""

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
        self._card_order_embedding = _embedding(
            CARD_ORDER_VOCAB_SIZE, d_model
        )
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
        self._action_embedding = nn.Embedding(
            ACTION_TOKEN_VOCAB_SIZE,
            d_model,
            padding_idx=PAD_TOKEN_ID,
        )
        self._action_position_embedding = nn.Embedding(
            MAX_ACTION_TOKENS,
            d_model,
        )
        action_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self._action_encoder = nn.TransformerEncoder(
            action_layer,
            num_layers=max(1, layers // 2),
            enable_nested_tensor=False,
        )
        self._policy_head = nn.Linear(
            d_model * 2, ACTION_TOKEN_VOCAB_SIZE
        )
        self._value_head = nn.Linear(d_model, 1)

    def forward_action(
        self,
        observation: ObservationTensorBatch,
        action_prefix_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Return next-token logits and state values.

        ``observation`` fields shape: [batch, obs_tokens]
        ``action_prefix_ids`` shape: [batch, action_prefix_tokens]
        """
        obs_padding = observation.token_type_ids.eq(OBS_PAD_ID)
        obs_embedded = self._embed_observation(observation)
        encoded_observation = self._observation_encoder(
            obs_embedded,
            src_key_padding_mask=obs_padding,
        )
        query_mask = observation.segment_ids.eq(
            SEGMENT_ACTION_QUERY_ID
        ) & (~obs_padding)
        obs_context = _query_or_all_mean(
            encoded_observation,
            padding_mask=obs_padding,
            query_mask=query_mask,
        )

        action_padding = action_prefix_ids.eq(PAD_TOKEN_ID)
        action_embedded = self._embed_action(action_prefix_ids)
        encoded_action = self._action_encoder(
            action_embedded,
            mask=_causal_mask(
                action_prefix_ids.shape[1], action_ids=action_prefix_ids
            ),
            src_key_padding_mask=action_padding,
        )
        action_context = _last_non_padding(
            encoded_action, action_padding
        )
        combined = torch.cat((obs_context, action_context), dim=-1)
        logits = self._policy_head(combined)
        values = self._value_head(obs_context).squeeze(-1)
        return logits, values

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
                self._card_order_embedding(observation.card_order_ids),
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

    def _embed_action(self, action_prefix_ids: Tensor) -> Tensor:
        positions = torch.arange(
            action_prefix_ids.shape[1],
            dtype=torch.long,
            device=action_prefix_ids.device,
        ).unsqueeze(0)
        return self._action_embedding(
            action_prefix_ids
        ) + self._action_position_embedding(positions)


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


def _last_non_padding(values: Tensor, padding_mask: Tensor) -> Tensor:
    lengths = (~padding_mask).sum(dim=1).clamp_min(1)
    batch_indices = torch.arange(values.shape[0], device=values.device)
    return values[batch_indices, lengths - 1]


def _causal_mask(size: int, *, action_ids: Tensor) -> Tensor:
    return torch.triu(
        torch.ones(
            (size, size),
            dtype=torch.bool,
            device=action_ids.device,
        ),
        diagonal=1,
    )
