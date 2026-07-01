"""Torch Transformer policy/value model for Tractor self-play."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.selection_actions import MAX_HAND_CARD_SLOTS
from server.training.tensorize import (
    SELECTION_FEATURE_COUNT,
    ObservationTensorBatch,
    SelectionStateTensorBatch,
)
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


@dataclass(frozen=True, slots=True)
class SelectionHeadOutput:
    """One decision head's logits plus the shared value estimate."""

    card_logits: Tensor
    pass_logits: Tensor | None
    stop_logits: Tensor | None
    values: Tensor


class TractorPolicyModel(nn.Module):
    """Shared observation encoder with selection heads."""

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
        self._selection_projection = nn.Linear(
            MAX_HAND_CARD_SLOTS + SELECTION_FEATURE_COUNT,
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
        self._bid_card_head = nn.Linear(d_model * 2, 1)
        self._stir_card_head = nn.Linear(d_model * 2, 1)
        self._discard_card_head = nn.Linear(d_model * 2, 1)
        self._lead_play_card_head = nn.Linear(d_model * 2, 1)
        self._follow_play_card_head = nn.Linear(d_model * 2, 1)
        self._bid_pass_head = nn.Linear(d_model, 1)
        self._stir_pass_head = nn.Linear(d_model, 1)
        self._bid_stop_head = nn.Linear(d_model, 1)
        self._stir_stop_head = nn.Linear(d_model, 1)
        self._lead_play_stop_head = nn.Linear(d_model, 1)
        self._value_head = nn.Linear(d_model, 1)

    def forward_bid(
        self,
        observation: ObservationTensorBatch,
        selection: SelectionStateTensorBatch,
    ) -> SelectionHeadOutput:
        """Return logits for a bid selection step."""
        return self._forward_head(
            observation,
            selection,
            card_head=self._bid_card_head,
            pass_head=self._bid_pass_head,
            stop_head=self._bid_stop_head,
        )

    def forward_stir(
        self,
        observation: ObservationTensorBatch,
        selection: SelectionStateTensorBatch,
    ) -> SelectionHeadOutput:
        """Return logits for a stir selection step."""
        return self._forward_head(
            observation,
            selection,
            card_head=self._stir_card_head,
            pass_head=self._stir_pass_head,
            stop_head=self._stir_stop_head,
        )

    def forward_discard(
        self,
        observation: ObservationTensorBatch,
        selection: SelectionStateTensorBatch,
    ) -> SelectionHeadOutput:
        """Return logits for a bottom-discard selection step."""
        return self._forward_head(
            observation,
            selection,
            card_head=self._discard_card_head,
            pass_head=None,
            stop_head=None,
        )

    def forward_lead_play(
        self,
        observation: ObservationTensorBatch,
        selection: SelectionStateTensorBatch,
    ) -> SelectionHeadOutput:
        """Return logits for a lead-play selection step."""
        return self._forward_head(
            observation,
            selection,
            card_head=self._lead_play_card_head,
            pass_head=None,
            stop_head=self._lead_play_stop_head,
        )

    def forward_follow_play(
        self,
        observation: ObservationTensorBatch,
        selection: SelectionStateTensorBatch,
    ) -> SelectionHeadOutput:
        """Return logits for a follow-play selection step."""
        return self._forward_head(
            observation,
            selection,
            card_head=self._follow_play_card_head,
            pass_head=None,
            stop_head=None,
        )

    def _forward_head(
        self,
        observation: ObservationTensorBatch,
        selection: SelectionStateTensorBatch,
        *,
        card_head: nn.Linear,
        pass_head: nn.Linear | None,
        stop_head: nn.Linear | None,
    ) -> SelectionHeadOutput:
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
        selection_context = self._embed_selection(selection)
        decision_context = obs_context + selection_context
        hand_contexts = _hand_contexts(encoded, observation)
        expanded_decision = decision_context.unsqueeze(1).expand(
            -1, MAX_HAND_CARD_SLOTS, -1
        )
        card_logits = card_head(
            torch.cat((hand_contexts, expanded_decision), dim=-1)
        ).squeeze(-1)
        return SelectionHeadOutput(
            card_logits=card_logits,
            pass_logits=None
            if pass_head is None
            else pass_head(decision_context).squeeze(-1),
            stop_logits=None
            if stop_head is None
            else stop_head(decision_context).squeeze(-1),
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

    def _embed_selection(
        self,
        selection: SelectionStateTensorBatch,
    ) -> Tensor:
        selection_input = torch.cat(
            (
                selection.selected_slot_masks,
                selection.feature_values,
            ),
            dim=-1,
        )
        return self._selection_projection(selection_input)


def _embedding(vocab_size: int, d_model: int) -> nn.Embedding:
    return nn.Embedding(vocab_size, d_model, padding_idx=OBS_PAD_ID)


def _hand_contexts(
    encoded_observation: Tensor,
    observation: ObservationTensorBatch,
) -> Tensor:
    gather_index = observation.hand_token_indices.unsqueeze(-1).expand(
        -1, -1, encoded_observation.shape[-1]
    )
    return encoded_observation.gather(dim=1, index=gather_index)


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
