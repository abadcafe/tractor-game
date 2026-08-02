"""Complete-action value evaluation with an independent observation."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from server.policy_model._schema.actions import (
    ACTION_CHOICE_COUNT,
    CARD_CHOICE_BASE_ID,
    MAX_ACTION_STEPS,
)

from .observation_encoder import (
    ActionDecoderInputs,
    EncodedObservation,
)


class CompleteActionEncoder(nn.Module):
    """Encode a complete action against separate observation memory."""

    def __init__(
        self,
        *,
        d_model: int,
        heads: int,
        layers: int,
    ) -> None:
        super().__init__()
        assert d_model > 0
        assert heads > 0
        assert layers > 0
        assert d_model % heads == 0
        self._control_choice_embeddings = nn.Parameter(
            torch.empty(CARD_CHOICE_BASE_ID, d_model)
        )
        nn.init.normal_(
            self._control_choice_embeddings,
            mean=0.0,
            std=d_model**-0.5,
        )
        self._selected_choice_adapter = nn.Linear(d_model, d_model)
        self._action_position_embedding = nn.Embedding(
            MAX_ACTION_STEPS + 1,
            d_model,
        )
        self._value_query = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(
            self._value_query,
            mean=0.0,
            std=d_model**-0.5,
        )
        self._layers = nn.ModuleList(
            _CompleteActionBlock(d_model=d_model, heads=heads)
            for _ in range(layers)
        )

    def forward(
        self,
        encoding: EncodedObservation,
        *,
        source_rows: Tensor,
        choice_ids_padded: Tensor,
        step_counts: Tensor,
    ) -> Tensor:
        """Return one observation-action feature row per trace."""
        row_count, max_steps = choice_ids_padded.shape
        assert row_count > 0
        assert 0 < max_steps <= MAX_ACTION_STEPS
        assert source_rows.shape == (row_count,)
        assert source_rows.dtype == torch.long
        assert source_rows.device == encoding.device
        assert step_counts.shape == (row_count,)
        assert step_counts.dtype == torch.long

        inputs = encoding.action_decoder_inputs()
        selected_inputs = _select_inputs(
            inputs, source_rows=source_rows
        )
        choice_embeddings = self._choice_embeddings(
            inputs=selected_inputs,
            row_count=row_count,
        )
        batch_indices = torch.arange(
            row_count,
            dtype=torch.long,
            device=choice_ids_padded.device,
        ).unsqueeze(1)
        selected = choice_embeddings[batch_indices, choice_ids_padded]
        positions = torch.arange(
            max_steps,
            dtype=torch.long,
            device=choice_ids_padded.device,
        )
        selected = self._selected_choice_adapter(selected)
        selected = selected + self._action_position_embedding(
            positions.unsqueeze(0)
        )
        action_padding_mask = positions.unsqueeze(
            0
        ) >= step_counts.unsqueeze(1)
        selected = selected.masked_fill(
            action_padding_mask.unsqueeze(-1),
            0.0,
        )
        value_query = self._value_query.unsqueeze(0).expand(
            row_count, -1
        )
        value_query = value_query + self._action_position_embedding(
            step_counts
        )
        sequence = torch.cat(
            (selected, value_query.unsqueeze(1)),
            dim=1,
        )
        sequence_padding_mask = torch.cat(
            (
                action_padding_mask,
                torch.zeros(
                    (row_count, 1),
                    dtype=torch.bool,
                    device=action_padding_mask.device,
                ),
            ),
            dim=1,
        )
        for layer in self._layers:
            sequence = layer(
                sequence,
                memory=selected_inputs.memory,
                sequence_padding_mask=sequence_padding_mask,
                memory_padding_mask=(
                    selected_inputs.memory_padding_mask
                ),
            )
        action_context = sequence[:, -1]
        return torch.cat(
            (
                selected_inputs.observation_context,
                action_context,
            ),
            dim=-1,
        )

    def _choice_embeddings(
        self,
        *,
        inputs: ActionDecoderInputs,
        row_count: int,
    ) -> Tensor:
        controls = self._control_choice_embeddings.unsqueeze(0).expand(
            row_count,
            -1,
            -1,
        )
        result = torch.cat(
            (controls, inputs.card_choice_embeddings),
            dim=1,
        )
        assert int(result.shape[1]) == ACTION_CHOICE_COUNT
        return result


class _CompleteActionBlock(nn.Module):
    """Contextualize a complete action against observation memory."""

    def __init__(self, *, d_model: int, heads: int) -> None:
        super().__init__()
        self._self_attention = nn.MultiheadAttention(
            d_model,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self._cross_attention = nn.MultiheadAttention(
            d_model,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self._norm1 = nn.LayerNorm(d_model)
        self._norm2 = nn.LayerNorm(d_model)
        self._norm3 = nn.LayerNorm(d_model)
        self._feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        sequence: Tensor,
        *,
        memory: Tensor,
        sequence_padding_mask: Tensor,
        memory_padding_mask: Tensor,
    ) -> Tensor:
        """Contextualize action states with visible information."""
        self_attended = _stable_training_attention(
            attention=self._self_attention,
            query=sequence,
            key=sequence,
            value=sequence,
            key_padding_mask=sequence_padding_mask,
        )
        hidden = self._norm1(sequence + self_attended)
        cross_attended = _stable_training_attention(
            attention=self._cross_attention,
            query=hidden,
            key=memory,
            value=memory,
            key_padding_mask=memory_padding_mask,
        )
        hidden = self._norm2(hidden + cross_attended)
        return self._norm3(hidden + self._feed_forward(hidden))


def _stable_training_attention(
    *,
    attention: nn.MultiheadAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_padding_mask: Tensor,
) -> Tensor:
    if not torch.is_grad_enabled():
        attended, _weights = attention(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return attended
    with sdpa_kernel(SDPBackend.MATH):
        attended, _weights = attention(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
    return attended


def _select_inputs(
    inputs: ActionDecoderInputs,
    *,
    source_rows: Tensor,
) -> ActionDecoderInputs:
    return ActionDecoderInputs(
        memory=inputs.memory.index_select(0, source_rows),
        memory_padding_mask=inputs.memory_padding_mask.index_select(
            0, source_rows
        ),
        observation_context=inputs.observation_context.index_select(
            0, source_rows
        ),
        card_choice_embeddings=(
            inputs.card_choice_embeddings.index_select(0, source_rows)
        ),
    )


__all__ = ("CompleteActionEncoder",)
