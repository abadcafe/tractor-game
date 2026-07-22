"""Observation encoding and reusable model memory."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from server.training.semantic_actions.choices import CARD_CHOICE_COUNT
from server.training.tensorize import ObservationTensorBatch

from .structured_attention import StructuredObservationEncoder
from .token_encoder import TypedTokenEncoder


@dataclass(frozen=True, slots=True)
class ActionDecoderInputs:
    """Tensor view for the model-internal action decoder."""

    memory: Tensor
    memory_padding_mask: Tensor
    observation_context: Tensor
    card_choice_embeddings: Tensor


@dataclass(frozen=True, slots=True)
class EncodedObservation:
    """Private encoder memory consumed by model-owned heads."""

    _memory: Tensor
    _memory_padding_mask: Tensor
    _observation_context: Tensor
    _card_choice_embeddings: Tensor

    def __post_init__(self) -> None:
        assert self._memory.ndim == 3
        assert self._memory_padding_mask.shape == self._memory.shape[:2]
        assert self._observation_context.shape == (
            self.batch_size,
            int(self._memory.shape[2]),
        )
        assert self._card_choice_embeddings.shape == (
            self.batch_size,
            CARD_CHOICE_COUNT,
            int(self._memory.shape[2]),
        )

    @property
    def batch_size(self) -> int:
        """Number of encoded observations."""
        return int(self._memory.shape[0])

    @property
    def device(self) -> torch.device:
        """Device shared by all tensors in this encoding."""
        return self._memory.device

    def value_context(self) -> Tensor:
        """Return contextual query state for the value head."""
        return self._observation_context

    def action_decoder_inputs(self) -> ActionDecoderInputs:
        """Return the tensor view consumed by action decoding."""
        return ActionDecoderInputs(
            memory=self._memory,
            memory_padding_mask=self._memory_padding_mask,
            observation_context=self._observation_context,
            card_choice_embeddings=self._card_choice_embeddings,
        )


class ObservationEncoder(nn.Module):
    """Own token encoding and structure-aware attention."""

    def __init__(
        self, *, d_model: int, layers: int, heads: int
    ) -> None:
        super().__init__()
        self._token_encoder = TypedTokenEncoder(d_model=d_model)
        self._structured_attention = StructuredObservationEncoder(
            d_model=d_model,
            layers=layers,
            heads=heads,
        )

    def forward(
        self, observation: ObservationTensorBatch
    ) -> EncodedObservation:
        """Encode a padded observation batch and every candidate."""
        memory_padding_mask = observation.category_ids[:, :, 0].eq(0)
        memory = self._structured_attention(
            self._token_encoder(
                category_ids=observation.category_ids,
                scalar_values=observation.scalar_values,
                card_rule_values=observation.card_rule_values,
            ),
            padding_mask=memory_padding_mask,
            encoded_structure_coordinates=(
                observation.encoded_structure_coordinates
            ),
        )
        batch_indices = torch.arange(
            int(memory.shape[0]),
            dtype=torch.long,
            device=memory.device,
        )
        observation_context = memory[
            batch_indices, observation.query_indices
        ]
        card_categories = observation.candidate_category_ids
        card_choice_embeddings = (
            self._token_encoder.encode_card_candidates(
                suit_ids=card_categories[:, :, 0],
                rank_ids=card_categories[:, :, 1],
                effective_suit_ids=card_categories[:, :, 2],
                counts=observation.candidate_counts,
                rule_values=observation.candidate_card_rule_values,
            )
        )
        assert int(card_choice_embeddings.shape[1]) == CARD_CHOICE_COUNT
        return EncodedObservation(
            _memory=memory,
            _memory_padding_mask=memory_padding_mask,
            _observation_context=observation_context,
            _card_choice_embeddings=card_choice_embeddings,
        )


__all__ = ("EncodedObservation", "ObservationEncoder")
