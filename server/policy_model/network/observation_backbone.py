"""Observation encoding and reusable model memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import final, override

import torch
from torch import Tensor, nn

from server.policy_model._schema.actions import CARD_CHOICE_COUNT
from server.policy_model.observation.tensor_batch import (
    ObservationTensorBatch,
)

from ._module_call import call_tensor
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

    def action_decoder_inputs(self) -> ActionDecoderInputs:
        """Return the tensor view consumed by model decoders."""
        return ActionDecoderInputs(
            memory=self._memory,
            memory_padding_mask=self._memory_padding_mask,
            observation_context=self._observation_context,
            card_choice_embeddings=self._card_choice_embeddings,
        )

    def observation_context(self) -> Tensor:
        """Return the query-position representation for scalar heads."""
        return self._observation_context


@final
class ObservationBackbone(nn.Module):
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

    @override
    def forward(
        self, observation: ObservationTensorBatch
    ) -> EncodedObservation:
        """Encode a padded observation batch and every candidate."""
        memory_padding_mask = observation.category_ids[:, :, 0].eq(0)
        token_values = call_tensor(
            self._token_encoder,
            category_ids=observation.category_ids,
            scalar_values=observation.scalar_values,
            card_rule_values=observation.card_rule_values,
        )
        memory = call_tensor(
            self._structured_attention,
            token_values,
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
        observation_context: Tensor = memory[
            batch_indices, observation.query_indices
        ]
        card_categories = observation.candidate_category_ids
        candidate_rows = _shared_candidate_rows(observation)
        card_choice_embeddings = (
            self._token_encoder.encode_card_candidates(
                suit_ids=card_categories[candidate_rows, :, 0],
                rank_ids=card_categories[candidate_rows, :, 1],
                effective_suit_ids=card_categories[
                    candidate_rows, :, 2
                ],
                counts=observation.candidate_counts[candidate_rows],
                rule_values=observation.candidate_card_rule_values[
                    candidate_rows
                ],
            )
        )
        if (
            int(card_choice_embeddings.shape[0]) == 1
            and batch_indices.numel() > 1
        ):
            card_choice_embeddings = card_choice_embeddings.expand(
                int(batch_indices.shape[0]), -1, -1
            )
        assert int(card_choice_embeddings.shape[1]) == CARD_CHOICE_COUNT
        return EncodedObservation(
            _memory=memory,
            _memory_padding_mask=memory_padding_mask,
            _observation_context=observation_context,
            _card_choice_embeddings=card_choice_embeddings,
        )


def _shared_candidate_rows(
    observation: ObservationTensorBatch,
) -> slice:
    batch_size = int(observation.category_ids.shape[0])
    if batch_size == 1:
        return slice(None)
    categories = observation.candidate_category_ids
    counts = observation.candidate_counts
    rules = observation.candidate_card_rule_values
    shared = categories.eq(categories[0:1].expand_as(categories)).all()
    shared = shared & counts.eq(counts[0:1].expand_as(counts)).all()
    shared = shared & rules.eq(rules[0:1].expand_as(rules)).all()
    if bool(shared.item()):
        return slice(0, 1)
    return slice(None)


__all__ = ("EncodedObservation", "ObservationBackbone")
