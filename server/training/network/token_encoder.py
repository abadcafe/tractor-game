"""Typed semantic token encoding with one shared card encoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from server.training.tokenization.encoding_schema import (
    ACTION_KIND_COUNT,
    ACTION_KIND_INDEX,
    ACTOR_INDEX,
    DISPOSITION_COUNT,
    DISPOSITION_INDEX,
    EFFECTIVE_SUIT_COUNT,
    EFFECTIVE_SUIT_INDEX,
    FAMILY_INDEX,
    PAYLOAD_ROLE_COUNT,
    PAYLOAD_ROLE_INDEX,
    RANK_COUNT,
    RANK_INDEX,
    SEMANTIC_STATE_COUNT,
    STATE_INDEX,
    SUIT_COUNT,
    SUIT_INDEX,
    TOKEN_VARIANT_COUNT,
    TRICK_POSITION_COUNT,
    TRICK_POSITION_INDEX,
    VARIANT_INDEX,
)

TOKEN_FAMILY_COUNT: int = 6


class TypedTokenEncoder(nn.Module):
    """Encode five closed families without learned NONE ids."""

    def __init__(self, *, d_model: int) -> None:
        super().__init__()
        self._d_model = d_model
        self._family = _embedding(TOKEN_FAMILY_COUNT, d_model)
        self._variant = _embedding(TOKEN_VARIANT_COUNT, d_model)
        self._rank = _embedding(RANK_COUNT, d_model)
        self._suit = _embedding(SUIT_COUNT, d_model)
        self._effective_suit = _embedding(EFFECTIVE_SUIT_COUNT, d_model)
        self._action_kind = _embedding(ACTION_KIND_COUNT, d_model)
        self._state = _embedding(SEMANTIC_STATE_COUNT, d_model)
        self._disposition = _embedding(DISPOSITION_COUNT, d_model)
        self._trick_position = _embedding(TRICK_POSITION_COUNT, d_model)
        self._payload_role = _embedding(PAYLOAD_ROLE_COUNT, d_model)
        self._actor_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self._face_mlp = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self._card_mlp = nn.Sequential(
            nn.Linear(d_model * 2 + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self._scalar_projection = nn.Linear(1, d_model, bias=False)
        self._family_adapters = nn.ModuleList(
            _family_adapter(d_model) for _ in range(5)
        )
        actor_features = torch.tensor(
            (
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (1.0, -1.0, 0.0),
                (-1.0, 0.0, 1.0),
                (-1.0, 0.0, -1.0),
            ),
            dtype=torch.float32,
        )
        self.register_buffer(
            "_actor_features", actor_features, persistent=False
        )
        self._actor_features: Tensor = actor_features

    def forward(
        self,
        *,
        category_ids: Tensor,
        scalar_values: Tensor,
        card_rule_values: Tensor,
    ) -> Tensor:
        """Encode one padded batch of compact typed token rows."""
        assert category_ids.ndim == 3
        family_ids = category_ids[:, :, FAMILY_INDEX]
        actor_ids = category_ids[:, :, ACTOR_INDEX]
        rank_ids = category_ids[:, :, RANK_INDEX]
        suit_ids = category_ids[:, :, SUIT_INDEX]
        base = (
            self._family(family_ids)
            + self._variant(category_ids[:, :, VARIANT_INDEX])
            + self._action_kind(category_ids[:, :, ACTION_KIND_INDEX])
            + self._state(category_ids[:, :, STATE_INDEX])
            + self._disposition(category_ids[:, :, DISPOSITION_INDEX])
            + self._trick_position(
                category_ids[:, :, TRICK_POSITION_INDEX]
            )
            + self._payload_role(category_ids[:, :, PAYLOAD_ROLE_INDEX])
        )
        base = base + self._actor_vectors(actor_ids)
        card_mask = family_ids.eq(5).unsqueeze(-1)
        non_card = ~card_mask
        base = base + non_card * (
            self._rank(rank_ids) + self._suit(suit_ids)
        )
        scalar = self._scalar_projection(scalar_values.unsqueeze(-1))
        base = base + non_card * scalar
        card_values = self._encode_cards(
            suit_ids=suit_ids,
            rank_ids=rank_ids,
            effective_suit_ids=category_ids[:, :, EFFECTIVE_SUIT_INDEX],
            counts=scalar_values,
            rule_values=card_rule_values,
        )
        base = base + card_mask * card_values
        encoded = torch.zeros_like(base)
        for family_id, adapter in enumerate(
            self._family_adapters, start=1
        ):
            mask = family_ids.eq(family_id).unsqueeze(-1)
            encoded = encoded + mask * adapter(base)
        return encoded

    def encode_card_candidates(
        self,
        *,
        suit_ids: Tensor,
        rank_ids: Tensor,
        effective_suit_ids: Tensor,
        counts: Tensor,
        rule_values: Tensor,
    ) -> Tensor:
        """Encode output candidates with the same card parameters."""
        return self._encode_cards(
            suit_ids=suit_ids,
            rank_ids=rank_ids,
            effective_suit_ids=effective_suit_ids,
            counts=counts,
            rule_values=rule_values,
        )

    def _actor_vectors(self, actor_ids: Tensor) -> Tensor:
        features = self._actor_features[actor_ids]
        vectors = self._actor_mlp(features)
        return vectors * actor_ids.ne(0).unsqueeze(-1)

    def _encode_cards(
        self,
        *,
        suit_ids: Tensor,
        rank_ids: Tensor,
        effective_suit_ids: Tensor,
        counts: Tensor,
        rule_values: Tensor,
    ) -> Tensor:
        point_values = rule_values[..., 0:1]
        strength_values = rule_values[..., 1:2]
        face = self._face_mlp(
            torch.cat(
                (
                    self._suit(suit_ids),
                    self._rank(rank_ids),
                    point_values,
                ),
                dim=-1,
            )
        )
        return self._card_mlp(
            torch.cat(
                (
                    face,
                    counts.unsqueeze(-1) / 2.0,
                    self._effective_suit(effective_suit_ids),
                    strength_values,
                ),
                dim=-1,
            )
        )


def _embedding(size: int, d_model: int) -> nn.Embedding:
    return nn.Embedding(size, d_model, padding_idx=0)


def _family_adapter(d_model: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model),
        nn.LayerNorm(d_model),
    )


__all__ = ("TypedTokenEncoder",)
