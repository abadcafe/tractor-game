"""Tests for the shared typed token and card encoder."""

import torch

from server.training.tokenization.encoding_schema import CATEGORY_COUNT

from .token_encoder import TypedTokenEncoder


def test_card_token_and_candidate_paths_share_semantic_shape() -> None:
    encoder = TypedTokenEncoder(d_model=8)
    category_ids = torch.zeros((1, 1, CATEGORY_COUNT), dtype=torch.long)
    category_ids[:, :, 0] = 5
    category_ids[:, :, 3] = 1
    category_ids[:, :, 4] = 1
    category_ids[:, :, 5] = 1
    counts = torch.ones((1, 1))
    rule_values = torch.tensor((((5.0, 0.25),),))

    encoded_token = encoder(
        category_ids=category_ids,
        scalar_values=counts,
        card_rule_values=rule_values,
    )
    candidate = encoder.encode_card_candidates(
        suit_ids=category_ids[:, :, 4],
        rank_ids=category_ids[:, :, 3],
        effective_suit_ids=category_ids[:, :, 5],
        counts=counts,
        rule_values=rule_values,
    )

    assert encoded_token.shape == candidate.shape == (1, 1, 8)
    assert bool(torch.isfinite(encoded_token).all().item())
    assert bool(torch.isfinite(candidate).all().item())
