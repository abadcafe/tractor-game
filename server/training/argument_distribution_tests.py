"""Tests for semantic argument distributions."""

from __future__ import annotations

import torch

from server.result import Ok, Rejected
from server.rules.card_faces import CardFace, FaceCount
from server.rules.cards import Rank, Suit
from server.training.argument_distribution import (
    argument_distribution,
    argument_logits_for_choices,
)
from server.training.semantic_actions import SemanticArgument
from server.training.semantic_actions.codec import (
    SEMANTIC_CODEC,
    semantic_argument_id,
)


def test_argument_distribution_scores_only_legal_choices() -> None:
    pass_argument = SemanticArgument("pass")
    stop_argument = SemanticArgument("stop")
    select_argument = SemanticArgument(
        "select_face_count",
        FaceCount(CardFace(Suit.HEARTS, Rank.ACE), 2),
    )
    logits = torch.full((SEMANTIC_CODEC.argument_vocab_size,), -1000.0)
    logits[semantic_argument_id(pass_argument)] = 1.0
    logits[semantic_argument_id(stop_argument)] = 3.0
    logits[semantic_argument_id(select_argument)] = 100.0

    distribution_result = argument_distribution(
        argument_logits=logits,
        choices=(pass_argument, stop_argument),
    )
    assert isinstance(distribution_result, Ok)
    distribution = distribution_result.value

    assert distribution.logits.shape == (2,)
    assert torch.equal(distribution.logits, torch.tensor([1.0, 3.0]))
    assert torch.allclose(
        distribution.probabilities,
        torch.softmax(torch.tensor([1.0, 3.0]), dim=0),
    )
    assert torch.allclose(
        distribution.log_probabilities,
        torch.log_softmax(torch.tensor([1.0, 3.0]), dim=0),
    )


def test_argument_distribution_rejects_non_finite_logits() -> None:
    pass_argument = SemanticArgument("pass")
    logits = torch.zeros((SEMANTIC_CODEC.argument_vocab_size,))
    logits[semantic_argument_id(pass_argument)] = torch.inf

    result = argument_distribution(
        argument_logits=logits,
        choices=(pass_argument,),
    )

    assert isinstance(result, Rejected)
    assert "logits must be finite" in result.reason


def test_argument_logits_for_choices_returns_requested_rows() -> None:
    logits = torch.zeros((SEMANTIC_CODEC.argument_vocab_size,))

    result = argument_logits_for_choices(
        argument_logits=logits,
        choices=(SemanticArgument("pass"),),
    )

    assert result.shape == (1,)
