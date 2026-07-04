"""Tests for semantic argument distributions."""

from __future__ import annotations

import torch

from server.result import Ok, Rejected
from server.rules.card_faces import CardFace, FaceCount
from server.rules.cards import Rank, Suit
from server.training.argument_distribution import (
    argument_distribution,
    argument_logits_for_choices,
    batched_argument_distribution,
)
from server.training.choice_trace import (
    semantic_choice_step_from_offset,
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


def test_batched_argument_distribution_matches_single_rows() -> None:
    pass_argument = SemanticArgument("pass")
    stop_argument = SemanticArgument("stop")
    first_select = SemanticArgument(
        "select_face_count",
        FaceCount(CardFace(Suit.HEARTS, Rank.ACE), 1),
    )
    second_select = SemanticArgument(
        "select_face_count",
        FaceCount(CardFace(Suit.SPADES, Rank.KING), 2),
    )
    first_logits = torch.linspace(
        -1.0, 1.0, SEMANTIC_CODEC.argument_vocab_size
    )
    second_logits = torch.linspace(
        1.0, -1.0, SEMANTIC_CODEC.argument_vocab_size
    )
    first_choices = (pass_argument, stop_argument, first_select)
    second_choices = (second_select, stop_argument)
    first_single = argument_distribution(
        argument_logits=first_logits,
        choices=first_choices,
    )
    second_single = argument_distribution(
        argument_logits=second_logits,
        choices=second_choices,
    )
    assert isinstance(first_single, Ok)
    assert isinstance(second_single, Ok)

    result = batched_argument_distribution(
        argument_logits=torch.stack((first_logits, second_logits)),
        choice_steps=(
            semantic_choice_step_from_offset(
                allowed=first_choices,
                selected_argument_offset=1,
            ),
            semantic_choice_step_from_offset(
                allowed=second_choices,
                selected_argument_offset=0,
            ),
        ),
    )

    assert isinstance(result, Ok)
    distribution = result.value
    assert torch.allclose(
        distribution.selected_log_probabilities,
        torch.stack(
            (
                first_single.value.log_probabilities[1],
                second_single.value.log_probabilities[0],
            )
        ),
    )
    assert torch.allclose(
        distribution.entropies,
        torch.stack(
            (first_single.value.entropy, second_single.value.entropy)
        ),
    )


def test_batched_argument_distribution_rejects_non_finite_logits() -> (
    None
):
    pass_argument = SemanticArgument("pass")
    logits = torch.zeros((1, SEMANTIC_CODEC.argument_vocab_size))
    logits[0, semantic_argument_id(pass_argument)] = torch.nan

    result = batched_argument_distribution(
        argument_logits=logits,
        choice_steps=(
            semantic_choice_step_from_offset(
                allowed=(pass_argument,),
                selected_argument_offset=0,
            ),
        ),
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
