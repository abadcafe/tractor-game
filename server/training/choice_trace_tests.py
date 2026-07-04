"""Tests for recorded semantic choice traces."""

from __future__ import annotations

from server.rules.card_faces import CardFace, FaceCount
from server.rules.cards import Rank, Suit
from server.training.choice_trace import (
    SemanticChoiceTrace,
    semantic_choice_step_from_argument,
    semantic_choice_step_from_offset,
)
from server.training.semantic_actions import SemanticArgument
from server.training.semantic_actions.codec import semantic_argument_id


def test_semantic_choice_step_from_offset_records_selected_id() -> None:
    pass_argument = SemanticArgument("pass")
    stop_argument = SemanticArgument("stop")

    step = semantic_choice_step_from_offset(
        allowed=(pass_argument, stop_argument),
        selected_argument_offset=1,
    )

    assert step.allowed_argument_ids == (
        semantic_argument_id(pass_argument),
        semantic_argument_id(stop_argument),
    )
    assert step.selected_argument_offset == 1
    assert step.selected_argument_id == semantic_argument_id(
        stop_argument
    )


def test_semantic_choice_step_from_argument_records_offset() -> None:
    pass_argument = SemanticArgument("pass")
    select_argument = SemanticArgument(
        "select_face_count",
        FaceCount(CardFace(Suit.CLUBS, Rank.THREE), 1),
    )

    step = semantic_choice_step_from_argument(
        allowed=(pass_argument, select_argument),
        selected_argument=select_argument,
    )

    assert step.selected_argument_offset == 1
    assert step.selected_argument_id == semantic_argument_id(
        select_argument
    )
    assert SemanticChoiceTrace(steps=(step,)).steps == (step,)
