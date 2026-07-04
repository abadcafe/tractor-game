"""Recorded semantic choices used to replay policy probabilities."""

from __future__ import annotations

from dataclasses import dataclass

from server.training.semantic_actions.arguments import SemanticArgument
from server.training.semantic_actions.codec import semantic_argument_id


@dataclass(frozen=True, slots=True)
class SemanticChoiceStep:
    """Legal semantic arguments and the selected offset for one step."""

    allowed_argument_ids: tuple[int, ...]
    selected_argument_offset: int
    selected_argument_id: int

    def __post_init__(self) -> None:
        assert self.allowed_argument_ids
        assert (
            0
            <= self.selected_argument_offset
            < len(self.allowed_argument_ids)
        )
        assert (
            self.allowed_argument_ids[self.selected_argument_offset]
            == self.selected_argument_id
        )


@dataclass(frozen=True, slots=True)
class SemanticChoiceTrace:
    """Recorded legal-choice stream for one generated action trace."""

    steps: tuple[SemanticChoiceStep, ...]

    def __post_init__(self) -> None:
        assert self.steps


def semantic_choice_step_from_offset(
    *,
    allowed: tuple[SemanticArgument, ...],
    selected_argument_offset: int,
) -> SemanticChoiceStep:
    """Record a selected semantic argument by legal-choice offset."""
    assert allowed
    assert 0 <= selected_argument_offset < len(allowed)
    allowed_argument_ids = tuple(
        semantic_argument_id(argument) for argument in allowed
    )
    return SemanticChoiceStep(
        allowed_argument_ids=allowed_argument_ids,
        selected_argument_offset=selected_argument_offset,
        selected_argument_id=allowed_argument_ids[
            selected_argument_offset
        ],
    )


def semantic_choice_step_from_argument(
    *,
    allowed: tuple[SemanticArgument, ...],
    selected_argument: SemanticArgument,
) -> SemanticChoiceStep:
    """Record a selected semantic argument by value in legal choices."""
    assert selected_argument in allowed
    return semantic_choice_step_from_offset(
        allowed=allowed,
        selected_argument_offset=allowed.index(selected_argument),
    )
