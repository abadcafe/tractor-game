"""Action trace choice-id conversion."""

from __future__ import annotations

from server.foundation.result import Ok, Rejected
from server.training.semantic_actions.choices import (
    ActionChoice,
    ActionTrace,
    action_choice_from_id,
    action_choice_id,
)


def action_trace_choice_ids(
    trace: ActionTrace,
) -> tuple[int, ...]:
    """Return dense choice ids for one complete action trace."""
    return tuple(action_choice_id(choice) for choice in trace.choices)


def action_trace_from_choice_ids(
    choice_ids: tuple[int, ...],
) -> Ok[ActionTrace] | Rejected:
    """Decode dense choice ids into an action trace."""
    choices: list[ActionChoice] = []
    for choice_id in choice_ids:
        choice_result = action_choice_from_id(choice_id)
        if isinstance(choice_result, Rejected):
            return choice_result
        choices.append(choice_result.value)
    return Ok(value=ActionTrace(choices=tuple(choices)))
