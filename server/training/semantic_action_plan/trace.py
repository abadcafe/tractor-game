"""Semantic trace token conversion."""

from __future__ import annotations

from server.result import Ok, Rejected
from server.training.semantic_actions.arguments import (
    SemanticArgument,
    SemanticArgumentTrace,
)
from server.training.semantic_actions.codec import (
    semantic_argument_from_id,
    semantic_argument_id,
)


def semantic_trace_token_ids(
    trace: SemanticArgumentTrace,
) -> tuple[int, ...]:
    """Return model token ids for one complete semantic trace."""
    return tuple(
        semantic_argument_id(argument) for argument in trace.arguments
    )


def semantic_trace_from_token_ids(
    token_ids: tuple[int, ...],
) -> Ok[SemanticArgumentTrace] | Rejected:
    """Decode model token ids into a semantic trace."""
    arguments: list[SemanticArgument] = []
    for token_id in token_ids:
        argument_result = semantic_argument_from_id(token_id)
        if isinstance(argument_result, Rejected):
            return argument_result
        arguments.append(argument_result.value)
    return Ok(value=SemanticArgumentTrace(arguments=tuple(arguments)))
