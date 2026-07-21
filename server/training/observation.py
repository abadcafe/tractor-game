"""Model-facing observations built from explicit observation memory."""

from __future__ import annotations

from dataclasses import dataclass

from server.foundation.result import Ok
from server.game.protocol import StateSnapshot
from server.game.rules.card_faces import FaceCount
from server.training.observation_memory import ObservationMemoryView
from server.training.relative_state import project_relative_observation
from server.training.relative_state.contexts import RoundContext
from server.training.semantic_actions.query import (
    ActionQuery,
    build_action_query,
)
from server.training.tokenization import (
    TokenNode,
    TokenSequence,
    tokenize,
)


@dataclass(frozen=True, slots=True)
class Observation:
    """Typed tokens and action-planning facts for one decision."""

    token_sequence: TokenSequence
    round_context: RoundContext
    hand_faces: tuple[FaceCount, ...]
    action_query: ActionQuery

    @property
    def tokens(self) -> tuple[TokenNode, ...]:
        """Return the flat typed semantic nodes."""
        return self.token_sequence.nodes


def build_observation(
    *,
    viewer: int,
    snapshot: StateSnapshot,
    memory: ObservationMemoryView,
) -> Observation:
    """Build an observation from state and explicit memory."""
    relative = project_relative_observation(
        viewer=viewer,
        snapshot=snapshot,
        memory=memory,
    )
    assert isinstance(relative, Ok)
    state = relative.value
    return Observation(
        token_sequence=tokenize(state),
        round_context=state.round_context,
        hand_faces=state.hand,
        action_query=build_action_query(
            player_index=viewer,
            snapshot=snapshot,
        ),
    )


__all__ = ("Observation", "build_observation")
