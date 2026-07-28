"""Model-facing observations built from explicit observation memory."""

from __future__ import annotations

from dataclasses import dataclass

from server.foundation.result import Ok
from server.game import Seat
from server.game.rules.cards.faces import FaceCount
from server.game.snapshots import PlayerSnapshot
from server.policy_model.observation.history import (
    ObservationMemoryView,
)
from server.policy_model.observation.query import (
    ActionQuery,
    build_action_query,
)
from server.policy_model.observation.relative_state import (
    project_relative_observation,
)
from server.policy_model.observation.relative_state.contexts import (
    RoundContext,
)
from server.policy_model.observation.tokenization import (
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
    viewer: Seat,
    snapshot: PlayerSnapshot,
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
            viewer=viewer,
            snapshot=snapshot,
        ),
    )


__all__ = ("Observation", "build_observation")
