"""Independent immutable game-engine branches owned by AI search."""

from __future__ import annotations

from dataclasses import dataclass

from server.foundation.result import Ok, Rejected
from server.game import (
    CommandRejected,
    GameState,
    Seat,
    SeatMap,
    apply,
    commands,
    observe,
    seats,
)
from server.game.snapshots import PlayerSnapshot
from server.training.legal_actions import (
    LegalActionIndex,
    build_legal_action_index,
)
from server.training.observation import Observation, build_observation
from server.training.observation_memory import ObservationMemory


@dataclass(frozen=True, slots=True)
class EngineBranch:
    """An engine state with branch-local observation histories."""

    state: GameState
    seq: int
    memories: SeatMap[ObservationMemory]

    @classmethod
    def start(
        cls,
        state: GameState,
    ) -> Ok[EngineBranch] | Rejected:
        """Initialize four model histories at the first deal state."""
        memories = SeatMap(
            a=ObservationMemory(),
            b=ObservationMemory(),
            c=ObservationMemory(),
            d=ObservationMemory(),
        )
        for seat in seats():
            observed = memories.at(seat).observe(
                seq=0,
                snapshot=observe(state, seat),
            )
            if isinstance(observed, Rejected):
                return observed
        return Ok(cls(state=state, seq=0, memories=memories))

    def fork(self) -> EngineBranch:
        """Return an independent branch at the same engine position."""
        return EngineBranch(
            state=self.state,
            seq=self.seq,
            memories=self.memories.map(lambda memory: memory.fork()),
        )

    def advance(
        self,
        *,
        actor: Seat,
        command: commands.Command,
    ) -> Ok[EngineBranch] | Rejected:
        """Apply one command and advance every branch-local history."""
        applied = apply(self.state, actor, command)
        if isinstance(applied, CommandRejected):
            return Rejected(reason=applied.reason)
        next_memories = self.memories.map(lambda memory: memory.fork())
        next_seq = self.seq + 1
        for seat in seats():
            observed = next_memories.at(seat).observe(
                seq=next_seq,
                snapshot=observe(applied.value, seat),
            )
            if isinstance(observed, Rejected):
                return observed
        return Ok(
            EngineBranch(
                state=applied.value,
                seq=next_seq,
                memories=next_memories,
            )
        )

    def snapshot(self, seat: Seat) -> PlayerSnapshot:
        """Return one player view in this hypothetical branch."""
        return observe(self.state, seat)

    def model_input(
        self,
        seat: Seat,
    ) -> tuple[Observation, LegalActionIndex]:
        """Build the training observation and legal action index."""
        snapshot = self.snapshot(seat)
        observation = build_observation(
            viewer=seat,
            snapshot=snapshot,
            memory=self.memories.at(seat).view(),
        )
        return (
            observation,
            build_legal_action_index(
                viewer=seat,
                snapshot=snapshot,
                query=observation.action_query,
            ),
        )


__all__ = ("EngineBranch",)
