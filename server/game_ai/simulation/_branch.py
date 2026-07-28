"""Independent immutable game branches owned by AI reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field

from server.foundation.result import Ok, Rejected
from server.game import (
    CommandRejected,
    GameState,
    Seat,
    apply,
    commands,
    observe,
)
from server.game.snapshots import PlayerSnapshot
from server.policy_model.actions import (
    LegalActionSpace,
    build_legal_action_space,
)
from server.policy_model.observation import (
    Observation,
    ObservationMemory,
    build_observation,
)

type ModelInput = tuple[Observation, LegalActionSpace]


def _snapshot_cache() -> dict[Seat, PlayerSnapshot]:
    return {}


def _model_input_cache() -> dict[Seat, ModelInput]:
    return {}


@dataclass(frozen=True, slots=True)
class SimulationBranch:
    """Engine state with shared history and lazy player projections."""

    state: GameState
    seq: int
    _memory: ObservationMemory
    _snapshots: dict[Seat, PlayerSnapshot] = field(
        default_factory=_snapshot_cache,
        compare=False,
        repr=False,
    )
    _model_inputs: dict[Seat, ModelInput] = field(
        default_factory=_model_input_cache,
        compare=False,
        repr=False,
    )

    @classmethod
    def start(
        cls,
        state: GameState,
    ) -> Ok[SimulationBranch] | Rejected:
        """Initialize the public model history at a round start."""
        memory = ObservationMemory()
        observed = memory.observe(
            seq=0,
            snapshot=observe(state, Seat.A),
        )
        if isinstance(observed, Rejected):
            return observed
        return Ok(cls(state=state, seq=0, _memory=memory))

    def advance(
        self,
        *,
        actor: Seat,
        command: commands.Command,
    ) -> Ok[SimulationBranch] | Rejected:
        """Apply one command without mutating this branch."""
        applied = apply(self.state, actor, command)
        if isinstance(applied, CommandRejected):
            return Rejected(reason=applied.reason)
        next_seq = self.seq + 1
        next_memory = self._memory.fork()
        remembered = next_memory.observe(
            seq=next_seq,
            snapshot=observe(applied.value, Seat.A),
        )
        if isinstance(remembered, Rejected):
            return remembered
        return Ok(
            SimulationBranch(
                state=applied.value,
                seq=next_seq,
                _memory=next_memory,
            )
        )

    def snapshot(self, seat: Seat) -> PlayerSnapshot:
        """Return one lazy player projection in this hidden world."""
        cached = self._snapshots.get(seat)
        if cached is not None:
            return cached
        snapshot = observe(self.state, seat)
        self._snapshots[seat] = snapshot
        return snapshot

    def model_input(self, seat: Seat) -> ModelInput:
        """Build the exact training-format input for one actor."""
        cached = self._model_inputs.get(seat)
        if cached is not None:
            return cached
        snapshot = self.snapshot(seat)
        observation = build_observation(
            viewer=seat,
            snapshot=snapshot,
            memory=self._memory.view(),
        )
        result = (
            observation,
            build_legal_action_space(
                viewer=seat,
                snapshot=snapshot,
                query=observation.action_query,
            ),
        )
        self._model_inputs[seat] = result
        return result


__all__ = ("SimulationBranch",)
