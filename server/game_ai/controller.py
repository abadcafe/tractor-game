"""Seat-local model decision controller."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, final

from server.foundation.result import Ok, Rejected
from server.game import Seat, commands
from server.game.snapshots import PlayerSnapshot
from server.policy_model.actions import (
    GeneratedAction,
    build_legal_action_space,
    physical_command,
)
from server.policy_model.inference import (
    PolicyDecisionRequest,
    PolicyQuery,
    SamplingSeed,
)
from server.policy_model.observation import (
    ObservationMemory,
    build_observation,
)


@dataclass(frozen=True, slots=True)
class AIUnavailable:
    """An external AI deployment cannot complete a decision."""

    error: str

    def __post_init__(self) -> None:
        assert self.error


class AIControllerPort(Protocol):
    """Sequenced observation and decision boundary for AI players."""

    def observe(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
        error: str | None,
    ) -> None:
        """Record one complete player observation."""
        ...

    async def decide(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[commands.Command] | AIUnavailable:
        """Return one strategic command or external unavailability."""
        ...


class ControllerInference(Protocol):
    """The single model operation needed by an AI controller."""

    async def decide(
        self,
        *,
        request: PolicyDecisionRequest,
    ) -> Ok[GeneratedAction] | Rejected:
        """Sample one complete action from the learned policy."""
        ...


@final
class AIController:
    """Produce commands from one contiguous seat-local game history."""

    def __init__(
        self,
        *,
        seat: Seat,
        model: ControllerInference,
        random_source: random.Random,
    ) -> None:
        self._seat = seat
        self._model = model
        self._random = random_source
        self._memory = ObservationMemory()

    def observe(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
        error: str | None,
    ) -> None:
        """Consume one contiguous real player view."""
        remembered = self._memory.observe(
            seq=seq,
            snapshot=snapshot,
            error=error,
        )
        assert isinstance(remembered, Ok), remembered.reason

    async def decide(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[commands.Command] | AIUnavailable:
        """Sample one action directly from the learned policy."""
        assert snapshot.awaiting_action in (
            "bid",
            "stir",
            "discard",
            "play",
        )
        decided = await self._model.decide(
            request=PolicyDecisionRequest(
                query=self._root_query(snapshot),
                draw=SamplingSeed(
                    seed=self._random.getrandbits(63),
                    ordinal=seq,
                ),
            )
        )
        assert isinstance(decided, Ok), decided.reason
        command = physical_command(
            action=decided.value,
            hand=snapshot.hand,
        )
        assert isinstance(command, Ok), command.reason
        return command

    def _root_query(self, snapshot: PlayerSnapshot) -> PolicyQuery:
        observation = build_observation(
            viewer=self._seat,
            snapshot=snapshot,
            memory=self._memory.view(),
        )
        return PolicyQuery(
            observation=observation,
            legal_actions=build_legal_action_space(
                viewer=self._seat,
                snapshot=snapshot,
                query=observation.action_query,
            ),
        )


__all__ = (
    "AIController",
    "AIControllerPort",
    "AIUnavailable",
    "ControllerInference",
)
