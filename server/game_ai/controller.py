"""Seat-local model decision controller."""

from __future__ import annotations

import logging
import random
import time
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

_LOGGER = logging.getLogger(__name__)


class AIControllerPort(Protocol):
    """Sequenced observation and decision boundary for AI players."""

    def observe(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
        error: str | None,
    ) -> Ok[None] | Rejected:
        """Record one complete player observation."""
        ...

    async def decide(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[commands.Command] | Rejected:
        """Return one strategic command for the current view."""
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
    ) -> Ok[None] | Rejected:
        """Consume one contiguous real player view."""
        remembered = self._memory.observe(
            seq=seq,
            snapshot=snapshot,
            error=error,
        )
        if isinstance(remembered, Rejected):
            return remembered
        return Ok(None)

    async def decide(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[commands.Command] | Rejected:
        """Sample one action directly from the learned policy."""
        if snapshot.awaiting_action not in (
            "bid",
            "stir",
            "discard",
            "play",
        ):
            return Rejected(
                reason="AI has no strategic action to decide"
            )
        started = time.perf_counter()
        decided = await self._model.decide(
            request=PolicyDecisionRequest(
                query=self._root_query(snapshot),
                draw=SamplingSeed(
                    seed=self._random.getrandbits(63),
                    ordinal=seq,
                ),
            )
        )
        if isinstance(decided, Rejected):
            return decided
        command = physical_command(
            action=decided.value,
            hand=snapshot.hand,
        )
        if isinstance(command, Rejected):
            return command
        _LOGGER.debug(
            "ai.decision kind=%s elapsed_ms=%.3f",
            snapshot.awaiting_action,
            (time.perf_counter() - started) * 1000.0,
        )
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
    "ControllerInference",
)
