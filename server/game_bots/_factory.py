"""Compose concrete bot players from requested policies."""

from __future__ import annotations

import random
from typing import Protocol, assert_never

from server.foundation.result import Ok, Rejected
from server.game import Seat
from server.game_ai import AIControllerPort
from server.game_runtime.player import BotPolicyName, Player

from ._ai import AIPolicy
from ._auto import AutoPolicy
from ._player import BotPlayer
from ._policy import DecisionPolicy


class AIControllerFactory(Protocol):
    """Create an isolated controller for one AI-controlled seat."""

    def controller(
        self,
        seat: Seat,
    ) -> Ok[AIControllerPort] | Rejected:
        """Create a controller backed by the current model artifact."""
        ...


class DefaultBotPlayerFactory:
    """Create BotPlayer instances with stateful policies."""

    def __init__(
        self,
        ai_service: AIControllerFactory,
    ) -> None:
        self._ai_service = ai_service

    def create(
        self,
        seat: Seat,
        policy_name: BotPolicyName,
    ) -> Ok[Player] | Rejected:
        """Create one unstarted bot player."""
        policy: DecisionPolicy
        match policy_name:
            case "auto":
                policy = AutoPolicy(random.Random())
            case "ai":
                controller = self._ai_service.controller(seat)
                if isinstance(controller, Rejected):
                    return controller
                policy = AIPolicy(controller.value)
            case _:
                assert_never(policy_name)
        return Ok(
            BotPlayer(
                policy_name=policy_name,
                policy=policy,
            )
        )


__all__ = ("AIControllerFactory", "DefaultBotPlayerFactory")
