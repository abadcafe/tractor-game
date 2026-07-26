"""Compose concrete bot players from requested policies."""

from __future__ import annotations

import random
from typing import assert_never

from server.game import Seat
from server.game_runtime.player import BotPolicyName, Player

from ._auto import AutoPolicy
from ._player import BotPlayer
from ._policy import DecisionPolicy
from .llm import LLMPolicy, LLMTranscript, LLMTranscriptRegistry


class DefaultBotPlayerFactory:
    """Create BotPlayer instances with stateful policies."""

    def __init__(
        self,
        transcripts: LLMTranscriptRegistry,
    ) -> None:
        self._transcripts = transcripts

    def create(
        self,
        seat: Seat,
        policy_name: BotPolicyName,
    ) -> Player:
        """Create one unstarted bot player."""
        policy: DecisionPolicy
        match policy_name:
            case "auto":
                policy = AutoPolicy(random.Random())
            case "llm":
                transcript = LLMTranscript()
                self._transcripts.register(seat, transcript)
                policy = LLMPolicy(
                    transcript=transcript,
                )
            case _:
                assert_never(policy_name)
        return BotPlayer(
            policy_name=policy_name,
            policy=policy,
        )


__all__ = ("DefaultBotPlayerFactory",)
