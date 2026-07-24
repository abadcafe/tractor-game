"""Construct configured runtime agents."""

from __future__ import annotations

import random

from server.game import Seat
from server.game_runtime.room import BotKind, RuntimeAgent
from server.game_runtime.session import AgentSubmission

from .auto import AutoAgent
from .llm import LLMAgent


class DefaultAgentFactory:
    """Create the configured concrete agent for a bot kind."""

    def create(
        self,
        seat: Seat,
        kind: BotKind,
        submission: AgentSubmission,
    ) -> RuntimeAgent:
        if kind == BotKind.AI:
            return LLMAgent(seat, submission)
        return AutoAgent(submission, random.Random())
