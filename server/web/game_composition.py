"""Compose one game room and its policy diagnostics."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from server.game import GameConfig, GameSeed
from server.game_bots import DefaultBotPlayerFactory
from server.game_bots.llm import LLMTranscriptRegistry
from server.game_runtime import BotPolicyName, GameRoom


@dataclass(frozen=True, slots=True)
class GameInstance:
    """One room plus diagnostics owned by the web composition root."""

    room: GameRoom
    llm_transcripts: LLMTranscriptRegistry


def create_game_instance() -> GameInstance:
    """Create one independently seeded game instance."""
    transcripts = LLMTranscriptRegistry()
    room = GameRoom(
        GameConfig(),
        GameSeed(secrets.randbits(64)),
        DefaultBotPlayerFactory(transcripts),
    )
    return GameInstance(
        room=room,
        llm_transcripts=transcripts,
    )


def bot_policy_name_from_str(
    value: str | None,
) -> BotPolicyName | None:
    """Parse one closed bot-policy name."""
    normalized = (value or "").strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized == "llm":
        return "llm"
    return None


__all__ = (
    "GameInstance",
    "bot_policy_name_from_str",
    "create_game_instance",
)
