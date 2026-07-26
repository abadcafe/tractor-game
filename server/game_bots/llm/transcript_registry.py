"""Per-game lookup of LLM decision transcripts."""

from __future__ import annotations

from server.game import Seat

from .transcript import LLMTranscript


class LLMTranscriptRegistry:
    """Own transcript stores without exposing runtime players."""

    def __init__(self) -> None:
        self._transcripts: dict[Seat, LLMTranscript] = {}

    def register(
        self,
        seat: Seat,
        transcript: LLMTranscript,
    ) -> None:
        """Register exactly one transcript for a seat."""
        assert seat not in self._transcripts
        self._transcripts[seat] = transcript

    def at(self, seat: Seat) -> LLMTranscript | None:
        """Return the transcript registered for one LLM seat."""
        return self._transcripts.get(seat)


__all__ = ("LLMTranscriptRegistry",)
