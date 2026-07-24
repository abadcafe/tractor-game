"""Typed command decoders used by in-process agents."""

from __future__ import annotations

from server.foundation.result import Ok
from server.game import CommandRejected, commands


class TypedCommandDecoder:
    """A decoder that already owns a typed command."""

    def __init__(self, command: commands.Command) -> None:
        self._command = command

    def decode(
        self,
    ) -> Ok[commands.Command] | CommandRejected:
        return Ok(self._command)
