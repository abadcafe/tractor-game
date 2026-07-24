"""The single public rejection type of the game aggregate."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommandRejected:
    """A command rejected as normal domain input."""

    reason: str
