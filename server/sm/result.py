"""Result type for state-machine operations that may reject invalid input.

Used to communicate precise rejection reasons back to the caller
without raising exceptions (which would break the async game loop).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Successful state-machine transition, carrying the new state."""

    value: T


@dataclass(frozen=True, slots=True)
class Rejected:
    """State-machine transition was rejected; carries a human-readable reason."""

    reason: str


type StateResult[T] = Ok[T] | Rejected
