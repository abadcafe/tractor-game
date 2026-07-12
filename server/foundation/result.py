"""
Shared Result primitives for operations that may reject normal input.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Ok[T]:
    """Successful operation result, carrying the produced value."""

    value: T


@dataclass(frozen=True, slots=True)
class Rejected:
    """Operation rejected normal input; carries a user-facing reason."""

    reason: str
