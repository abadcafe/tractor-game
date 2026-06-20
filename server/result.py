"""Shared Result primitives for operations that may reject normal input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Successful operation result, carrying the produced value."""

    value: T


@dataclass(frozen=True, slots=True)
class Rejected:
    """Operation rejected normal input; carries a user-facing reason."""

    reason: str
