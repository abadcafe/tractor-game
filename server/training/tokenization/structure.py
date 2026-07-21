"""Flat token nodes and their non-lexical structure addresses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

from server.training.tokenization.payloads import (
    ActionToken,
    CardToken,
    GlobalToken,
    RoundToken,
    TokenPayload,
    TrickToken,
)

type PayloadRole = Literal[
    "hand",
    "visible_bottom",
    "bid_reveal",
    "stir_reveal",
    "exchange_pickup",
    "exchange_discard",
    "played",
    "revealed_extra",
]


class TokenFamily(IntEnum):
    """The five semantic token families; zero remains tensor padding."""

    GLOBAL_CONTEXT = 1
    ROUND_CONTEXT = 2
    TRICK_CONTEXT = 3
    ACTION = 4
    CARD = 5


@dataclass(frozen=True, slots=True)
class TokenAddress:
    """Temporal and ownership structure independent of token content."""

    round_event_time: int | None = None
    trick_time: int | None = None
    action_position: int | None = None
    payload_role: PayloadRole | None = None


@dataclass(frozen=True, slots=True)
class TokenNode:
    """One semantic payload at one structural address."""

    payload: TokenPayload
    address: TokenAddress

    @property
    def family(self) -> TokenFamily:
        """Return the unique family implied by the payload type."""
        if isinstance(self.payload, GlobalToken):
            return TokenFamily.GLOBAL_CONTEXT
        if isinstance(self.payload, RoundToken):
            return TokenFamily.ROUND_CONTEXT
        if isinstance(self.payload, TrickToken):
            return TokenFamily.TRICK_CONTEXT
        if isinstance(self.payload, ActionToken):
            return TokenFamily.ACTION
        assert isinstance(self.payload, CardToken)
        return TokenFamily.CARD


@dataclass(frozen=True, slots=True)
class TokenSequence:
    """A flat typed sequence with one explicit query anchor."""

    nodes: tuple[TokenNode, ...]
    query_index: int

    def __post_init__(self) -> None:
        assert 0 <= self.query_index < len(self.nodes)
        query = self.nodes[self.query_index].payload
        assert isinstance(query, ActionToken)
        assert query.occurrence == "query"


__all__ = (
    "TokenAddress",
    "TokenFamily",
    "TokenNode",
    "TokenSequence",
)
