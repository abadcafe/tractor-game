"""Formatting helpers for player-facing cards."""

from __future__ import annotations

from server.rules.cards import Card, card_display


def card_text(card: Card) -> str:
    """Format a protocol card with the shared rule-layer display."""
    return card_display(card)


def card_points(card: Card) -> int:
    return card.points
