"""Visible in-memory game records for AIPlayer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import assert_never

from server.sm.card_model import Suit, card_display
from server.sm.types import CompletedTrick
from server.snapshot import SerializedSuit, StateSnapshot

type AITrickKey = tuple[int, int, int, tuple[tuple[int, tuple[str, ...]], ...]]


@dataclass(frozen=True, slots=True)
class AIPlayRecord:
    player: int
    cards: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AITrickRecord:
    index: int
    lead_player: int
    plays: tuple[AIPlayRecord, ...]
    winner: int
    points: int


@dataclass(frozen=True, slots=True)
class AIBidRecord:
    player: int
    cards: tuple[str, ...]
    suit: SerializedSuit | None
    count: int


@dataclass(frozen=True, slots=True)
class AIFailedThrowRecord:
    player: int
    attempted_cards: tuple[str, ...]
    forced_cards: tuple[str, ...]


def _empty_bids() -> list[AIBidRecord]:
    return []


def _empty_tricks() -> list[AITrickRecord]:
    return []


def _empty_trick_keys() -> set[AITrickKey]:
    return set()


def _empty_failed_throws() -> list[AIFailedThrowRecord]:
    return []


@dataclass(slots=True)
class AIMemory:
    """Memory derived only from player-facing snapshots."""

    bids: list[AIBidRecord] = field(default_factory=_empty_bids)
    tricks: list[AITrickRecord] = field(default_factory=_empty_tricks)
    seen_trick_keys: set[AITrickKey] = field(default_factory=_empty_trick_keys)
    failed_throws: list[AIFailedThrowRecord] = field(default_factory=_empty_failed_throws)
    last_seq: int | None = None

    def update(self, snapshot: StateSnapshot, *, seq: int) -> None:
        self.last_seq = seq
        self.bids = [
            AIBidRecord(
                player=event.player,
                cards=tuple(card_display(card) for card in event.cards),
                suit=_suit_value(event.suit),
                count=event.count,
            )
            for event in snapshot.bid_events
        ]
        if snapshot.last_completed_trick is not None:
            key = _trick_key(snapshot.last_completed_trick)
            if key not in self.seen_trick_keys:
                self.seen_trick_keys.add(key)
                self.tricks.append(
                    _trick_record(
                        len(self.tricks) + 1,
                        snapshot.last_completed_trick,
                    )
                )
        if snapshot.failed_throw is not None:
            record = AIFailedThrowRecord(
                player=snapshot.failed_throw.player,
                attempted_cards=tuple(card_display(card) for card in snapshot.failed_throw.attempted_cards),
                forced_cards=tuple(card_display(card) for card in snapshot.failed_throw.forced_cards),
            )
            if record not in self.failed_throws:
                self.failed_throws.append(record)

    def summary(self) -> str:
        lines = ["AI 可见记忆:"]
        lines.append(f"- last_seq: {self.last_seq}")
        lines.append(f"- bid_count: {len(self.bids)}")
        if self.bids:
            last_bid = self.bids[-1]
            lines.append(
                f"- last_bid: player={last_bid.player}, cards={list(last_bid.cards)}, "
                f"suit={last_bid.suit}, count={last_bid.count}"
            )
        lines.append(f"- completed_tricks: {len(self.tricks)}")
        for trick in self.tricks[-6:]:
            play_text = "; ".join(
                f"p{play.player}={list(play.cards)}"
                for play in trick.plays
            )
            lines.append(
                f"- trick {trick.index}: lead={trick.lead_player}, "
                f"winner={trick.winner}, points={trick.points}, {play_text}"
            )
        if self.failed_throws:
            for item in self.failed_throws[-3:]:
                lines.append(
                    f"- failed_throw: player={item.player}, "
                    f"attempted={list(item.attempted_cards)}, forced={list(item.forced_cards)}"
                )
        return "\n".join(lines)


def _suit_value(suit: Suit | None) -> SerializedSuit | None:
    if suit is None:
        return None
    match suit:
        case Suit.HEARTS:
            return "hearts"
        case Suit.SPADES:
            return "spades"
        case Suit.DIAMONDS:
            return "diamonds"
        case Suit.CLUBS:
            return "clubs"
        case Suit.JOKER:
            return "joker"
    assert_never(suit)


def _trick_key(trick: CompletedTrick) -> AITrickKey:
    return (
        trick.lead_player,
        trick.winner,
        trick.points,
        tuple(
            (slot.player, tuple(card.id for card in slot.cards))
            for slot in trick.slots
        ),
    )


def _trick_record(index: int, trick: CompletedTrick) -> AITrickRecord:
    return AITrickRecord(
        index=index,
        lead_player=trick.lead_player,
        plays=tuple(
            AIPlayRecord(
                player=slot.player,
                cards=tuple(card_display(card) for card in slot.cards),
            )
            for slot in trick.slots
        ),
        winner=trick.winner,
        points=trick.points,
    )
