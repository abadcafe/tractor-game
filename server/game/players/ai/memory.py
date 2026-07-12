"""Visible in-memory game records for AIPlayer."""

from __future__ import annotations

from dataclasses import dataclass, field

from server.game.players.ai.formatting import card_text
from server.game.protocol import (
    CompletedTrickSnapshot,
    FailedThrowSnapshot,
    StateSnapshot,
)
from server.game.rules.cards import Suit

_SUIT_TEXT: dict[Suit, str] = {
    Suit.HEARTS: "红桃",
    Suit.SPADES: "黑桃",
    Suit.DIAMONDS: "方片",
    Suit.CLUBS: "梅花",
    Suit.JOKER: "王",
}

type AITrickKey = tuple[
    int,
    int,
    int,
    tuple[tuple[int, tuple[str, ...]], ...],
    tuple[int, tuple[str, ...], tuple[str, ...]] | None,
]


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
    suit: Suit | None
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
    seen_trick_keys: set[AITrickKey] = field(
        default_factory=_empty_trick_keys
    )
    failed_throws: list[AIFailedThrowRecord] = field(
        default_factory=_empty_failed_throws
    )
    last_seq: int | None = None

    def update(self, snapshot: StateSnapshot, *, seq: int) -> None:
        self.last_seq = seq
        self.bids = [
            AIBidRecord(
                player=event.player,
                cards=tuple(card_text(card) for card in event.cards),
                suit=event.suit,
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
            _append_failed_throw(
                self.failed_throws,
                snapshot.last_completed_trick.failed_throw,
            )
        if snapshot.trick is not None:
            _append_failed_throw(
                self.failed_throws,
                snapshot.trick.failed_throw,
            )

    def summary(self) -> str:
        lines = ["已知牌局记录:"]
        lines.append(
            f"- 最近状态序号：{_optional_seq_text(self.last_seq)}"
        )
        lines.append(f"- 亮主/反主记录数：{len(self.bids)}")
        if self.bids:
            last_bid = self.bids[-1]
            lines.append(
                f"- 最近亮主/反主：{_player_text(last_bid.player)}，"
                f"牌={list(last_bid.cards)}，"
                f"花色={_optional_suit_text(last_bid.suit)}，"
                f"张数={last_bid.count}"
            )
        lines.append(f"- 已完成墩数：{len(self.tricks)}")
        for trick in self.tricks[-6:]:
            play_text = "; ".join(
                f"{_player_text(play.player)}={list(play.cards)}"
                for play in trick.plays
            )
            lines.append(
                f"- 第 {trick.index} 墩：首出="
                f"{_player_text(trick.lead_player)}，"
                f"赢家={_player_text(trick.winner)}，"
                f"分={trick.points}，"
                f"{play_text}"
            )
        if self.failed_throws:
            for item in self.failed_throws[-3:]:
                lines.append(
                    f"- 甩牌失败：{_player_text(item.player)}，"
                    f"尝试甩={list(item.attempted_cards)}，"
                    f"被捡小={list(item.forced_cards)}"
                )
        return "\n".join(lines)


def _trick_key(trick: CompletedTrickSnapshot) -> AITrickKey:
    failed_throw_key: (
        tuple[int, tuple[str, ...], tuple[str, ...]] | None
    ) = None
    if trick.failed_throw is not None:
        failed_throw_key = (
            trick.failed_throw.player,
            tuple(
                card.id for card in trick.failed_throw.attempted_cards
            ),
            tuple(card.id for card in trick.failed_throw.forced_cards),
        )
    return (
        trick.lead_player,
        trick.winner,
        trick.points,
        tuple(
            (slot.player, tuple(card.id for card in slot.cards))
            for slot in trick.slots
        ),
        failed_throw_key,
    )


def _append_failed_throw(
    failed_throws: list[AIFailedThrowRecord],
    event: FailedThrowSnapshot | None,
) -> None:
    if event is None:
        return
    record = AIFailedThrowRecord(
        player=event.player,
        attempted_cards=tuple(
            card_text(card) for card in event.attempted_cards
        ),
        forced_cards=tuple(
            card_text(card) for card in event.forced_cards
        ),
    )
    if record not in failed_throws:
        failed_throws.append(record)


def _trick_record(
    index: int, trick: CompletedTrickSnapshot
) -> AITrickRecord:
    return AITrickRecord(
        index=index,
        lead_player=trick.lead_player,
        plays=tuple(
            AIPlayRecord(
                player=slot.player,
                cards=tuple(card_text(card) for card in slot.cards),
            )
            for slot in trick.slots
        ),
        winner=trick.winner,
        points=trick.points,
    )


def _player_text(player: int) -> str:
    return f"玩家 {player}"


def _optional_seq_text(seq: int | None) -> str:
    if seq is None:
        return "无"
    return str(seq)


def _optional_suit_text(suit: Suit | None) -> str:
    if suit is None:
        return "无主"
    return _SUIT_TEXT[suit]
