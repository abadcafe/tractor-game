"""Continuous player-visible history for training observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from server.foundation.result import Ok, Rejected
from server.game.protocol import (
    CompletedTrickSnapshot,
    StateMessage,
    StateSnapshot,
)
from server.game.rules.cards import Card
from server.training.observation_structure import RoundEventOrdinal

type BidDisposition = Literal["pass", "reveal"]
type CompletedTrickKey = tuple[
    int,
    int,
    tuple[tuple[int, tuple[str, ...]], ...],
]
type PendingBid = tuple[int, RoundEventOrdinal]


class _ObservationMemoryRejected(Rejected):
    """A complete player-visible round history cannot be produced."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ObservedBidAction:
    """One public bid decision at a precise deal ordinal."""

    actor: int
    disposition: BidDisposition
    revealed_cards: tuple[Card, ...]
    deal_ordinal: RoundEventOrdinal


@dataclass(frozen=True, slots=True)
class ObservationMemoryView:
    """Immutable player-visible history accumulated for this round."""

    bid_actions: tuple[ObservedBidAction, ...]
    completed_tricks: tuple[CompletedTrickSnapshot, ...]


def _bid_actions() -> list[ObservedBidAction]:
    return []


def _completed_tricks() -> list[CompletedTrickSnapshot]:
    return []


def _completed_keys() -> set[CompletedTrickKey]:
    return set()


@dataclass(slots=True)
class ObservationMemory:
    """Accumulate public history from contiguous state pushes."""

    _last_seq: int | None = None
    _previous: StateSnapshot | None = None
    _pending_bid: PendingBid | None = None
    _seen_bid_count: int = 0
    _bid_actions: list[ObservedBidAction] = field(
        default_factory=_bid_actions
    )
    _completed_tricks: list[CompletedTrickSnapshot] = field(
        default_factory=_completed_tricks
    )
    _completed_keys: set[CompletedTrickKey] = field(
        default_factory=_completed_keys
    )

    def observe(
        self, message: StateMessage
    ) -> Ok[ObservationMemoryView] | Rejected:
        """Consume one state envelope and return the current memory."""
        if message.error is not None:
            if self._last_seq is None or message.seq != self._last_seq:
                return _ObservationMemoryRejected(
                    "observation memory received an unknown-state error"
                )
            return Ok(value=self.view())

        if self._last_seq is None:
            initial = self._accept_initial(message)
            if isinstance(initial, Rejected):
                return initial
            return Ok(value=self.view())

        if message.seq == self._last_seq:
            previous = self._previous
            assert previous is not None
            assert message.state == previous
            return Ok(value=self.view())

        expected = self._last_seq + 1
        if message.seq != expected:
            return _ObservationMemoryRejected(
                f"observation memory missed state sequence {expected}"
            )

        previous = self._previous
        assert previous is not None
        if previous.phase in ("SCORING", "WAITING") and (
            message.state.phase == "DEAL_BID"
        ):
            self._clear_round()

        bid_result = self._settle_pending_bid(message.state)
        if isinstance(bid_result, Rejected):
            return bid_result
        self._record_completed(message.state)
        self._pending_bid = _pending_bid(previous, message.state)
        self._seen_bid_count = len(message.state.bid_events)
        self._last_seq = message.seq
        self._previous = message.state
        return Ok(value=self.view())

    def view(self) -> ObservationMemoryView:
        """Return an immutable snapshot of the accumulated memory."""
        return ObservationMemoryView(
            bid_actions=tuple(self._bid_actions),
            completed_tricks=tuple(self._completed_tricks),
        )

    def reset_episode(self) -> None:
        """Forget all sequence and round state before a new game."""
        self._last_seq = None
        self._previous = None
        self._clear_round()

    def _accept_initial(
        self, message: StateMessage
    ) -> Ok[None] | Rejected:
        snapshot = message.state
        if snapshot.phase == "WAITING":
            pass
        elif snapshot.phase == "DEAL_BID":
            if (
                sum(snapshot.player_hand_counts) > 1
                or snapshot.bid_events
            ):
                return _ObservationMemoryRejected(
                    "observation memory did not observe round start"
                )
        else:
            return _ObservationMemoryRejected(
                "observation memory did not observe round start"
            )
        self._last_seq = message.seq
        self._previous = snapshot
        self._seen_bid_count = len(snapshot.bid_events)
        self._pending_bid = _initial_pending_bid(snapshot)
        self._record_completed(snapshot)
        return Ok(value=None)

    def _settle_pending_bid(
        self, snapshot: StateSnapshot
    ) -> Ok[None] | Rejected:
        pending = self._pending_bid
        if pending is None:
            if len(snapshot.bid_events) != self._seen_bid_count:
                return _ObservationMemoryRejected(
                    "bid history changed without an observed decision"
                )
            return Ok(value=None)
        actor, deal_ordinal = pending
        new_count = len(snapshot.bid_events) - self._seen_bid_count
        if new_count not in (0, 1):
            return _ObservationMemoryRejected(
                "bid history appended more than one decision"
            )
        if new_count == 0:
            action = ObservedBidAction(
                actor=actor,
                disposition="pass",
                revealed_cards=(),
                deal_ordinal=deal_ordinal,
            )
        else:
            event = snapshot.bid_events[-1]
            if event.player != actor:
                return _ObservationMemoryRejected(
                    "bid event actor does not match observed bidder"
                )
            action = ObservedBidAction(
                actor=actor,
                disposition="reveal",
                revealed_cards=tuple(event.cards),
                deal_ordinal=deal_ordinal,
            )
        self._bid_actions.append(action)
        return Ok(value=None)

    def _record_completed(self, snapshot: StateSnapshot) -> None:
        completed = snapshot.last_completed_trick
        if completed is None:
            return
        key = _completed_trick_key(completed)
        if key in self._completed_keys:
            return
        self._completed_keys.add(key)
        self._completed_tricks.append(completed)

    def _clear_round(self) -> None:
        self._pending_bid = None
        self._seen_bid_count = 0
        self._bid_actions.clear()
        self._completed_tricks.clear()
        self._completed_keys.clear()


def _initial_pending_bid(
    snapshot: StateSnapshot,
) -> PendingBid | None:
    if snapshot.phase != "DEAL_BID":
        return None
    dealt = sum(snapshot.player_hand_counts)
    if dealt == 0:
        return None
    actors = [
        player
        for player, count in enumerate(snapshot.player_hand_counts)
        if count == 1
    ]
    if len(actors) != 1:
        return None
    return (actors[0], RoundEventOrdinal(dealt))


def _pending_bid(
    previous: StateSnapshot, current: StateSnapshot
) -> PendingBid | None:
    if current.phase != "DEAL_BID":
        return None
    previous_counts = previous.player_hand_counts
    current_counts = current.player_hand_counts
    increased = [
        player
        for player, count in enumerate(current_counts)
        if player < len(previous_counts)
        and count == previous_counts[player] + 1
    ]
    if len(increased) != 1:
        return None
    return (
        increased[0],
        RoundEventOrdinal(sum(current_counts)),
    )


def _completed_trick_key(
    trick: CompletedTrickSnapshot,
) -> CompletedTrickKey:
    return (
        trick.lead_player,
        trick.winner,
        tuple(
            (slot.player, tuple(card.id for card in slot.cards))
            for slot in trick.slots
        ),
    )


__all__ = (
    "ObservedBidAction",
    "ObservationMemory",
    "ObservationMemoryView",
)
