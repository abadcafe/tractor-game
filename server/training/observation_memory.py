"""Continuous player-visible history for training observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from server.foundation.result import Ok, Rejected
from server.game import Seat
from server.game.rules.cards import Card
from server.game.snapshots import PlayerSnapshot
from server.game.snapshots.tricks import CompletedTrickSnapshot
from server.training.observation_structure import RoundEventOrdinal

type BidDisposition = Literal["pass", "reveal"]
type CompletedTrickKey = tuple[
    Seat,
    Seat,
    tuple[tuple[Seat, tuple[str, ...]], ...],
]
type PendingBid = tuple[Seat, RoundEventOrdinal]


class _ObservationMemoryRejected(Rejected):
    """A complete player-visible round history cannot be produced."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ObservedBidAction:
    """One public bid decision at a precise deal ordinal."""

    actor: Seat
    disposition: BidDisposition
    revealed_cards: tuple[Card, ...]
    deal_ordinal: RoundEventOrdinal


@dataclass(frozen=True, slots=True)
class ObservationMemoryView:
    """Immutable player-visible history accumulated for this round."""

    bid_actions: tuple[ObservedBidAction, ...]
    completed_tricks: tuple[CompletedTrickSnapshot, ...]


@dataclass(slots=True)
class ObservationMemory:
    """Accumulate public history from contiguous state pushes."""

    _last_seq: int | None = None
    _previous: PlayerSnapshot | None = None
    _pending_bid: PendingBid | None = None
    _seen_bid_count: int = 0
    _bid_actions: tuple[ObservedBidAction, ...] = ()
    _completed_tricks: tuple[CompletedTrickSnapshot, ...] = ()
    _completed_keys: frozenset[CompletedTrickKey] = frozenset()

    def observe(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
        error: str | None = None,
    ) -> Ok[ObservationMemoryView] | Rejected:
        """Consume one contiguous player snapshot."""
        if error is not None:
            if self._last_seq is None or seq != self._last_seq:
                return _ObservationMemoryRejected(
                    "observation memory received an unknown-state error"
                )
            return Ok(value=self.view())

        if self._last_seq is None:
            initial = self._accept_initial(seq, snapshot)
            if isinstance(initial, Rejected):
                return initial
            return Ok(value=self.view())

        if seq == self._last_seq:
            previous = self._previous
            assert previous is not None
            assert snapshot == previous
            return Ok(value=self.view())

        expected = self._last_seq + 1
        if seq != expected:
            return _ObservationMemoryRejected(
                f"observation memory missed state sequence {expected}"
            )

        previous = self._previous
        assert previous is not None
        if previous.phase in ("SCORING", "WAITING") and (
            snapshot.phase == "DEAL_BID"
        ):
            self._clear_round()

        bid_result = self._settle_pending_bid(snapshot)
        if isinstance(bid_result, Rejected):
            return bid_result
        self._record_completed(snapshot)
        self._pending_bid = _pending_bid(previous, snapshot)
        self._seen_bid_count = len(snapshot.bid_events)
        self._last_seq = seq
        self._previous = snapshot
        return Ok(value=self.view())

    def view(self) -> ObservationMemoryView:
        """Return an immutable snapshot of the accumulated memory."""
        return ObservationMemoryView(
            bid_actions=self._bid_actions,
            completed_tricks=self._completed_tricks,
        )

    def reset_episode(self) -> None:
        """Forget all sequence and round state before a new game."""
        self._last_seq = None
        self._previous = None
        self._clear_round()

    def fork(self) -> ObservationMemory:
        """Return an independent continuation of this exact history."""
        return ObservationMemory(
            _last_seq=self._last_seq,
            _previous=self._previous,
            _pending_bid=self._pending_bid,
            _seen_bid_count=self._seen_bid_count,
            _bid_actions=self._bid_actions,
            _completed_tricks=self._completed_tricks,
            _completed_keys=self._completed_keys,
        )

    def _accept_initial(
        self,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[None] | Rejected:
        if snapshot.phase == "WAITING":
            pass
        elif snapshot.phase == "DEAL_BID":
            if (
                sum(snapshot.remaining_cards.values()) > 1
                or snapshot.bid_events
            ):
                return _ObservationMemoryRejected(
                    "observation memory did not observe round start"
                )
        else:
            return _ObservationMemoryRejected(
                "observation memory did not observe round start"
            )
        self._last_seq = seq
        self._previous = snapshot
        self._seen_bid_count = len(snapshot.bid_events)
        self._pending_bid = _initial_pending_bid(snapshot)
        self._record_completed(snapshot)
        return Ok(value=None)

    def _settle_pending_bid(
        self, snapshot: PlayerSnapshot
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
            if event.actor != actor:
                return _ObservationMemoryRejected(
                    "bid event actor does not match observed bidder"
                )
            action = ObservedBidAction(
                actor=actor,
                disposition="reveal",
                revealed_cards=tuple(event.cards),
                deal_ordinal=deal_ordinal,
            )
        self._bid_actions = (*self._bid_actions, action)
        return Ok(value=None)

    def _record_completed(self, snapshot: PlayerSnapshot) -> None:
        completed = snapshot.last_completed_trick
        if completed is None:
            return
        key = _completed_trick_key(completed)
        if key in self._completed_keys:
            return
        self._completed_keys = self._completed_keys.union((key,))
        self._completed_tricks = (*self._completed_tricks, completed)

    def _clear_round(self) -> None:
        self._pending_bid = None
        self._seen_bid_count = 0
        self._bid_actions = ()
        self._completed_tricks = ()
        self._completed_keys = frozenset()


def _initial_pending_bid(
    snapshot: PlayerSnapshot,
) -> PendingBid | None:
    if snapshot.phase != "DEAL_BID":
        return None
    dealt = sum(snapshot.remaining_cards.values())
    if dealt == 0:
        return None
    actors = [
        seat
        for seat, count in snapshot.remaining_cards.items()
        if count == 1
    ]
    if len(actors) != 1:
        return None
    return (actors[0], RoundEventOrdinal(dealt))


def _pending_bid(
    previous: PlayerSnapshot, current: PlayerSnapshot
) -> PendingBid | None:
    if current.phase != "DEAL_BID":
        return None
    previous_counts = previous.remaining_cards
    current_counts = current.remaining_cards
    increased = [
        seat
        for seat, count in current_counts.items()
        if count == previous_counts.at(seat) + 1
    ]
    if len(increased) != 1:
        return None
    return (
        increased[0],
        RoundEventOrdinal(sum(current_counts.values())),
    )


def _completed_trick_key(
    trick: CompletedTrickSnapshot,
) -> CompletedTrickKey:
    return (
        trick.lead_actor,
        trick.winner,
        tuple(
            (slot.actor, tuple(card.id for card in slot.cards))
            for slot in trick.slots
        ),
    )


__all__ = (
    "ObservedBidAction",
    "ObservationMemory",
    "ObservationMemoryView",
)
