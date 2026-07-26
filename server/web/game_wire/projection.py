"""Project domain snapshots into the browser wire protocol."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from server.game import (
    Partnership,
    PartnershipId,
    Seat,
    SeatId,
    partnership_id,
    seat_id,
    snapshots,
)
from server.game.rules import bidding, play
from server.game.rules.cards import Card, Rank, Suit
from server.game.snapshots.contract import (
    NoTrump,
    PendingTrump,
    SuitedTrump,
)
from server.game.snapshots.events import (
    BidEventSnapshot,
    BottomExchangeSnapshot,
    StirDeclarationEventSnapshot,
    StirringStateSnapshot,
)
from server.game.snapshots.review import ScoringSnapshot
from server.game.snapshots.tricks import (
    CompletedTrickSnapshot,
    FailedThrowSnapshot,
    TrickSlotSnapshot,
    TrickSnapshot,
)

_MAX_BID_HINTS = 10


class _WireModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CardWire(_WireModel):
    id: str
    suit: Suit
    rank: Rank


class RemainingCardsWire(_WireModel):
    a: int
    b: int
    c: int
    d: int


class PartnershipLevelsWire(_WireModel):
    first: Rank
    second: Rank


class PendingTrumpWire(_WireModel):
    kind: Literal["pending"] = "pending"
    rank: Rank


class NoTrumpWire(_WireModel):
    kind: Literal["no_trump"] = "no_trump"
    rank: Rank


class SuitedTrumpWire(_WireModel):
    kind: Literal["suited"] = "suited"
    rank: Rank
    suit: Suit


type TrumpWire = Annotated[
    PendingTrumpWire | NoTrumpWire | SuitedTrumpWire,
    Field(discriminator="kind"),
]


class TrickSlotWire(_WireModel):
    actor: SeatId
    cards: tuple[CardWire, ...]


class FailedThrowWire(_WireModel):
    actor: SeatId
    attempted_cards: tuple[CardWire, ...]
    forced_cards: tuple[CardWire, ...]


class TrickWire(_WireModel):
    lead_actor: SeatId
    slots: tuple[TrickSlotWire, ...]
    current_actor: SeatId
    failed_throw: FailedThrowWire | None


class CompletedTrickWire(_WireModel):
    lead_actor: SeatId
    slots: tuple[TrickSlotWire, ...]
    winner: SeatId
    points: int
    failed_throw: FailedThrowWire | None


class BidEventWire(_WireModel):
    actor: SeatId
    cards: tuple[CardWire, ...]
    kind: Literal["trump_rank", "joker"]
    suit: Suit | None
    joker_type: Literal["big", "small"] | None
    count: int
    deal_ordinal: int


class BottomExchangeWire(_WireModel):
    actor: SeatId
    trigger: Literal["initial", "stir"]
    stir_event_index: int | None
    picked_up_bottom_cards: tuple[CardWire, ...]
    discarded_bottom_cards: tuple[CardWire, ...]
    resulting_bottom_cards: tuple[CardWire, ...]


class StirEventWire(_WireModel):
    actor: SeatId
    kind: Literal["stir", "pass"]
    cards: tuple[CardWire, ...]
    new_suit: Suit | None
    priority: int | None
    own_bottom_exchange: BottomExchangeWire | None


class StirringStateWire(_WireModel):
    phase: Literal["WAITING", "EXCHANGING"]
    trump_suit: Suit | None
    current_actor: SeatId
    declarer: SeatId
    exchanging_actor: SeatId | None
    exchange_count: int | None


class ScoringWire(_WireModel):
    winning_partnership: PartnershipId
    defender_points: int
    total_defender_points: int
    bottom_card_bonus: int
    bottom_cards: tuple[CardWire, ...]


class StateWire(_WireModel):
    """Complete reconnect-safe browser state."""

    phase: Literal["DEAL_BID", "STIRRING", "PLAYING", "WAITING"]
    round_number: int
    hand: tuple[CardWire, ...]
    remaining_cards: RemainingCardsWire
    bottom_cards: tuple[CardWire, ...]
    trump: TrumpWire
    declarer: SeatId | None
    defender_points: int
    trick: TrickWire | None
    last_completed_trick: CompletedTrickWire | None
    defender_point_cards: tuple[CardWire, ...]
    action_hints: tuple[tuple[CardWire, ...], ...]
    awaiting_action: (
        Literal["bid", "discard", "stir", "play", "next_round"] | None
    )
    scoring: ScoringWire | None
    winning_partnership: PartnershipId | None
    partnership_levels: PartnershipLevelsWire
    mandatory_levels: tuple[Rank, ...]
    bid_events: tuple[BidEventWire, ...]
    bid_winner: BidEventWire | None
    own_initial_bottom_exchange: BottomExchangeWire | None
    stir_events: tuple[StirEventWire, ...]
    stirring_state: StirringStateWire | None
    next_round_confirmed: tuple[SeatId, ...]


class StateMessage(_WireModel):
    """Server-to-browser state envelope."""

    type: Literal["state"] = "state"
    seq: int
    state: StateWire
    error: str | None


def encode_state(
    viewer: Seat,
    seq: int,
    snapshot: snapshots.PlayerSnapshot,
    error: str | None,
) -> StateMessage:
    """Encode one player view without serializing domain models."""
    return StateMessage(
        seq=seq,
        state=StateWire(
            phase=snapshot.phase,
            round_number=snapshot.round_number,
            hand=_cards(snapshot.hand),
            remaining_cards=RemainingCardsWire(
                a=snapshot.remaining_cards.at(Seat.A),
                b=snapshot.remaining_cards.at(Seat.B),
                c=snapshot.remaining_cards.at(Seat.C),
                d=snapshot.remaining_cards.at(Seat.D),
            ),
            bottom_cards=_cards(snapshot.bottom_cards),
            trump=_trump(snapshot),
            declarer=_seat_id(snapshot.declarer),
            defender_points=snapshot.defender_points,
            trick=_trick(snapshot.trick),
            last_completed_trick=_completed_trick(
                snapshot.last_completed_trick
            ),
            defender_point_cards=_cards(snapshot.defender_point_cards),
            action_hints=tuple(
                _cards(hint) for hint in _action_hints(viewer, snapshot)
            ),
            awaiting_action=snapshot.awaiting_action,
            scoring=_scoring(snapshot.scoring),
            winning_partnership=_partnership_id(
                snapshot.winning_partnership
            ),
            partnership_levels=PartnershipLevelsWire(
                first=snapshot.partnership_levels.at(Partnership.FIRST),
                second=snapshot.partnership_levels.at(
                    Partnership.SECOND
                ),
            ),
            mandatory_levels=snapshot.mandatory_levels,
            bid_events=tuple(
                _bid_event(event) for event in snapshot.bid_events
            ),
            bid_winner=_optional_bid_event(snapshot.bid_winner),
            own_initial_bottom_exchange=_bottom_exchange(
                snapshot.own_initial_bottom_exchange
            ),
            stir_events=tuple(
                _stir_event(event) for event in snapshot.stir_events
            ),
            stirring_state=_stirring_state(snapshot.stirring_state),
            next_round_confirmed=tuple(
                seat_id(seat)
                for seat in sorted(
                    snapshot.next_round_confirmed,
                    key=seat_id,
                )
            ),
        ),
        error=error,
    )


def _card(card: Card) -> CardWire:
    return CardWire(id=card.id, suit=card.suit, rank=card.rank)


def _cards(cards: tuple[Card, ...]) -> tuple[CardWire, ...]:
    return tuple(_card(card) for card in cards)


def _seat_id(seat: Seat | None) -> SeatId | None:
    if seat is None:
        return None
    return seat_id(seat)


def _required_seat_id(seat: Seat) -> SeatId:
    return seat_id(seat)


def _partnership_id(
    partnership: Partnership | None,
) -> PartnershipId | None:
    if partnership is None:
        return None
    return partnership_id(partnership)


def _required_partnership_id(
    partnership: Partnership,
) -> PartnershipId:
    return partnership_id(partnership)


def _trump(snapshot: snapshots.PlayerSnapshot) -> TrumpWire:
    trump = snapshot.trump
    if isinstance(trump, PendingTrump):
        return PendingTrumpWire(rank=trump.rank)
    if isinstance(trump, NoTrump):
        return NoTrumpWire(rank=trump.rank)
    assert isinstance(trump, SuitedTrump)
    return SuitedTrumpWire(rank=trump.rank, suit=trump.suit)


def _trick_slot(slot: TrickSlotSnapshot) -> TrickSlotWire:
    return TrickSlotWire(
        actor=_required_seat_id(slot.actor),
        cards=_cards(slot.cards),
    )


def _failed_throw(
    failed: FailedThrowSnapshot | None,
) -> FailedThrowWire | None:
    if failed is None:
        return None
    return FailedThrowWire(
        actor=_required_seat_id(failed.actor),
        attempted_cards=_cards(failed.attempted_cards),
        forced_cards=_cards(failed.forced_cards),
    )


def _trick(trick: TrickSnapshot | None) -> TrickWire | None:
    if trick is None:
        return None
    return TrickWire(
        lead_actor=_required_seat_id(trick.lead_actor),
        slots=tuple(_trick_slot(slot) for slot in trick.slots),
        current_actor=_required_seat_id(trick.current_actor),
        failed_throw=_failed_throw(trick.failed_throw),
    )


def _completed_trick(
    trick: CompletedTrickSnapshot | None,
) -> CompletedTrickWire | None:
    if trick is None:
        return None
    return CompletedTrickWire(
        lead_actor=_required_seat_id(trick.lead_actor),
        slots=tuple(_trick_slot(slot) for slot in trick.slots),
        winner=_required_seat_id(trick.winner),
        points=trick.points,
        failed_throw=_failed_throw(trick.failed_throw),
    )


def _bid_event(event: BidEventSnapshot) -> BidEventWire:
    return BidEventWire(
        actor=_required_seat_id(event.actor),
        cards=_cards(event.cards),
        kind=event.kind,
        suit=event.suit,
        joker_type=event.joker_type,
        count=event.count,
        deal_ordinal=event.deal_ordinal,
    )


def _optional_bid_event(
    event: BidEventSnapshot | None,
) -> BidEventWire | None:
    if event is None:
        return None
    return _bid_event(event)


def _bottom_exchange(
    exchange: BottomExchangeSnapshot | None,
) -> BottomExchangeWire | None:
    if exchange is None:
        return None
    return BottomExchangeWire(
        actor=_required_seat_id(exchange.actor),
        trigger=exchange.trigger,
        stir_event_index=exchange.stir_event_index,
        picked_up_bottom_cards=_cards(exchange.picked_up_bottom_cards),
        discarded_bottom_cards=_cards(exchange.discarded_bottom_cards),
        resulting_bottom_cards=_cards(exchange.resulting_bottom_cards),
    )


def _stir_event(
    event: StirDeclarationEventSnapshot,
) -> StirEventWire:
    return StirEventWire(
        actor=_required_seat_id(event.actor),
        kind=event.kind,
        cards=_cards(event.cards),
        new_suit=event.new_suit,
        priority=event.priority,
        own_bottom_exchange=_bottom_exchange(event.own_bottom_exchange),
    )


def _stirring_state(
    state: StirringStateSnapshot | None,
) -> StirringStateWire | None:
    if state is None:
        return None
    return StirringStateWire(
        phase=state.phase,
        trump_suit=state.trump_suit,
        current_actor=_required_seat_id(state.current_actor),
        declarer=_required_seat_id(state.declarer),
        exchanging_actor=_seat_id(state.exchanging_actor),
        exchange_count=state.exchange_count,
    )


def _scoring(
    scoring: ScoringSnapshot | None,
) -> ScoringWire | None:
    if scoring is None:
        return None
    return ScoringWire(
        winning_partnership=_required_partnership_id(
            scoring.winning_partnership
        ),
        defender_points=scoring.defender_points,
        total_defender_points=scoring.total_defender_points,
        bottom_card_bonus=scoring.bottom_card_bonus,
        bottom_cards=_cards(scoring.bottom_cards),
    )


def _action_hints(
    viewer: Seat,
    snapshot: snapshots.PlayerSnapshot,
) -> tuple[tuple[Card, ...], ...]:
    action = snapshot.awaiting_action
    if action == "bid":
        if (
            snapshot.bid_winner is not None
            and snapshot.bid_winner.actor == viewer
        ):
            return ()
        reveals = bidding.legal_reveals(
            snapshot.hand,
            snapshot.trump_rank,
            _declaration(snapshot),
        )
        if len(reveals) > _MAX_BID_HINTS:
            return ()
        return tuple(reveal.cards for reveal in reveals)
    if action == "stir":
        reveals = tuple(
            reveal
            for reveal in bidding.legal_reveals(
                snapshot.hand,
                snapshot.trump_rank,
                _declaration(snapshot),
            )
            if len(reveal.cards) == 2
        )
        if len(reveals) > _MAX_BID_HINTS:
            return ()
        return tuple(reveal.cards for reveal in reveals)
    if action == "play":
        lead = None
        if snapshot.trick is not None and snapshot.trick.slots:
            lead = snapshot.trick.slots[0].cards
        return play.closed_hints(
            snapshot.hand,
            lead,
            snapshot.trump_suit,
            snapshot.trump_rank,
        )
    return ()


def _declaration(
    snapshot: snapshots.PlayerSnapshot,
) -> bidding.Declaration | None:
    if snapshot.bid_winner is None:
        return None
    return bidding.Declaration(cards=snapshot.bid_winner.cards)
