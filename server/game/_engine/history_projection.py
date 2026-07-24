"""Project durable round events and trick facts."""

from server.game.config import Seat
from server.game.snapshots.events import (
    BidEventSnapshot,
    BottomExchangeSnapshot,
    StirDeclarationEventSnapshot,
)
from server.game.snapshots.tricks import (
    CompletedTrickSnapshot,
    FailedThrowSnapshot,
    TrickSlotSnapshot,
    TrickSnapshot,
)

from . import phases
from .round_values import (
    BottomExchangeCause,
    BottomExchangeEvent,
    Declaration,
    InternalCompletedTrick,
    InternalFailedThrow,
    StirEvent,
)


def bid_event(event: Declaration) -> BidEventSnapshot:
    first = event.cards[0]
    is_joker = first.is_joker
    return BidEventSnapshot(
        player=event.actor,
        cards=event.cards,
        kind="joker" if is_joker else "trump_rank",
        suit=None if is_joker else first.suit,
        joker_type=(
            "big"
            if first.is_big_joker
            else "small"
            if is_joker
            else None
        ),
        count=len(event.cards),
        deal_ordinal=event.deal_ordinal,
    )


def optional_bid_event(
    event: Declaration | None,
) -> BidEventSnapshot | None:
    if event is None:
        return None
    return bid_event(event)


def own_initial_exchange(
    events: tuple[BottomExchangeEvent, ...],
    viewer: Seat,
) -> BottomExchangeSnapshot | None:
    for event in events:
        if (
            event.actor == viewer
            and event.cause == BottomExchangeCause.INITIAL
        ):
            return _exchange_snapshot(event)
    return None


def stir_events(
    events: tuple[StirEvent, ...],
    exchanges: tuple[BottomExchangeEvent, ...],
    viewer: Seat,
) -> tuple[StirDeclarationEventSnapshot, ...]:
    own_by_index = {
        event.stir_event_index: _exchange_snapshot(event)
        for event in exchanges
        if event.actor == viewer
        and event.cause == BottomExchangeCause.STIR
    }
    result: list[StirDeclarationEventSnapshot] = []
    for index, event in enumerate(events):
        is_stir = bool(event.cards)
        first = event.cards[0] if is_stir else None
        result.append(
            StirDeclarationEventSnapshot(
                player=event.actor,
                kind="stir" if is_stir else "pass",
                cards=event.cards,
                new_suit=(
                    None
                    if first is None or first.is_joker
                    else first.suit
                ),
                priority=event.priority,
                own_bottom_exchange=own_by_index.get(index),
            )
        )
    return tuple(result)


def current_trick(phase: phases.Playing) -> TrickSnapshot:
    trick = phase.current_trick
    return TrickSnapshot(
        lead_player=trick.lead_actor,
        slots=tuple(
            TrickSlotSnapshot(
                player=slot.actor,
                cards=slot.cards,
            )
            for slot in trick.slots
        ),
        current_player=trick.current_actor,
        failed_throw=_failed_throw(trick.failed_throw),
    )


def completed_trick(
    trick: InternalCompletedTrick,
) -> CompletedTrickSnapshot:
    return CompletedTrickSnapshot(
        lead_player=trick.lead_actor,
        slots=tuple(
            TrickSlotSnapshot(
                player=slot.actor,
                cards=slot.cards,
            )
            for slot in trick.slots
        ),
        winner=trick.winner,
        points=trick.points,
        failed_throw=_failed_throw(trick.failed_throw),
    )


def optional_completed_trick(
    trick: InternalCompletedTrick | None,
) -> CompletedTrickSnapshot | None:
    if trick is None:
        return None
    return completed_trick(trick)


def _exchange_snapshot(
    event: BottomExchangeEvent,
) -> BottomExchangeSnapshot:
    return BottomExchangeSnapshot(
        player=event.actor,
        trigger=(
            "initial"
            if event.cause == BottomExchangeCause.INITIAL
            else "stir"
        ),
        stir_event_index=event.stir_event_index,
        picked_up_bottom_cards=event.picked_up,
        discarded_bottom_cards=event.discarded,
        resulting_bottom_cards=event.resulting_bottom,
    )


def _failed_throw(
    failed: InternalFailedThrow | None,
) -> FailedThrowSnapshot | None:
    if failed is None:
        return None
    return FailedThrowSnapshot(
        player=failed.actor,
        attempted_cards=failed.attempted_cards,
        forced_cards=failed.forced_cards,
    )
