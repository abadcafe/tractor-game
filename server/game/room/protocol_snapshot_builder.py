"""Conversion helpers from internal SM models to player snapshots."""

from __future__ import annotations

from server.game.protocol import (
    BidEventSnapshot,
    BottomExchangeSnapshot,
    CompletedTrickSnapshot,
    FailedThrowSnapshot,
    ScoringSnapshot,
    StirDeclarationEventSnapshot,
    StirringPhase,
    StirringStateSnapshot,
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.game.rules.cards import Card, Suit
from server.game.state_machine.types import (
    BidEvent,
    BottomExchangeEvent,
    CompletedTrick,
    FailedThrow,
    StirDeclarationEvent,
)


def bid_event_snapshot(event: BidEvent) -> BidEventSnapshot:
    return BidEventSnapshot(
        player=event.player,
        cards=event.cards,
        kind=event.kind,
        suit=event.suit,
        joker_type=event.joker_type,
        count=event.count,
    )


def optional_bid_event_snapshot(
    event: BidEvent | None,
) -> BidEventSnapshot | None:
    if event is None:
        return None
    return bid_event_snapshot(event)


def stir_declaration_event_snapshot(
    event: StirDeclarationEvent,
    *,
    own_bottom_exchange: BottomExchangeEvent | None = None,
) -> StirDeclarationEventSnapshot:
    return StirDeclarationEventSnapshot(
        player=event.player,
        kind=event.kind,
        cards=list(event.cards),
        new_suit=event.new_suit,
        priority=event.priority,
        own_bottom_exchange=None
        if own_bottom_exchange is None
        else bottom_exchange_snapshot(own_bottom_exchange),
    )


def bottom_exchange_snapshot(
    event: BottomExchangeEvent,
) -> BottomExchangeSnapshot:
    return BottomExchangeSnapshot(
        picked_up_bottom_cards=list(event.picked_up_bottom_cards),
        discarded_bottom_cards=list(event.discarded_bottom_cards),
    )


def failed_throw_snapshot(event: FailedThrow) -> FailedThrowSnapshot:
    return FailedThrowSnapshot(
        player=event.player,
        attempted_cards=event.attempted_cards,
        forced_cards=event.forced_cards,
    )


def optional_failed_throw_snapshot(
    event: FailedThrow | None,
) -> FailedThrowSnapshot | None:
    if event is None:
        return None
    return failed_throw_snapshot(event)


def completed_trick_snapshot(
    trick: CompletedTrick,
) -> CompletedTrickSnapshot:
    return CompletedTrickSnapshot(
        lead_player=trick.lead_player,
        slots=[
            trick_slot_snapshot(slot.player, slot.cards)
            for slot in trick.slots
        ],
        winner=trick.winner,
        points=trick.points,
        failed_throw=optional_failed_throw_snapshot(trick.failed_throw),
    )


def optional_completed_trick_snapshot(
    trick: CompletedTrick | None,
) -> CompletedTrickSnapshot | None:
    if trick is None:
        return None
    return completed_trick_snapshot(trick)


def trick_slot_snapshot(
    player: int, cards: list[Card]
) -> TrickSlotSnapshot:
    return TrickSlotSnapshot(player=player, cards=cards)


def trick_snapshot(
    *,
    lead_player: int,
    slots: list[TrickSlotSnapshot],
    current_player: int,
    failed_throw: FailedThrow | None,
) -> TrickSnapshot:
    return TrickSnapshot(
        lead_player=lead_player,
        slots=slots,
        current_player=current_player,
        failed_throw=optional_failed_throw_snapshot(failed_throw),
    )


def scoring_snapshot(
    *,
    round_winning_team: int,
    defender_points: int,
    total_defender_points: int,
    bottom_card_bonus: int,
    bottom_cards: list[Card],
) -> ScoringSnapshot:
    return ScoringSnapshot(
        round_winning_team=round_winning_team,
        defender_points=defender_points,
        total_defender_points=total_defender_points,
        bottom_card_bonus=bottom_card_bonus,
        bottom_cards=bottom_cards,
    )


def stirring_state_snapshot(
    *,
    phase: StirringPhase,
    trump_suit: Suit | None,
    current_player: int,
    declarer_player: int,
    exchanging_player: int | None,
    exchange_count: int | None,
) -> StirringStateSnapshot:
    return StirringStateSnapshot(
        phase=phase,
        trump_suit=trump_suit,
        current_player=current_player,
        declarer_player=declarer_player,
        exchanging_player=exchanging_player,
        exchange_count=exchange_count,
    )
