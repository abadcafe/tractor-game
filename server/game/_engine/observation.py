"""Derive complete player snapshots from opaque game state."""

from __future__ import annotations

from server.game.rules import play
from server.game.rules.cards import Card, Rank, Suit
from server.game.seating import Partnership, Seat, SeatMap
from server.game.snapshots import PlayerSnapshot
from server.game.snapshots.contract import PendingTrump
from server.game.snapshots.events import StirringStateSnapshot

from . import phases
from .contract_projection import (
    contract_snapshot,
    deal_trump_rank,
)
from .history_projection import (
    bid_event,
    current_trick,
    optional_bid_event,
    optional_completed_trick,
    own_initial_exchange,
    stir_events,
)
from .review_projection import completed_round_snapshot
from .round_values import Hands
from .state import GameState, config_of, phase_of


def observe_game(state: GameState, viewer: Seat) -> PlayerSnapshot:
    """Return the entire reconnect-safe view for one seat."""
    phase = phase_of(state)
    if isinstance(phase, phases.AwaitingStart):
        return PlayerSnapshot(
            phase="WAITING",
            round_number=0,
            hand=(),
            remaining_cards=SeatMap(a=0, b=0, c=0, d=0),
            bottom_cards=(),
            trump=PendingTrump(rank=phase.levels.at(Partnership.FIRST)),
            declarer=None,
            defender_points=0,
            trick=None,
            last_completed_trick=None,
            defender_point_cards=(),
            awaiting_action=(
                "next_round" if viewer not in phase.confirmed else None
            ),
            scoring=None,
            winning_partnership=None,
            partnership_levels=phase.levels,
            mandatory_levels=config_of(state).mandatory_levels,
            bid_events=(),
            bid_winner=None,
            own_initial_bottom_exchange=None,
            stir_events=(),
            stirring_state=None,
            next_round_confirmed=phase.confirmed,
        )
    if isinstance(phase, phases.DealBid):
        rank = deal_trump_rank(phase)
        return PlayerSnapshot(
            phase="DEAL_BID",
            round_number=phase.round_number,
            hand=_sorted_hand(
                phase.hands.at(viewer),
                None,
                rank,
            ),
            remaining_cards=_hand_counts(phase.hands),
            bottom_cards=(),
            trump=PendingTrump(rank=rank),
            declarer=phase.fixed_declarer,
            defender_points=0,
            trick=None,
            last_completed_trick=None,
            defender_point_cards=(),
            awaiting_action=(
                "bid" if viewer == phase.current_actor else None
            ),
            scoring=None,
            winning_partnership=None,
            partnership_levels=phase.levels,
            mandatory_levels=config_of(state).mandatory_levels,
            bid_events=tuple(
                bid_event(event) for event in phase.bid_reveals
            ),
            bid_winner=optional_bid_event(phase.declaration),
            own_initial_bottom_exchange=None,
            stir_events=(),
            stirring_state=None,
            next_round_confirmed=frozenset(),
        )
    if isinstance(phase, phases.BottomExchange):
        hand = phase.hands.at(viewer)
        counts = _hand_counts(phase.hands)
        if viewer == phase.current_actor:
            hand = phase.hand_after_pickup
            counts = _replace_count(
                counts,
                viewer,
                len(phase.hand_after_pickup),
            )
        return PlayerSnapshot(
            phase="STIRRING",
            round_number=phase.round_number,
            hand=_sorted_hand(
                hand,
                phase.contract.trump_suit,
                phase.contract.trump_rank,
            ),
            remaining_cards=counts,
            bottom_cards=(
                phase.bottom_cards
                if viewer == phase.current_actor
                else ()
            ),
            trump=contract_snapshot(phase.contract),
            declarer=phase.contract.declarer,
            defender_points=0,
            trick=None,
            last_completed_trick=None,
            defender_point_cards=(),
            awaiting_action=(
                "discard" if viewer == phase.current_actor else None
            ),
            scoring=None,
            winning_partnership=None,
            partnership_levels=phase.levels,
            mandatory_levels=config_of(state).mandatory_levels,
            bid_events=tuple(
                bid_event(event) for event in phase.bid_reveals
            ),
            bid_winner=optional_bid_event(phase.declaration),
            own_initial_bottom_exchange=own_initial_exchange(
                phase.exchange_events,
                viewer,
            ),
            stir_events=stir_events(
                phase.stir_events,
                phase.exchange_events,
                viewer,
            ),
            stirring_state=StirringStateSnapshot(
                phase="EXCHANGING",
                trump_suit=phase.contract.trump_suit,
                current_actor=phase.current_actor,
                declarer=phase.contract.declarer,
                exchanging_actor=phase.current_actor,
                exchange_count=len(phase.bottom_cards),
            ),
            next_round_confirmed=frozenset(),
        )
    if isinstance(phase, phases.StirDecision):
        return PlayerSnapshot(
            phase="STIRRING",
            round_number=phase.round_number,
            hand=_sorted_hand(
                phase.hands.at(viewer),
                phase.contract.trump_suit,
                phase.contract.trump_rank,
            ),
            remaining_cards=_hand_counts(phase.hands),
            bottom_cards=(
                phase.bottom_cards
                if viewer == phase.bottom_owner
                else ()
            ),
            trump=contract_snapshot(phase.contract),
            declarer=phase.contract.declarer,
            defender_points=0,
            trick=None,
            last_completed_trick=None,
            defender_point_cards=(),
            awaiting_action=(
                "stir" if viewer == phase.current_actor else None
            ),
            scoring=None,
            winning_partnership=None,
            partnership_levels=phase.levels,
            mandatory_levels=config_of(state).mandatory_levels,
            bid_events=tuple(
                bid_event(event) for event in phase.bid_reveals
            ),
            bid_winner=optional_bid_event(phase.declaration),
            own_initial_bottom_exchange=own_initial_exchange(
                phase.exchange_events,
                viewer,
            ),
            stir_events=stir_events(
                phase.stir_events,
                phase.exchange_events,
                viewer,
            ),
            stirring_state=StirringStateSnapshot(
                phase="WAITING",
                trump_suit=phase.contract.trump_suit,
                current_actor=phase.current_actor,
                declarer=phase.contract.declarer,
                exchanging_actor=None,
                exchange_count=None,
            ),
            next_round_confirmed=frozenset(),
        )
    if isinstance(phase, phases.Playing):
        return PlayerSnapshot(
            phase="PLAYING",
            round_number=phase.round_number,
            hand=_sorted_hand(
                phase.hands.at(viewer),
                phase.contract.trump_suit,
                phase.contract.trump_rank,
            ),
            remaining_cards=_hand_counts(phase.hands),
            bottom_cards=(
                phase.bottom_cards
                if viewer == phase.bottom_owner
                else ()
            ),
            trump=contract_snapshot(phase.contract),
            declarer=phase.contract.declarer,
            defender_points=sum(
                card.points for card in phase.defender_point_cards
            ),
            trick=current_trick(phase),
            last_completed_trick=optional_completed_trick(
                phase.previous_trick
            ),
            defender_point_cards=phase.defender_point_cards,
            awaiting_action=(
                "play"
                if viewer == phase.current_trick.current_actor
                else None
            ),
            scoring=None,
            winning_partnership=None,
            partnership_levels=phase.levels,
            mandatory_levels=config_of(state).mandatory_levels,
            bid_events=tuple(
                bid_event(event) for event in phase.bid_reveals
            ),
            bid_winner=optional_bid_event(phase.declaration),
            own_initial_bottom_exchange=own_initial_exchange(
                phase.exchange_events,
                viewer,
            ),
            stir_events=stir_events(
                phase.stir_events,
                phase.exchange_events,
                viewer,
            ),
            stirring_state=None,
            next_round_confirmed=frozenset(),
        )
    if isinstance(phase, phases.RoundReview):
        return completed_round_snapshot(
            phase.completed,
            viewer,
            phase.confirmed,
            config_of(state).mandatory_levels,
            winning_partnership=None,
        )
    assert isinstance(phase, phases.Finished)
    return completed_round_snapshot(
        phase.completed,
        viewer,
        frozenset(),
        config_of(state).mandatory_levels,
        winning_partnership=phase.winning_partnership,
    )


def _sorted_hand(
    hand: tuple[Card, ...],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> tuple[Card, ...]:
    return play.sort_hand(hand, trump_suit, trump_rank)


def _hand_counts(hands: Hands) -> SeatMap[int]:
    return hands.map(len)


def _replace_count(
    counts: SeatMap[int],
    seat: Seat,
    value: int,
) -> SeatMap[int]:
    return counts.replace(seat, value)
