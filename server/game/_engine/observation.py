"""Derive complete player snapshots from opaque game state."""

from __future__ import annotations

from server.game.config import Seat
from server.game.rules import play
from server.game.rules.cards import Card, Rank, Suit
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
            player_hand=(),
            player_hand_counts=(0, 0, 0, 0),
            bottom_cards=(),
            trump=PendingTrump(rank=phase.levels[0]),
            declarer_player=None,
            defender_points=0,
            trick=None,
            last_completed_trick=None,
            defender_point_cards=(),
            awaiting_action=(
                "next_round" if viewer not in phase.confirmed else None
            ),
            scoring=None,
            winning_team=None,
            team_levels=phase.levels,
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
            player_hand=_sorted_hand(
                phase.hands[int(viewer)],
                None,
                rank,
            ),
            player_hand_counts=_hand_counts(phase.hands),
            bottom_cards=(),
            trump=PendingTrump(rank=rank),
            declarer_player=phase.fixed_declarer,
            defender_points=0,
            trick=None,
            last_completed_trick=None,
            defender_point_cards=(),
            awaiting_action=(
                "bid" if viewer == phase.current_actor else None
            ),
            scoring=None,
            winning_team=None,
            team_levels=phase.levels,
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
        hand = phase.hands[int(viewer)]
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
            player_hand=_sorted_hand(
                hand,
                phase.contract.trump_suit,
                phase.contract.trump_rank,
            ),
            player_hand_counts=counts,
            bottom_cards=(
                phase.bottom_cards
                if viewer == phase.current_actor
                else ()
            ),
            trump=contract_snapshot(phase.contract),
            declarer_player=phase.contract.declarer,
            defender_points=0,
            trick=None,
            last_completed_trick=None,
            defender_point_cards=(),
            awaiting_action=(
                "discard" if viewer == phase.current_actor else None
            ),
            scoring=None,
            winning_team=None,
            team_levels=phase.levels,
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
                current_player=phase.current_actor,
                declarer_player=phase.contract.declarer,
                exchanging_player=phase.current_actor,
                exchange_count=len(phase.bottom_cards),
            ),
            next_round_confirmed=frozenset(),
        )
    if isinstance(phase, phases.StirDecision):
        return PlayerSnapshot(
            phase="STIRRING",
            round_number=phase.round_number,
            player_hand=_sorted_hand(
                phase.hands[int(viewer)],
                phase.contract.trump_suit,
                phase.contract.trump_rank,
            ),
            player_hand_counts=_hand_counts(phase.hands),
            bottom_cards=(
                phase.bottom_cards
                if viewer == phase.bottom_owner
                else ()
            ),
            trump=contract_snapshot(phase.contract),
            declarer_player=phase.contract.declarer,
            defender_points=0,
            trick=None,
            last_completed_trick=None,
            defender_point_cards=(),
            awaiting_action=(
                "stir" if viewer == phase.current_actor else None
            ),
            scoring=None,
            winning_team=None,
            team_levels=phase.levels,
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
                current_player=phase.current_actor,
                declarer_player=phase.contract.declarer,
                exchanging_player=None,
                exchange_count=None,
            ),
            next_round_confirmed=frozenset(),
        )
    if isinstance(phase, phases.Playing):
        return PlayerSnapshot(
            phase="PLAYING",
            round_number=phase.round_number,
            player_hand=_sorted_hand(
                phase.hands[int(viewer)],
                phase.contract.trump_suit,
                phase.contract.trump_rank,
            ),
            player_hand_counts=_hand_counts(phase.hands),
            bottom_cards=(
                phase.bottom_cards
                if viewer == phase.bottom_owner
                else ()
            ),
            trump=contract_snapshot(phase.contract),
            declarer_player=phase.contract.declarer,
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
            winning_team=None,
            team_levels=phase.levels,
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
            winning_team=None,
        )
    assert isinstance(phase, phases.Finished)
    return completed_round_snapshot(
        phase.completed,
        viewer,
        frozenset(),
        config_of(state).mandatory_levels,
        winning_team=phase.winning_team,
    )


def _sorted_hand(
    hand: tuple[Card, ...],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> tuple[Card, ...]:
    return play.sort_hand(hand, trump_suit, trump_rank)


def _hand_counts(hands: Hands) -> tuple[int, int, int, int]:
    return (
        len(hands[0]),
        len(hands[1]),
        len(hands[2]),
        len(hands[3]),
    )


def _replace_count(
    counts: tuple[int, int, int, int],
    seat: Seat,
    value: int,
) -> tuple[int, int, int, int]:
    values = list(counts)
    values[int(seat)] = value
    return (values[0], values[1], values[2], values[3])
