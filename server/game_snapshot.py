"""Build player-facing snapshots from game and round state."""

from __future__ import annotations

from server import game_hints
from server.protocol import (
    AwaitingAction,
    RoundPhase,
    ScoringSnapshot,
    StateSnapshot,
    StirringStateSnapshot,
    TrickSnapshot,
)
from server.protocol_snapshot_builder import (
    bid_event_snapshot,
    optional_bid_event_snapshot,
    optional_completed_trick_snapshot,
    optional_failed_throw_snapshot,
    scoring_snapshot,
    stirring_state_snapshot,
    trick_slot_snapshot,
    trick_snapshot,
)
from server.rules.cards import Card, Rank
from server.rules.ordering import sort_by_display_order
from server.sm import game_sm, round_sm
from server.sm.types import BidEvent, FailedThrow


def trump_rank_for_round(state: game_sm.GameState) -> Rank:
    """
    Return the level rank played by the next round's declarer team.
    """
    if state.declarer_team == 1:
        return state.team1_level
    return state.team0_level


def build_state_snapshot(
    *,
    for_player: int,
    game_state: game_sm.GameState,
    round_state: round_sm.RoundState | None,
    bid_turn: int,
    next_round_confirmed: set[int],
) -> StateSnapshot:
    if round_state is None:
        awaiting_action: AwaitingAction | None
        if for_player not in next_round_confirmed:
            awaiting_action = "next_round"
        else:
            awaiting_action = None
        return StateSnapshot(
            phase="WAITING",
            player_hand=[],
            player_hand_counts=[0, 0, 0, 0],
            bottom_cards=[],
            trump_suit=None,
            trump_rank=game_state.team0_level,
            declarer_team=None,
            declarer_player=None,
            defender_points=0,
            trick=None,
            last_completed_trick=None,
            defender_point_cards=[],
            failed_throw=None,
            action_hints=[],
            awaiting_action=awaiting_action,
            scoring=None,
            winning_team=None,
            team0_level=game_state.team0_level,
            team1_level=game_state.team1_level,
            bid_events=[],
            bid_winner=None,
            stirring_state=None,
            next_round_confirmed=sorted(next_round_confirmed),
        )

    player_hand = (
        list(round_state.players_hand[for_player])
        if for_player < len(round_state.players_hand)
        else []
    )
    player_hand_counts = [
        len(hand) for hand in round_state.players_hand
    ]

    if (
        round_state.phase == "STIRRING"
        and round_state.stirring_state is not None
        and round_state.stirring_state.phase == "EXCHANGING"
        and round_state.stirring_state.exchanging_player == for_player
        and round_state.stirring_state.exchange_state is not None
    ):
        player_hand = list(
            round_state.stirring_state.exchange_state.hand_after_pickup
        )
        player_hand_counts[for_player] = len(player_hand)

    player_hand = sort_by_display_order(
        player_hand, round_state.trump_suit, round_state.trump_rank
    )
    awaiting_action = _awaiting_action(
        game_state=game_state,
        round_state=round_state,
        for_player=for_player,
        bid_turn=bid_turn,
        next_round_confirmed=next_round_confirmed,
    )
    action_hints = game_hints.action_hints(
        awaiting_action=awaiting_action,
        round_state=round_state,
        player_index=for_player,
        player_hand=player_hand,
    )

    return StateSnapshot(
        phase=_phase(round_state),
        player_hand=player_hand,
        player_hand_counts=player_hand_counts,
        bottom_cards=_visible_bottom_cards(
            for_player=for_player,
            round_state=round_state,
        ),
        trump_suit=round_state.trump_suit,
        trump_rank=round_state.trump_rank,
        declarer_team=round_state.declarer_team,
        declarer_player=_snapshot_declarer_player(round_state),
        defender_points=round_state.defender_points,
        trick=_current_trick_snapshot(round_state),
        last_completed_trick=optional_completed_trick_snapshot(
            round_state.last_completed_trick
        ),
        defender_point_cards=list(round_state.defender_point_cards),
        failed_throw=optional_failed_throw_snapshot(
            _current_failed_throw(round_state)
        ),
        action_hints=action_hints,
        awaiting_action=awaiting_action,
        scoring=_scoring_snapshot(round_state),
        winning_team=game_state.winning_team,
        team0_level=game_state.team0_level,
        team1_level=game_state.team1_level,
        bid_events=[
            bid_event_snapshot(event)
            for event in _bid_events(round_state)
        ],
        bid_winner=optional_bid_event_snapshot(round_state.bid_winner),
        stirring_state=_stirring_snapshot(round_state),
        next_round_confirmed=sorted(next_round_confirmed),
    )


def _phase(round_state: round_sm.RoundState | None) -> RoundPhase:
    if round_state is None:
        return "WAITING"
    return round_state.phase


def _snapshot_declarer_player(
    state: round_sm.RoundState,
) -> int | None:
    if (
        state.phase == "DEAL_BID"
        and state.next_declarer_player is not None
    ):
        return state.next_declarer_player
    return state.declarer_player


def _awaiting_action(
    *,
    game_state: game_sm.GameState,
    round_state: round_sm.RoundState,
    for_player: int,
    bid_turn: int,
    next_round_confirmed: set[int],
) -> AwaitingAction | None:
    if game_state.winning_team is not None:
        return None
    if round_state.phase == "DEAL_BID":
        if for_player == bid_turn:
            return "bid"
        return None
    if (
        round_state.phase == "STIRRING"
        and round_state.stirring_state is not None
    ):
        if (
            round_state.stirring_state.phase == "EXCHANGING"
            and for_player
            == round_state.stirring_state.exchanging_player
        ):
            return "discard"
        if (
            round_state.stirring_state.phase == "WAITING"
            and for_player == round_state.stirring_state.current_player
        ):
            return "stir"
    if (
        round_state.phase == "PLAYING"
        and _can_act_in_playing(round_state)
        and round_state.trick_state is not None
        and for_player == round_state.trick_state.cur
    ):
        return "play"
    if (
        round_state.phase == "WAITING"
        and for_player not in next_round_confirmed
    ):
        return "next_round"
    return None


def _can_act_in_playing(state: round_sm.RoundState) -> bool:
    if state.phase != "PLAYING" or state.trick_state is None:
        return False
    if state.trick_state.phase == "LEADING":
        return True
    lead_slots = state.trick_state.slots
    if not lead_slots:
        return False
    lead_cards = lead_slots[state.trick_state.lead_player].cards
    return bool(lead_cards)


def _current_trick_snapshot(
    state: round_sm.RoundState,
) -> TrickSnapshot | None:
    if state.phase != "PLAYING" or state.trick_state is None:
        return None
    trick_state = state.trick_state
    return trick_snapshot(
        lead_player=trick_state.lead_player,
        slots=[
            trick_slot_snapshot(slot.player, slot.cards)
            for slot in trick_state.slots
        ],
        current_player=trick_state.cur,
    )


def _current_failed_throw(
    state: round_sm.RoundState,
) -> FailedThrow | None:
    if state.phase == "PLAYING" and state.trick_state is not None:
        return state.trick_state.failed_throw
    return None


def _bid_events(state: round_sm.RoundState) -> list[BidEvent]:
    if state.deal_bid_state is None:
        return []
    return list(state.deal_bid_state.bid_events)


def _stirring_snapshot(
    state: round_sm.RoundState,
) -> StirringStateSnapshot | None:
    if state.stirring_state is None or state.phase != "STIRRING":
        return None
    exchanging_player: int | None = None
    exchange_count: int | None = None
    if state.stirring_state.phase == "EXCHANGING":
        exchanging_player = state.stirring_state.exchanging_player
        if state.stirring_state.exchange_state is not None:
            exchange_count = state.stirring_state.exchange_state.count
    return stirring_state_snapshot(
        phase=state.stirring_state.phase,
        trump_suit=state.stirring_state.trump_suit,
        current_player=state.stirring_state.current_player,
        declarer_player=state.stirring_state.declarer_player,
        exchanging_player=exchanging_player,
        exchange_count=exchange_count,
    )


def _scoring_snapshot(
    state: round_sm.RoundState,
) -> ScoringSnapshot | None:
    if state.result is None:
        return None
    return scoring_snapshot(
        declarer_team=state.declarer_team,
        defender_points=state.defender_points,
        total_defender_points=state.result.total_defender_points,
        bottom_card_bonus=state.result.bottom_card_bonus,
        bottom_cards=list(state.bottom_cards),
    )


def _visible_bottom_cards(
    *,
    for_player: int,
    round_state: round_sm.RoundState,
) -> list[Card]:
    if round_state.result is not None:
        return list(round_state.bottom_cards)
    stirring_state = round_state.stirring_state
    if stirring_state is None:
        return []
    if stirring_state.phase == "EXCHANGING":
        return []
    if stirring_state.bottom_owner_player == for_player:
        return list(round_state.bottom_cards)
    return []
