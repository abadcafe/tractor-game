"""Player action hint generation for game snapshots."""

from __future__ import annotations

from server.foundation.result import Ok, Rejected
from server.game.protocol import AwaitingAction
from server.game.rules import bid as bid_rules
from server.game.rules import hints as play_rules
from server.game.rules.cards import Card
from server.game.state_machine import deal_bid_sm, round_sm


def action_hints(
    *,
    awaiting_action: AwaitingAction | None,
    round_state: round_sm.RoundState | None,
    player_index: int,
    player_hand: list[Card],
) -> list[list[Card]]:
    """
    Return a closed card-group hint set for the current awaited action.
    """
    if round_state is None:
        return []

    if (
        awaiting_action == "bid"
        and round_state.phase == "DEAL_BID"
        and round_state.deal_bid_state is not None
    ):
        hints = deal_bid_sm.get_bid_action_hints(
            round_state.deal_bid_state, player_index
        )
        if len(hints) > deal_bid_sm.MAX_BID_ACTION_HINTS:
            return []
        return hints

    if (
        awaiting_action == "stir"
        and round_state.phase == "STIRRING"
        and round_state.stirring_state is not None
    ):
        hints = _legal_stir_actions(
            player_hand, round_state, player_index
        )
        if len(hints) > deal_bid_sm.MAX_BID_ACTION_HINTS:
            return []
        return hints

    if awaiting_action == "play" and round_state.phase == "PLAYING":
        return _play_action_hints(round_state, player_index)

    return []


def _legal_stir_actions(
    hand: list[Card],
    state: round_sm.RoundState,
    player_index: int,
) -> list[list[Card]]:
    result: list[list[Card]] = []
    for candidate in bid_rules.bid_card_candidates(
        hand, state.trump_rank
    ):
        if len(candidate) != 2:
            continue
        match round_sm.stir(state, player_index, candidate):
            case Ok():
                result.append(candidate)
            case Rejected():
                continue
    return bid_rules.sort_bid_action_hints(result, state.trump_rank)


def _play_action_hints(
    state: round_sm.RoundState, player_index: int
) -> list[list[Card]]:
    if state.phase != "PLAYING" or state.trick_state is None:
        return []
    if player_index != state.trick_state.cur:
        return []

    player_hand = list(state.players_hand[player_index])
    if state.trick_state.phase == "LEADING":
        return []

    lead_slots = state.trick_state.slots
    if not lead_slots:
        return []
    lead_cards = lead_slots[state.trick_state.lead_player].cards
    if not lead_cards:
        return []

    hints_result = play_rules.get_legal_play_hints(
        hand=player_hand,
        lead_cards=lead_cards,
        trump_suit=state.trump_suit,
        trump_rank=state.trump_rank,
        max_hints=play_rules.MAX_PLAY_ACTION_HINTS,
    )
    if isinstance(hints_result, Rejected):
        return []
    return play_rules.sort_play_action_hints(
        hints_result.value,
        trump_suit=state.trump_suit,
        trump_rank=state.trump_rank,
    )
