"""Deal-Bid combined state machine for 升级 (Shengji/Tractor).

Manages the combined dealing and bidding phase with progressive card
distribution. Players can reveal cards at any time during dealing. The
state machine tracks dealing progress and bid events. After all 100
cards are dealt, the highest bidder wins or it is a no-bid (空主) round.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from server.sm.card_model import Card, Rank, Suit
from server.sm.comparator import bid_value
from server.sm.constants import get_team_index, next_player_ccw
from server.sm.types import BidEvent


# ---- Data Models ----


class DealBidInput(BaseModel):
    """Input for creating a new deal-bid state machine."""

    model_config = ConfigDict(frozen=True)

    deck: list[Card]
    declarer_team: int | None
    trump_rank: Rank
    start_player: int


class DealBidResult(BaseModel):
    """Result produced when the deal-bid phase is complete."""

    model_config = ConfigDict(frozen=True)

    winner: int | None
    trump_suit: Suit | None
    bid_count: int
    players_hand: list[list[Card]]
    bid_events: list[BidEvent]


class DealBidState(BaseModel):
    """Internal state of the deal-bid state machine."""

    model_config = ConfigDict(frozen=True)

    phase: Literal["DEALING", "COMPLETE", "NO_BID"]
    deck: list[Card]
    deal_cursor: int
    deal_target: int
    bid_winner: BidEvent | None
    bid_events: list[BidEvent]
    players_hand: list[list[Card]]
    declarer_team: int | None
    trump_rank: Rank
    start_player: int


# ---- State Machine Operations ----


def create_deal_bid(input: DealBidInput) -> DealBidState:
    """Create a new deal-bid state machine with initial state."""
    return DealBidState(
        phase="DEALING",
        deck=input.deck,
        deal_cursor=0,
        deal_target=input.start_player,
        bid_winner=None,
        bid_events=[],
        players_hand=[[], [], [], []],
        declarer_team=input.declarer_team,
        trump_rank=input.trump_rank,
        start_player=input.start_player,
    )


def deal_next_card(state: DealBidState) -> DealBidState:
    """Deal the next card from the deck to the current target player.

    Preconditions: state.phase == "DEALING" and deal_cursor < 100.

    After dealing, if all 100 cards have been dealt:
      - If bid_winner is set: phase transitions to COMPLETE
      - If no bid_winner: phase transitions to NO_BID
    """
    if state.phase != "DEALING":
        return state

    cursor = state.deal_cursor
    if cursor >= 100:
        return state

    card = state.deck[cursor]
    target = state.deal_target

    # Append card to target player's hand
    new_hands = [list(hand) for hand in state.players_hand]
    new_hands[target] = new_hands[target] + [card]

    new_cursor = cursor + 1
    new_target = next_player_ccw(target)

    # Determine phase after dealing
    if new_cursor >= 100:
        if state.bid_winner is not None:
            new_phase: Literal["DEALING", "COMPLETE", "NO_BID"] = "COMPLETE"
        else:
            new_phase = "NO_BID"
    else:
        new_phase = "DEALING"

    return state.model_copy(update={
        "phase": new_phase,
        "deal_cursor": new_cursor,
        "deal_target": new_target,
        "players_hand": new_hands,
    })


def reveal(state: DealBidState, event: BidEvent) -> DealBidState:
    """Process a reveal (bid) event from a player.

    Validates the bid per spec section 5 and updates the bid_winner
    if the new bid has strictly higher value.

    Returns the updated state; if the bid is invalid, returns the
    original state unchanged.
    """
    # Precondition 1: must be in DEALING phase
    if state.phase != "DEALING":
        return state

    # Precondition 6: subsequent rounds - only declarer team can reveal
    if state.declarer_team is not None:
        player_team = get_team_index(event.player)
        if player_team != state.declarer_team:
            return state

    player_idx = event.player
    hand = state.players_hand[player_idx]

    # Precondition 4: cards must be in player's hand
    hand_ids = {c.id for c in hand}
    for card in event.cards:
        if card.id not in hand_ids:
            return state

    # Precondition 2 & 3: validate card types
    if event.kind == "trump_rank":
        # Each card must be trump_rank and same suit
        for card in event.cards:
            if card.rank != state.trump_rank:
                return state
            if card.suit != event.suit:
                return state
        if event.count not in (1, 2):
            return state
        if len(event.cards) != event.count:
            return state
    elif event.kind == "joker":
        # Must be a pair of jokers (same type), single joker rejected
        if event.count != 2:
            return state
        if len(event.cards) != 2:
            return state
        for card in event.cards:
            if not card.is_joker:
                return state
        # Both must be same joker type
        if event.cards[0].rank != event.cards[1].rank:
            return state
        if event.suit is not None:
            return state

    # Precondition 5: if bid_winner exists, new bid value must be strictly greater
    new_value = bid_value(event.cards, state.trump_rank)
    if new_value == 0:
        return state  # Invalid bid

    if state.bid_winner is not None:
        current_value = bid_value(state.bid_winner.cards, state.trump_rank)
        if new_value <= current_value:
            return state  # Not strictly greater

    # Valid bid: update state
    new_bid_events = state.bid_events + [event]

    return state.model_copy(update={
        "bid_winner": event,
        "bid_events": new_bid_events,
    })
