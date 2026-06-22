"""Deal-Bid combined state machine for 升级 (Shengji/Tractor).

Manages the combined dealing and bidding phase with progressive card
distribution. Players can reveal cards at any time during dealing. The
state machine tracks dealing progress and bid events. After all 100
cards are dealt, the highest bidder wins or it is a no-bid (空主) round.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from server.result import Ok, Rejected
from server.rules import bid as bid_rules
from server.rules.cards import Card, Rank, Suit
from server.rules.rejections import (
    CardNotInHandRejected,
    CurrentBidWinnerCannotRebidRejected,
)

from .constants import next_player_ccw
from .rejections import (
    AllCardsDealtRejected,
    BidNotAllowedInDealBidPhaseRejected,
    DealCardNotAllowedInDealBidPhaseRejected,
    DealNotCompleteRejected,
    InvalidPlayerIndexRejected,
)
from .types import BidEvent, DealBidPhase

MAX_BID_ACTION_HINTS: int = bid_rules.MAX_BID_ACTION_HINTS


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

    phase: DealBidPhase
    deck: list[Card]
    deal_cursor: int
    deal_target: int
    bid_winner: BidEvent | None
    bid_events: list[BidEvent]
    players_hand: list[list[Card]]
    declarer_team: int | None
    trump_rank: Rank
    start_player: int
    all_dealt: bool = False


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


def deal_next_card(state: DealBidState) -> Ok[DealBidState] | Rejected:
    """Deal the next card from the deck to the current target player.

    After dealing, if all 100 cards have been dealt, phase stays DEALING
    with all_dealt=True so the last card recipient can bid or skip.
    Call finalize_dealing() after their action to complete the phase.

    Returns Ok(new_state) on success, Rejected(reason) if not in DEALING
    phase
    or all cards have already been dealt.
    """
    if state.phase != "DEALING":
        return DealCardNotAllowedInDealBidPhaseRejected(state.phase)

    cursor = state.deal_cursor
    if cursor >= 100:
        return AllCardsDealtRejected()

    card = state.deck[cursor]
    target = state.deal_target

    # Append card to target player's hand
    new_hands = [list(hand) for hand in state.players_hand]
    new_hands[target] = new_hands[target] + [card]

    new_cursor = cursor + 1
    new_target = next_player_ccw(target)

    # After the last card, keep phase as DEALING so the recipient can
    # act.
    # Set all_dealt=True so Game can finalize after their action.
    all_dealt = new_cursor >= 100

    return Ok(
        state.model_copy(
            update={
                "phase": "DEALING",
                "deal_cursor": new_cursor,
                "deal_target": new_target,
                "players_hand": new_hands,
                "all_dealt": all_dealt,
            }
        )
    )


def get_bid_action_hints(
    state: DealBidState, player: int
) -> list[list[Card]]:
    """Compute accepted logical bid hints for one player.

    The returned options are advisory UI hints, so a non-empty result
    must
    match the same validation path used by real bid actions. Candidate
    card
    groups are generated from the player's hand as logical bid options,
    then
    filtered through reveal() against the full deal-bid state so current
    bid-priority and card-ownership rules are respected.
    """
    if state.phase != "DEALING" or player < 0 or player >= 4:
        return []
    if (
        state.bid_winner is not None
        and state.bid_winner.player == player
    ):
        return []

    current_cards = (
        state.bid_winner.cards if state.bid_winner is not None else None
    )
    return bid_rules.legal_bid_hints(
        state.players_hand[player], state.trump_rank, current_cards
    )


def reveal(
    state: DealBidState, event: BidEvent
) -> Ok[DealBidState] | Rejected:
    """Process a reveal (bid) event from a player.

    Validates the bid per spec section 5 and updates the bid_winner
    if the new bid has strictly higher value.

    Returns Ok(new_state) on success, Rejected(reason) on invalid input.
    """
    # Precondition 0: player index must be valid
    if event.player < 0 or event.player >= 4:
        return InvalidPlayerIndexRejected(event.player)

    # Precondition 1: must be in DEALING phase
    if state.phase != "DEALING":
        return BidNotAllowedInDealBidPhaseRejected(state.phase)

    player_idx = event.player
    hand = state.players_hand[player_idx]

    # Precondition 4: cards must be in player's hand
    hand_ids = {c.id for c in hand}
    for card in event.cards:
        if card.id not in hand_ids:
            return CardNotInHandRejected(card.id)

    match bid_rules.validate_distinct_bid_cards(event.cards):
        case Ok():
            pass
        case Rejected() as rejected:
            return rejected

    match bid_rules.validate_bid_cards(
        kind=event.kind,
        cards=event.cards,
        declared_suit=event.suit,
        declared_count=event.count,
        trump_rank=state.trump_rank,
    ):
        case Ok():
            pass
        case Rejected() as rejected:
            return rejected

    current_cards = (
        state.bid_winner.cards if state.bid_winner is not None else None
    )
    if (
        state.bid_winner is not None
        and state.bid_winner.player == event.player
    ):
        return CurrentBidWinnerCannotRebidRejected()
    match bid_rules.bid_beats_current(
        event.cards, current_cards, state.trump_rank
    ):
        case Ok():
            pass
        case Rejected() as rejected:
            return rejected

    # Valid bid: update state
    new_bid_events = state.bid_events + [event]

    return Ok(
        state.model_copy(
            update={
                "bid_winner": event,
                "bid_events": new_bid_events,
            }
        )
    )


def finalize_dealing(
    state: DealBidState,
) -> Ok[DealBidState] | Rejected:
    """
    After all cards are dealt and the last recipient has acted,
    finalize.

    Transitions phase to COMPLETE (if bid_winner exists) or NO_BID.
    Must only be called when all_dealt=True.

    Returns Ok(new_state) on success, Rejected(reason) if not ready.
    """
    if not state.all_dealt:
        return DealNotCompleteRejected()
    new_phase: DealBidPhase = (
        "COMPLETE" if state.bid_winner is not None else "NO_BID"
    )
    return Ok(state.model_copy(update={"phase": new_phase}))
