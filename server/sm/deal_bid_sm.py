"""Deal-Bid combined state machine for 升级 (Shengji/Tractor).

Manages the combined dealing and bidding phase with progressive card
distribution. Players can reveal cards at any time during dealing. The
state machine tracks dealing progress and bid events. After all 100
cards are dealt, the highest bidder wins or it is a no-bid (空主) round.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from server.result import Ok, Rejected

from .card_model import Card, Rank, Suit
from .comparator import bid_value
from .constants import next_player_ccw
from .rejections import (
    AllCardsDealtRejected,
    BidNotAllowedInDealBidPhaseRejected,
    BidCardSuitMismatchRejected,
    BidCardWrongRankRejected,
    BidCardsCountMismatchRejected,
    BidCountRejected,
    BidPriorityTooLowRejected,
    CardNotInHandRejected,
    DealNotCompleteRejected,
    DuplicateBidCardsRejected,
    InvalidPlayerIndexRejected,
    JokerBidCountRejected,
    JokerBidMustBePairRejected,
    JokerBidSuitRejected,
    MissingBidSuitRejected,
    MixedJokerPairRejected,
    NotJokerRejected,
    DealCardNotAllowedInDealBidPhaseRejected,
    ZeroBidValueRejected,
)
from .types import BidEvent, DealBidPhase

_BID_SUIT_ORDER: tuple[Suit, ...] = (
    Suit.SPADES,
    Suit.HEARTS,
    Suit.CLUBS,
    Suit.DIAMONDS,
)
MAX_BID_ACTION_HINTS: int = 10


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

    Returns Ok(new_state) on success, Rejected(reason) if not in DEALING phase
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

    # After the last card, keep phase as DEALING so the recipient can act.
    # Set all_dealt=True so Game can finalize after their action.
    all_dealt = new_cursor >= 100

    return Ok(state.model_copy(update={
        "phase": "DEALING",
        "deal_cursor": new_cursor,
        "deal_target": new_target,
        "players_hand": new_hands,
        "all_dealt": all_dealt,
    }))


def get_bid_action_hints(state: DealBidState, player: int) -> list[list[Card]]:
    """Compute accepted logical bid hints for one player.

    The returned options are advisory UI hints, so a non-empty result must
    match the same validation path used by real bid actions. Candidate card
    groups are generated from the player's hand as logical bid options, then
    filtered through reveal() against the full deal-bid state so current
    bid-priority and card-ownership rules are respected.
    """
    if state.phase != "DEALING" or player < 0 or player >= 4:
        return []

    result: list[list[Card]] = []
    for candidate in _get_bid_card_candidates(state.players_hand[player], state.trump_rank):
        event = _bid_event_from_cards(player, candidate)
        match reveal(state, event):
            case Ok():
                result.append(candidate)
            case Rejected():
                continue
    result.sort(key=lambda cards: _bid_hint_sort_key(cards, state.trump_rank))
    return result


def _bid_hint_sort_key(cards: list[Card], trump_rank: Rank) -> tuple[int, tuple[str, ...]]:
    return (bid_value(cards, trump_rank), tuple(sorted(card.id for card in cards)))


def _get_bid_card_candidates(hand: list[Card], trump_rank: Rank) -> list[list[Card]]:
    """Compute logical bid card groups from a player's hand.

    Returns list of bid options where each option is 1 card (single) or
    2 cards (pair). Singles are one canonical trump-rank card per suit.
    Pairs are two trump-rank cards of the same suit or two jokers of same type.
    Non-trump-rank cards and single jokers are excluded.

    Only includes logical options that have a non-zero bid_value:
    - Singles: one trump-rank card per suit
    - Pairs: two trump-rank cards of the same suit, or two jokers of same type
    """
    # Group trump-rank and joker cards
    suit_groups: dict[Suit, list[Card]] = {}
    small_jokers: list[Card] = []
    big_jokers: list[Card] = []

    for card in hand:
        if card.is_joker:
            if card.is_big_joker:
                big_jokers.append(card)
            else:
                small_jokers.append(card)
        elif card.rank == trump_rank:
            suit_groups.setdefault(card.suit, []).append(card)

    result: list[list[Card]] = []

    # Pairs: big joker, small joker, spades, hearts, clubs, diamonds.
    if len(big_jokers) >= 2:
        result.append(big_jokers[:2])
    if len(small_jokers) >= 2:
        result.append(small_jokers[:2])
    for suit in _BID_SUIT_ORDER:
        cards = suit_groups.get(suit, [])
        if len(cards) >= 2:
            result.append(cards[:2])

    # Singles: spades, hearts, clubs, diamonds.
    for suit in _BID_SUIT_ORDER:
        cards = suit_groups.get(suit, [])
        if len(cards) >= 1:
            result.append([cards[0]])

    return result


def _bid_event_from_cards(player: int, cards: list[Card]) -> BidEvent:
    card = cards[0]
    if card.is_joker:
        return BidEvent(
            player=player,
            cards=cards,
            kind="joker",
            suit=None,
            joker_type="big" if card.is_big_joker else "small",
            count=len(cards),
        )
    return BidEvent(
        player=player,
        cards=cards,
        kind="trump_rank",
        suit=card.suit,
        joker_type=None,
        count=len(cards),
    )


def reveal(state: DealBidState, event: BidEvent) -> Ok[DealBidState] | Rejected:
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

    # Precondition 4a: cards must be distinct physical cards (no duplicate IDs)
    if len(set(c.id for c in event.cards)) != len(event.cards):
        return DuplicateBidCardsRejected()

    # Precondition 2 & 3: validate card types
    if event.kind == "trump_rank":
        if event.suit is None:
            return MissingBidSuitRejected()
        # Each card must be trump_rank and same suit
        for card in event.cards:
            if card.rank != state.trump_rank:
                return BidCardWrongRankRejected(card.id, state.trump_rank)
            if card.suit != event.suit:
                return BidCardSuitMismatchRejected(card.suit, event.suit)
        if event.count not in (1, 2):
            return BidCountRejected(event.count)
        if len(event.cards) != event.count:
            return BidCardsCountMismatchRejected(len(event.cards), event.count)
    elif event.kind == "joker":
        if event.count != 2:
            return JokerBidMustBePairRejected()
        if len(event.cards) != 2:
            return JokerBidCountRejected(len(event.cards))
        for card in event.cards:
            if not card.is_joker:
                return NotJokerRejected(card.id)
        if event.cards[0].rank != event.cards[1].rank:
            return MixedJokerPairRejected()
        if event.suit is not None:
            return JokerBidSuitRejected()

    # Precondition 5: if bid_winner exists, new bid value must be strictly greater
    new_value = bid_value(event.cards, state.trump_rank)
    if new_value == 0:
        return ZeroBidValueRejected()

    if state.bid_winner is not None:
        current_value = bid_value(state.bid_winner.cards, state.trump_rank)
        if new_value <= current_value:
            return BidPriorityTooLowRejected()

    # Valid bid: update state
    new_bid_events = state.bid_events + [event]

    return Ok(state.model_copy(update={
        "bid_winner": event,
        "bid_events": new_bid_events,
    }))


def finalize_dealing(state: DealBidState) -> Ok[DealBidState] | Rejected:
    """After all cards are dealt and the last recipient has acted, finalize.

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
