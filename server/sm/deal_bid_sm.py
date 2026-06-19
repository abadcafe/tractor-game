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
from server.sm.result import Ok, Rejected, StateResult
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


def deal_next_card(state: DealBidState) -> StateResult[DealBidState]:
    """Deal the next card from the deck to the current target player.

    After dealing, if all 100 cards have been dealt, phase stays DEALING
    with all_dealt=True so the last card recipient can bid or skip.
    Call finalize_dealing() after their action to complete the phase.

    Returns Ok(new_state) on success, Rejected(reason) if not in DEALING phase
    or all cards have already been dealt.
    """
    if state.phase != "DEALING":
        return Rejected(f"发牌只能在发牌阶段进行，当前阶段：{state.phase}")

    cursor = state.deal_cursor
    if cursor >= 100:
        return Rejected("所有牌已发完")

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
    """Compute complete accepted bid hints for one player.

    The returned options are advisory UI hints, so a non-empty result must
    match the same validation path used by real bid actions. Candidate card
    groups are generated from the player's hand, then filtered through
    reveal() against the full deal-bid state so declarer-team and current
    bid-priority rules are respected.
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
    return result


def _get_bid_card_candidates(hand: list[Card], trump_rank: Rank) -> list[list[Card]]:
    """Compute raw bid card groups from a player's hand.

    Returns list of bid options where each option is 1 card (single) or
    2 cards (pair). Singles are individual trump-rank cards (one per suit).
    Pairs are two trump-rank cards of the same suit or two jokers of same type.
    Non-trump-rank cards and single jokers are excluded (single jokers have
    bid_value 0 and cannot be used to bid).

    Only includes options that have a non-zero bid_value:
    - Singles: trump-rank cards (one per suit group; not jokers)
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

    # Add pairs from suit groups (two trump-rank cards of same suit)
    for _suit, cards in suit_groups.items():
        if len(cards) >= 2:
            result.append(cards[:2])

    # Add joker pairs
    if len(big_jokers) >= 2:
        result.append(big_jokers[:2])
    if len(small_jokers) >= 2:
        result.append(small_jokers[:2])

    # Add singles (trump-rank cards, one per suit group)
    for _suit, cards in suit_groups.items():
        # Add first card as single (if no pair was added, or always add singles)
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


def reveal(state: DealBidState, event: BidEvent) -> StateResult[DealBidState]:
    """Process a reveal (bid) event from a player.

    Validates the bid per spec section 5 and updates the bid_winner
    if the new bid has strictly higher value.

    Returns Ok(new_state) on success, Rejected(reason) on invalid input.
    """
    # Precondition 0: player index must be valid
    if event.player < 0 or event.player >= 4:
        return Rejected(f"玩家索引无效：{event.player}")

    # Precondition 1: must be in DEALING phase
    if state.phase != "DEALING":
        return Rejected(f"叫牌只能在发牌阶段进行，当前阶段：{state.phase}")

    # Precondition 6: subsequent rounds - only declarer team can reveal
    if state.declarer_team is not None:
        player_team = get_team_index(event.player)
        if player_team != state.declarer_team:
            return Rejected("非庄家方不能叫牌")

    player_idx = event.player
    hand = state.players_hand[player_idx]

    # Precondition 4: cards must be in player's hand
    hand_ids = {c.id for c in hand}
    for card in event.cards:
        if card.id not in hand_ids:
            return Rejected(f"牌 {card.id} 不在手牌中")

    # Precondition 4a: cards must be distinct physical cards (no duplicate IDs)
    if len(set(c.id for c in event.cards)) != len(event.cards):
        return Rejected("牌张重复，不能使用同一张牌两次")

    # Precondition 2 & 3: validate card types
    if event.kind == "trump_rank":
        if event.suit is None:
            return Rejected("主牌叫牌必须指定花色")
        # Each card must be trump_rank and same suit
        for card in event.cards:
            if card.rank != state.trump_rank:
                return Rejected(f"牌 {card.id} 不是主牌等级 {state.trump_rank.value}")
            if card.suit != event.suit:
                return Rejected(f"牌花色 {card.suit.value} 与声明花色 {event.suit.value} 不一致")
        if event.count not in (1, 2):
            return Rejected(f"叫牌数量必须为1或2，实际 {event.count}")
        if len(event.cards) != event.count:
            return Rejected(f"牌张数量 {len(event.cards)} 与声明数量 {event.count} 不一致")
    elif event.kind == "joker":
        # Must be a pair of jokers (same type), single joker rejected
        if event.count != 2:
            return Rejected("王叫牌必须出对子")
        if len(event.cards) != 2:
            return Rejected(f"王叫牌必须出2张，实际 {len(event.cards)} 张")
        for card in event.cards:
            if not card.is_joker:
                return Rejected(f"牌 {card.id} 不是王")
        # Both must be same joker type
        if event.cards[0].rank != event.cards[1].rank:
            return Rejected("两种王不能配对")
        if event.suit is not None:
            return Rejected("王叫牌不能指定花色")

    # Precondition 5: if bid_winner exists, new bid value must be strictly greater
    new_value = bid_value(event.cards, state.trump_rank)
    if new_value == 0:
        return Rejected("叫牌无效：牌张价值为零")

    if state.bid_winner is not None:
        current_value = bid_value(state.bid_winner.cards, state.trump_rank)
        if new_value <= current_value:
            return Rejected("叫牌优先级不足")

    # Valid bid: update state
    new_bid_events = state.bid_events + [event]

    return Ok(state.model_copy(update={
        "bid_winner": event,
        "bid_events": new_bid_events,
    }))


def finalize_dealing(state: DealBidState) -> StateResult[DealBidState]:
    """After all cards are dealt and the last recipient has acted, finalize.

    Transitions phase to COMPLETE (if bid_winner exists) or NO_BID.
    Must only be called when all_dealt=True.

    Returns Ok(new_state) on success, Rejected(reason) if not ready.
    """
    if not state.all_dealt:
        return Rejected("还有牌未发完，不能结束发牌")
    new_phase: Literal["DEALING", "COMPLETE", "NO_BID"] = (
        "COMPLETE" if state.bid_winner is not None else "NO_BID"
    )
    return Ok(state.model_copy(update={"phase": new_phase}))
