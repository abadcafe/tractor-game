"""Round state machine for 升级 (Shengji/Tractor).

Orchestrates one round by serially executing sub-state machines and
threading intermediate data: DealBid -> Stirring (with embedded Exchange) -> Trick x 25 -> Scoring.

Exposes sub-state-machine operations through its own API, delegating
to the sub-state machines while managing phase transitions.
"""

from __future__ import annotations

import random

from typing import Literal

from pydantic import BaseModel, ConfigDict

from server.sm.card_model import Card, Rank, Suit, create_decks
from server.sm.constants import (
    BOTTOM_CARD_COUNT,
    get_team_index,
)
from server.sm import deal_bid_sm as db
from server.sm import stirring_sm as stir_mod
from server.sm import trick_sm as trick_mod
from server.sm import scoring
from server.sm.result import Ok, Rejected, StateResult
from server.sm.types import BidEvent, CompletedTrick
from server.sm.scoring import RoundResult


# ---- Data Models ----


class RoundInput(BaseModel):
    """Input for creating a new round."""

    model_config = ConfigDict(frozen=True)

    declarer_team: int | None
    trump_rank: Rank
    last_declarer_player: int | None
    team0_level: Rank
    team1_level: Rank


class RoundState(BaseModel):
    """Internal state of the round state machine."""

    model_config = ConfigDict(frozen=True)

    phase: Literal["DEAL_BID", "STIRRING", "PLAYING", "SCORING", "WAITING"]
    declarer_team: int | None
    declarer_player: int | None
    defender_team: int | None
    trump_suit: Suit | None
    trump_rank: Rank
    bid_winner: BidEvent | None
    players_hand: list[list[Card]]
    bottom_cards: list[Card]
    defender_points: int
    trick_history: list[CompletedTrick]
    deal_bid_state: db.DealBidState | None
    stirring_state: stir_mod.StirringState | None
    trick_state: trick_mod.TrickState | None
    result: RoundResult | None
    team0_level: Rank
    team1_level: Rank
    start_player: int
    last_declarer_player: int | None


# ---- State Machine Operations ----


def create_round(input: RoundInput) -> RoundState:
    """Create a new round with shuffled deck, split into 8 bottom + 100 deal deck."""
    decks = create_decks()
    random.shuffle(decks)

    bottom_cards = decks[:BOTTOM_CARD_COUNT]
    deck = decks[BOTTOM_CARD_COUNT:]

    start_player = 0

    deal_bid_state = db.create_deal_bid(db.DealBidInput(
        deck=deck,
        declarer_team=input.declarer_team,
        trump_rank=input.trump_rank,
        start_player=start_player,
    ))

    return RoundState(
        phase="DEAL_BID",
        declarer_team=input.declarer_team,
        declarer_player=None,
        defender_team=None,
        trump_suit=None,
        trump_rank=input.trump_rank,
        bid_winner=None,
        players_hand=[[], [], [], []],
        bottom_cards=bottom_cards,
        defender_points=0,
        trick_history=[],
        deal_bid_state=deal_bid_state,
        stirring_state=None,
        trick_state=None,
        result=None,
        team0_level=input.team0_level,
        team1_level=input.team1_level,
        start_player=start_player,
        last_declarer_player=input.last_declarer_player,
    )


def deal_next_card(state: RoundState) -> StateResult[RoundState]:
    """Deal the next card during DEAL_BID phase.

    Delegates to deal_bid.deal_next_card. After dealing, syncs players_hand.
    If deal-bid is complete (COMPLETE or NO_BID), transitions to STIRRING.

    Returns Ok(new_state) on success, Rejected(reason) on invalid state.
    """
    if state.phase != "DEAL_BID":
        return Rejected(
            f"发牌只能在发牌阶段进行，当前阶段：{state.phase}"
        )
    if state.deal_bid_state is None:
        return Rejected("发牌状态异常")

    match db.deal_next_card(state.deal_bid_state):
        case Ok(value=new_db):
            new_state = state.model_copy(update={
                "deal_bid_state": new_db,
                "bid_winner": new_db.bid_winner,
                "players_hand": [list(h) for h in new_db.players_hand],
            })

            # Check if deal-bid is complete
            if new_db.phase in ("COMPLETE", "NO_BID"):
                return Ok(_transition_to_stirring(new_state, new_db))

            return Ok(new_state)
        case Rejected(reason=reason):
            return Rejected(reason)


def reveal(state: RoundState, event: BidEvent) -> StateResult[RoundState]:
    """Reveal (bid) a card during DEAL_BID phase.

    Delegates to deal_bid.reveal.

    Returns Ok(new_state) on success, Rejected(reason) on invalid input.
    """
    if state.phase != "DEAL_BID":
        return Rejected(
            f"叫牌只能在发牌阶段进行，当前阶段：{state.phase}"
        )
    if state.deal_bid_state is None:
        return Rejected("叫牌状态异常")

    match db.reveal(state.deal_bid_state, event):
        case Ok(value=new_db):
            return Ok(state.model_copy(update={
                "deal_bid_state": new_db,
                "bid_winner": new_db.bid_winner,
            }))
        case Rejected(reason=reason):
            return Rejected(reason)


def finalize_deal_bid(state: RoundState) -> StateResult[RoundState]:
    """Finalize deal-bid after all cards dealt and last recipient has acted.

    Delegates to deal_bid.finalize_dealing, then transitions to STIRRING.

    Returns Ok(new_state) on success, Rejected(reason) on invalid state.
    """
    if state.phase != "DEAL_BID":
        return Rejected(
            f"只能在发牌阶段结束发牌，当前阶段：{state.phase}"
        )
    if state.deal_bid_state is None:
        return Rejected("发牌状态异常")

    match db.finalize_dealing(state.deal_bid_state):
        case Ok(value=new_db):
            return Ok(_transition_to_stirring(
                state.model_copy(update={
                    "deal_bid_state": new_db,
                    "bid_winner": new_db.bid_winner,
                    "players_hand": [list(h) for h in new_db.players_hand],
                }),
                new_db,
            ))
        case Rejected(reason=reason):
            return Rejected(reason)


def pass_stir(state: RoundState, player_index: int) -> StateResult[RoundState]:
    """Pass during STIRRING phase.

    Delegates to stirring.pass_stir. If stirring completes, transitions to PLAYING.

    Returns Ok(new_state) on success, Rejected(reason) on invalid input.
    """
    if state.phase != "STIRRING":
        return Rejected(
            f"不能在 {state.phase} 阶段跳过反主"
        )
    if state.stirring_state is None:
        return Rejected("反主状态异常")

    cur = state.stirring_state.current_player
    if player_index != cur:
        return Rejected(
            f"不是你的回合，当前是玩家 {cur} 的回合"
        )
    match stir_mod.pass_stir(state.stirring_state, cur):
        case Ok(value=new_ss):
            if new_ss.phase == "COMPLETE":
                # Stirring complete: sync hands/bottom from StirResult,
                # then transition directly to PLAYING
                new_state = state.model_copy(update={
                    "stirring_state": new_ss,
                    "trump_suit": new_ss.trump_suit,
                    "players_hand": [list(h) for h in new_ss.players_hand],
                    "bottom_cards": list(new_ss.bottom_cards),
                })
                return Ok(_transition_to_playing(new_state))
            return Ok(state.model_copy(update={
                "stirring_state": new_ss,
                "trump_suit": new_ss.trump_suit,
            }))
        case Rejected(reason=reason):
            return Rejected(reason)


def stir(state: RoundState, player_index: int, cards: list[Card]) -> StateResult[RoundState]:
    """Stir (change trump suit) during STIRRING phase.

    Validates cards are in the current player's hand, then delegates
    to stirring.stir.

    Returns Ok(new_state) on success, Rejected(reason) on invalid input.
    """
    if state.phase != "STIRRING":
        return Rejected(
            f"不能在 {state.phase} 阶段反主"
        )
    if state.stirring_state is None:
        return Rejected("反主状态异常")

    cur = state.stirring_state.current_player
    if player_index != cur:
        return Rejected(
            f"不是你的回合，当前是玩家 {cur} 的回合"
        )
    hand = state.players_hand[cur]

    # Validate cards are in player's hand
    hand_ids = {c.id for c in hand}
    for card in cards:
        if card.id not in hand_ids:
            return Rejected(
                f"牌 {card.id} 不在玩家 {cur} 的手牌中"
            )

    match stir_mod.stir(state.stirring_state, cur, cards):
        case Ok(value=new_ss):
            new_state = state.model_copy(update={
                "stirring_state": new_ss,
                "trump_suit": new_ss.trump_suit,
                "bid_winner": _bid_event_from_stir_cards(cur, cards),
            })
            return Ok(new_state)
        case Rejected(reason=reason):
            return Rejected(reason)


def stir_discard(
    state: RoundState, player_index: int, cards: list[Card]
) -> StateResult[RoundState]:
    """Discard bottom cards during STIRRING EXCHANGING sub-phase.

    The player who just established/changed the trump must pick up bottom
    cards and discard the same number back. Delegates to stirring.stir_discard.
    After successful discard, syncs hands and bottom_cards.

    Returns Ok(new_state) on success, Rejected(reason) on invalid input.
    """
    if state.phase != "STIRRING":
        return Rejected(
            f"不能在 {state.phase} 阶段换底牌"
        )
    if state.stirring_state is None:
        return Rejected("反主状态异常")

    match stir_mod.stir_discard(state.stirring_state, player_index, cards):
        case Ok(value=new_ss):
            new_state = state.model_copy(update={
                "stirring_state": new_ss,
                "trump_suit": new_ss.trump_suit,
                "players_hand": [list(h) for h in new_ss.players_hand],
                "bottom_cards": list(new_ss.bottom_cards),
            })
            if new_ss.phase == "COMPLETE":
                return Ok(_transition_to_playing(new_state))
            return Ok(new_state)
        case Rejected(reason=reason):
            return Rejected(reason)


def play(state: RoundState, player_index: int, cards: list[Card]) -> StateResult[RoundState]:
    """Play cards during PLAYING phase.

    Validates that player_index matches the current player, then delegates
    leading and following legality to trick.play.
    When trick resolves, records result and starts next trick.
    After 25 tricks, transitions to SCORING -> COMPLETE.

    Returns Ok(new_state) on success, Rejected(reason) on invalid input.
    """
    if state.phase != "PLAYING":
        return Rejected(
            f"不能在 {state.phase} 阶段出牌"
        )
    if state.trick_state is None:
        return Rejected("出牌状态异常")

    cur = state.trick_state.cur
    if player_index != cur:
        return Rejected(
            f"不是你的回合，当前是玩家 {cur} 的回合"
        )

    match trick_mod.play(state.trick_state, cur, cards):
        case Ok(value=new_trick):
            pass  # proceed below
        case Rejected(reason=reason):
            return Rejected(reason)

    # Sync hands from trick to round state
    new_hands = [list(h) for h in new_trick.hands]

    if new_trick.phase == "RESOLVED" and new_trick.result is not None:
        # Record the completed trick
        completed_tricks = list(state.trick_history) + [
            new_trick.result.completed_trick
        ]
        new_state = state.model_copy(update={
            "trick_state": new_trick,
            "players_hand": new_hands,
            "trick_history": completed_tricks,
            "defender_points": new_trick.result.updated_defender_points,
        })

        # If all hands are empty, round is over (can happen before 25 tricks
        # when players play pairs/tractors).
        if all(len(h) == 0 for h in new_hands):
            return Ok(_transition_to_scoring(new_state))

        trick_count = len(new_state.trick_history)
        if trick_count >= 25:
            return Ok(_transition_to_scoring(new_state))

        # Start next trick: winner leads
        return Ok(_start_next_trick(new_state, new_trick.result.winner))

    # Trick not yet resolved (still in progress)
    return Ok(state.model_copy(update={
        "trick_state": new_trick,
        "players_hand": new_hands,
    }))


def is_round_complete(state: RoundState) -> bool:
    """Return True if the round is complete."""
    return state.phase == "WAITING"


def get_round_result(state: RoundState) -> RoundResult | None:
    """Return the round result if available."""
    return state.result


# ---- Internal Helpers ----


def _transition_to_stirring(state: RoundState, deal_bid: db.DealBidState) -> RoundState:
    """Transition from DEAL_BID to STIRRING."""
    declarer_team = state.declarer_team
    declarer_player: int | None = None
    trump_suit: Suit | None = None
    bid_winner: BidEvent | None = None
    defender_team: int | None = None
    initial_bid_cards: list[Card] = []

    if deal_bid.phase == "COMPLETE" and deal_bid.bid_winner is not None:
        winner = deal_bid.bid_winner.player
        winner_team = get_team_index(winner)

        valid_winner = False
        if declarer_team is None:
            # Case A: First round with winner
            declarer_team = winner_team
            declarer_player = winner
            valid_winner = True
        else:
            # Case B: Subsequent round with winner
            # Validate winner is on declarer team
            if winner_team == declarer_team:
                declarer_player = winner
                valid_winner = True
            else:
                # Invalid: ignore and treat as no bid
                declarer_player = state.last_declarer_player if state.last_declarer_player is not None else state.start_player

        trump_suit = deal_bid.bid_winner.suit if valid_winner else None
        initial_bid_cards = list(deal_bid.bid_winner.cards) if valid_winner else []
        bid_winner = deal_bid.bid_winner if valid_winner else None
        defender_team = 1 - declarer_team

    elif deal_bid.phase == "NO_BID":
        # Case C: No bid (空主)
        trump_suit = None
        if declarer_team is None:
            # First round: declarer_player = start_player
            declarer_player = state.start_player
            declarer_team = get_team_index(declarer_player)
        else:
            # Subsequent round: declarer_player = last_declarer_player
            declarer_player = state.last_declarer_player if state.last_declarer_player is not None else state.start_player
        defender_team = 1 - declarer_team
    else:
        raise ValueError(
            f"Unexpected deal_bid phase: {deal_bid.phase}"
        )

    # At this point declarer_player and declarer_team are guaranteed non-None
    assert declarer_player is not None
    assert declarer_team is not None

    # Update players_hand from deal_bid
    new_hands = [list(h) for h in deal_bid.players_hand]

    # Create stirring state (includes initial exchange for declarer)
    stirring_input = stir_mod.StirInput(
        trump_suit=trump_suit,
        trump_rank=state.trump_rank,
        initial_bid_cards=initial_bid_cards,
        declarer_player=declarer_player,
        bottom_cards=list(state.bottom_cards),
        players_hand=new_hands,
    )
    stirring_state = stir_mod.create_stirring(stirring_input)

    # Sync hands and bottom cards from initial exchange state
    # (exchange state is created with declarer's hand + bottom cards picked up,
    # but no discard yet, so hands are unchanged until stir_discard is called)
    return state.model_copy(update={
        "phase": "STIRRING",
        "declarer_team": declarer_team,
        "declarer_player": declarer_player,
        "defender_team": defender_team,
        "trump_suit": trump_suit,
        "bid_winner": bid_winner,
        "players_hand": [list(h) for h in stirring_state.players_hand],
        "stirring_state": stirring_state,
        "deal_bid_state": deal_bid,
    })


def _bid_event_from_stir_cards(player: int, cards: list[Card]) -> BidEvent:
    """Convert validated stir cards into the current public bid winner."""
    assert len(cards) > 0
    first = cards[0]
    if first.is_joker:
        return BidEvent(
            player=player,
            cards=list(cards),
            kind="joker",
            suit=None,
            joker_type="big" if first.is_big_joker else "small",
            count=len(cards),
        )
    return BidEvent(
        player=player,
        cards=list(cards),
        kind="trump_rank",
        suit=first.suit,
        joker_type=None,
        count=len(cards),
    )


def _transition_to_playing(state: RoundState) -> RoundState:
    """Transition from STIRRING COMPLETE to PLAYING. Create first trick."""
    declarer_player = state.declarer_player
    assert declarer_player is not None

    return _start_next_trick(state, declarer_player)


def _start_next_trick(state: RoundState, lead_player: int) -> RoundState:
    """Start a new trick with the given lead player."""
    declarer_team = state.declarer_team
    assert declarer_team is not None

    trick_input = trick_mod.TrickInput(
        lead_player=lead_player,
        hands=[list(h) for h in state.players_hand],
        trump_suit=state.trump_suit,
        trump_rank=state.trump_rank,
        defender_points=state.defender_points,
        declarer_team=declarer_team,
    )
    trick_state = trick_mod.create_trick(trick_input)

    return state.model_copy(update={
        "phase": "PLAYING",
        "trick_state": trick_state,
    })


def _transition_to_scoring(state: RoundState) -> RoundState:
    """Transition from PLAYING to SCORING -> COMPLETE."""
    declarer_team = state.declarer_team
    declarer_player = state.declarer_player
    assert declarer_team is not None
    assert declarer_player is not None

    assert state.trick_history, "trick_history must be non-empty after 25 tricks"

    last_trick = state.trick_history[-1]

    result = scoring.calculate_score(
        defender_points=state.defender_points,
        bottom_cards=state.bottom_cards,
        last_trick=last_trick,
        declarer_team=declarer_team,
        declarer_player=declarer_player,
        team0_level=state.team0_level,
        team1_level=state.team1_level,
        trump_suit=state.trump_suit,
        trump_rank=state.trump_rank,
    )

    return state.model_copy(update={
        "phase": "WAITING",
        "result": result,
    })
