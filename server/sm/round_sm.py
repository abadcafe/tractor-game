"""Round state machine for 升级 (Shengji/Tractor).

Orchestrates one round by serially executing sub-state machines and
threading intermediate data: DealBid -> Stirring -> Exchange -> Trick x 25 -> Scoring.

Exposes sub-state-machine operations through its own API, delegating
to the sub-state machines while managing phase transitions.
"""

from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict

from server.sm.card_model import Card, Rank, Suit, create_decks
from server.sm.constants import (
    BOTTOM_CARD_COUNT,
    get_team_index,
)
from server.sm import deal_bid as db
from server.sm import stirring as stir_mod
from server.sm import exchange as exc
from server.sm import trick as trick_mod
from server.sm import scoring
from server.sm.types import BidEvent, CompletedTrick


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

    model_config = ConfigDict(frozen=False)

    phase: str  # "DEAL_BID" | "STIRRING" | "EXCHANGE" | "PLAYING" | "SCORING" | "COMPLETE"
    declarer_team: int | None
    declarer_player: int | None
    defender_team: int | None
    trump_suit: Suit | None
    trump_rank: Rank
    players_hand: list[list[Card]]
    bottom_cards: list[Card]
    defender_points: int
    trick_history: list[CompletedTrick]
    deal_bid_state: db.DealBidState | None
    stirring_state: stir_mod.StirringState | None
    exchange_state: exc.ExchangeState | None
    trick_state: trick_mod.TrickState | None
    current_lead_player: int | None
    result: scoring.RoundResult | None
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
        players_hand=[[], [], [], []],
        bottom_cards=bottom_cards,
        defender_points=0,
        trick_history=[],
        deal_bid_state=deal_bid_state,
        stirring_state=None,
        exchange_state=None,
        trick_state=None,
        current_lead_player=None,
        result=None,
        team0_level=input.team0_level,
        team1_level=input.team1_level,
        start_player=start_player,
        last_declarer_player=input.last_declarer_player,
    )


def deal_next_card(state: RoundState) -> RoundState:
    """Deal the next card during DEAL_BID phase.

    Delegates to deal_bid.deal_next_card. After dealing, syncs players_hand.
    If deal-bid is complete (COMPLETE or NO_BID), transitions to STIRRING.
    """
    if state.phase != "DEAL_BID":
        raise ValueError(
            f"Cannot deal_next_card in phase {state.phase}; expected DEAL_BID"
        )
    if state.deal_bid_state is None:
        raise ValueError("deal_bid_state is None")

    new_db = db.deal_next_card(state.deal_bid_state)
    new_state = state.model_copy(update={
        "deal_bid_state": new_db,
        "players_hand": [list(h) for h in new_db.players_hand],
    })

    # Check if deal-bid is complete
    if new_db.phase in ("COMPLETE", "NO_BID"):
        return _transition_to_stirring(new_state, new_db)

    return new_state


def reveal(state: RoundState, event: BidEvent) -> RoundState:
    """Reveal (bid) a card during DEAL_BID phase.

    Delegates to deal_bid.reveal.
    """
    if state.phase != "DEAL_BID":
        raise ValueError(
            f"Cannot reveal in phase {state.phase}; expected DEAL_BID"
        )
    if state.deal_bid_state is None:
        raise ValueError("deal_bid_state is None")

    new_db = db.reveal(state.deal_bid_state, event)
    return state.model_copy(update={"deal_bid_state": new_db})


def pass_stir(state: RoundState) -> RoundState:
    """Pass during STIRRING phase.

    Delegates to stirring.pass_stir. If stirring completes, transitions to EXCHANGE.
    """
    if state.phase != "STIRRING":
        raise ValueError(
            f"Cannot pass_stir in phase {state.phase}; expected STIRRING"
        )
    if state.stirring_state is None:
        raise ValueError("stirring_state is None")

    cur = state.stirring_state.current_player
    new_ss = stir_mod.pass_stir(state.stirring_state, cur)

    if new_ss.phase == "COMPLETE":
        new_state = state.model_copy(update={
            "stirring_state": new_ss,
            "trump_suit": new_ss.trump_suit,
        })
        return _transition_to_exchange(new_state)

    return state.model_copy(update={"stirring_state": new_ss})


def stir(state: RoundState, cards: list[Card]) -> RoundState:
    """Stir (change trump suit) during STIRRING phase.

    Validates cards are in the current player's hand, then delegates
    to stirring.stir.
    """
    if state.phase != "STIRRING":
        raise ValueError(
            f"Cannot stir in phase {state.phase}; expected STIRRING"
        )
    if state.stirring_state is None:
        raise ValueError("stirring_state is None")

    cur = state.stirring_state.current_player
    hand = state.players_hand[cur]

    # Validate cards are in player's hand
    hand_ids = {c.id for c in hand}
    for card in cards:
        if card.id not in hand_ids:
            raise ValueError(
                f"Card {card.id} not in hand of player {cur}"
            )

    new_ss = stir_mod.stir(state.stirring_state, cur, cards)
    new_state = state.model_copy(update={
        "stirring_state": new_ss,
        "trump_suit": new_ss.trump_suit,
    })

    return new_state


def discard(state: RoundState, cards: list[Card]) -> RoundState:
    """Discard bottom cards during EXCHANGE phase.

    Delegates to exchange.discard. If exchange completes, transitions to PLAYING.
    """
    if state.phase != "EXCHANGE":
        raise ValueError(
            f"Cannot discard in phase {state.phase}; expected EXCHANGE"
        )
    if state.exchange_state is None:
        raise ValueError("exchange_state is None")

    new_exc = exc.discard(state.exchange_state, cards)

    if new_exc.phase == "COMPLETE" and new_exc.result is not None:
        new_hand = new_exc.result.new_hand
        new_bottom = new_exc.result.new_bottom_cards
        declarer = new_exc.declarer_player

        new_hands = [list(h) for h in state.players_hand]
        new_hands[declarer] = list(new_hand)

        new_state = state.model_copy(update={
            "exchange_state": new_exc,
            "players_hand": new_hands,
            "bottom_cards": new_bottom,
        })
        return _transition_to_playing(new_state)

    return state.model_copy(update={"exchange_state": new_exc})


def play(state: RoundState, cards: list[Card]) -> RoundState:
    """Play cards during PLAYING phase.

    Delegates to trick.play. When trick resolves, records result and
    starts next trick. After 25 tricks, transitions to SCORING -> COMPLETE.
    """
    if state.phase != "PLAYING":
        raise ValueError(
            f"Cannot play in phase {state.phase}; expected PLAYING"
        )
    if state.trick_state is None:
        raise ValueError("trick_state is None")

    cur = state.trick_state.cur
    new_trick = trick_mod.play(state.trick_state, cur, cards)

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

        trick_count = len(new_state.trick_history)
        if trick_count >= 25:
            return _transition_to_scoring(new_state)

        # Start next trick: winner leads
        return _start_next_trick(new_state, new_trick.result.winner)

    # Trick not yet resolved (still in progress)
    return state.model_copy(update={
        "trick_state": new_trick,
        "players_hand": new_hands,
    })


def is_round_complete(state: RoundState) -> bool:
    """Return True if the round is complete."""
    return state.phase == "COMPLETE"


def get_round_result(state: RoundState) -> scoring.RoundResult | None:
    """Return the round result if available."""
    return state.result


# ---- Internal Helpers ----


def _transition_to_stirring(state: RoundState, deal_bid: db.DealBidState) -> RoundState:
    """Transition from DEAL_BID to STIRRING."""
    declarer_team = state.declarer_team
    declarer_player: int | None = None
    trump_suit: Suit | None = None
    defender_team: int | None = None

    if deal_bid.phase == "COMPLETE" and deal_bid.bid_winner is not None:
        winner = deal_bid.bid_winner.player
        winner_team = get_team_index(winner)

        if declarer_team is None:
            # Case A: First round with winner
            declarer_team = winner_team
            declarer_player = winner
        else:
            # Case B: Subsequent round with winner
            # Validate winner is on declarer team
            if winner_team != declarer_team:
                # Invalid: ignore and treat as no bid
                declarer_player = state.last_declarer_player
                trump_suit = None
            else:
                declarer_player = winner

        if trump_suit is None:
            trump_suit = deal_bid.bid_winner.suit if declarer_player == winner else None

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

    # Update players_hand from deal_bid
    new_hands = [list(h) for h in deal_bid.players_hand]

    # Create stirring state
    stirring_input = stir_mod.StirInput(
        trump_suit=trump_suit,
        trump_rank=state.trump_rank,
        declarer_player=declarer_player,
    )
    stirring_state = stir_mod.create_stirring(stirring_input)

    return state.model_copy(update={
        "phase": "STIRRING",
        "declarer_team": declarer_team,
        "declarer_player": declarer_player,
        "defender_team": defender_team,
        "trump_suit": trump_suit,
        "players_hand": new_hands,
        "stirring_state": stirring_state,
        "deal_bid_state": deal_bid,
    })


def _transition_to_exchange(state: RoundState) -> RoundState:
    """Transition from STIRRING to EXCHANGE."""
    declarer_player = state.declarer_player
    assert declarer_player is not None

    declarer_hand = state.players_hand[declarer_player]
    exchange_input = exc.ExchangeInput(
        declarer_player=declarer_player,
        bottom_cards=state.bottom_cards,
        declarer_hand=declarer_hand,
    )
    exchange_state = exc.create_exchange(exchange_input)

    return state.model_copy(update={
        "phase": "EXCHANGE",
        "exchange_state": exchange_state,
    })


def _transition_to_playing(state: RoundState) -> RoundState:
    """Transition from EXCHANGE to PLAYING. Create first trick."""
    declarer_player = state.declarer_player
    assert declarer_player is not None
    declarer_team = state.declarer_team
    assert declarer_team is not None

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
        "current_lead_player": lead_player,
    })


def _transition_to_scoring(state: RoundState) -> RoundState:
    """Transition from PLAYING to SCORING -> COMPLETE."""
    declarer_team = state.declarer_team
    declarer_player = state.declarer_player
    assert declarer_team is not None
    assert declarer_player is not None

    if not state.trick_history:
        # Should not happen after 25 tricks, but guard anyway
        return state.model_copy(update={"phase": "SCORING"})

    last_trick = state.trick_history[-1]

    result = scoring.calculate_score(
        defender_points=state.defender_points,
        bottom_cards=state.bottom_cards,
        last_trick=last_trick,
        declarer_team=declarer_team,
        declarer_player=declarer_player,
        team0_level=state.team0_level,
        team1_level=state.team1_level,
    )

    return state.model_copy(update={
        "phase": "COMPLETE",
        "result": result,
    })
