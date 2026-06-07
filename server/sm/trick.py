"""Trick (one-trick) state machine for Shengji/Tractor.

Manages one trick: leading player plays, then 3 followers in CCW order.
After all 4 play, determine winner and points.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from server.sm.card_model import Card, Suit, Rank
from server.sm.comparator import compare_plays, effective_suit
from server.sm.constants import next_player_ccw, get_team_index
from server.sm.play_rules import infer_play_type, get_legal_plays
from server.sm.types import PlayAction, PlayType, CompletedTrick, CompletedTrickSlot


# ---- Models ----


class TrickInput(BaseModel):
    """Input for creating a new trick."""

    model_config = ConfigDict(frozen=True)

    lead_player: int
    hands: list[list[Card]]
    trump_suit: Suit | None
    trump_rank: Rank
    defender_points: int
    declarer_team: int


class TrickResult(BaseModel):
    """Result of a completed trick."""

    model_config = ConfigDict(frozen=True)

    winner: int
    points: int
    updated_defender_points: int
    completed_trick: CompletedTrick


class TrickState(BaseModel):
    """State of a trick in progress."""

    model_config = ConfigDict(frozen=False)

    phase: Literal["LEADING", "FOLLOWING", "RESOLVED"]
    lead_player: int
    lead_type: PlayType | None
    slots: list[CompletedTrickSlot]
    played: int
    cur: int
    trump_suit: Suit | None
    trump_rank: Rank
    defender_points: int
    declarer_team: int
    hands: list[list[Card]]
    result: TrickResult | None


# ---- Public API ----


def create_trick(input: TrickInput) -> TrickState:
    """Create a new trick in LEADING phase."""
    return TrickState(
        phase="LEADING",
        lead_player=input.lead_player,
        lead_type=None,
        slots=[CompletedTrickSlot(player=i, cards=[]) for i in range(4)],
        played=0,
        cur=input.lead_player,
        trump_suit=input.trump_suit,
        trump_rank=input.trump_rank,
        defender_points=input.defender_points,
        declarer_team=input.declarer_team,
        hands=[list(h) for h in input.hands],  # copy hands
        result=None,
    )


def play(state: TrickState, player: int, cards: list[Card]) -> TrickState:
    """Play cards for the current player.

    Validates:
    - player == cur (right player)
    - cards are in player's hand
    - following players follow suit if possible

    Returns a new TrickState (immutable state machine pattern).
    """
    # Validate it's this player's turn
    if player != state.cur:
        raise ValueError(f"Not player {player}'s turn; expected player {state.cur}")

    # Validate phase is not already resolved
    if state.phase == "RESOLVED":
        raise ValueError("Trick is already resolved")

    # Validate at least one card is being played
    if not cards:
        raise ValueError("Must play at least one card")

    # Validate cards are in player's hand
    hand = state.hands[player]
    played_ids = {c.id for c in cards}
    hand_ids = {c.id for c in hand}
    if not played_ids.issubset(hand_ids):
        raise ValueError("Cards not in player's hand")

    # Validate follow-suit if following
    if state.phase == "FOLLOWING":
        lead_cards = state.slots[state.lead_player].cards
        if len(lead_cards) == 0:
            raise ValueError("Lead cards must exist in FOLLOWING phase")
        assert state.lead_type is not None, "lead_type must be set in FOLLOWING phase"
        lead_action = PlayAction(type=state.lead_type, cards=lead_cards)
        legal_plays = get_legal_plays(
            hand=hand,
            is_leading=False,
            lead_action=lead_action,
            trump_suit=state.trump_suit,
            trump_rank=state.trump_rank,
        )
        # Check that the played cards match one of the legal plays
        played_card_ids = frozenset(c.id for c in cards)
        legal = False
        for lp in legal_plays:
            if frozenset(c.id for c in lp.cards) == played_card_ids:
                legal = True
                break
        if not legal:
            raise ValueError("Play does not follow suit rules")

    # Build new state (immutable)
    new_phase = state.phase
    new_lead_type = state.lead_type
    new_played = state.played + 1
    new_cur = next_player_ccw(player)

    # Update slots: copy and set player's slot
    new_slots = list(state.slots)
    new_slots[player] = CompletedTrickSlot(player=player, cards=list(cards))

    # Remove cards from hand
    new_hands = [list(h) for h in state.hands]
    new_hands[player] = [c for c in hand if c.id not in played_ids]

    # Determine lead_type on first play
    if state.played == 0:
        new_lead_type = infer_play_type(
            cards, state.trump_suit, state.trump_rank
        )
        new_phase = "FOLLOWING"

    new_state = TrickState(
        phase=new_phase,
        lead_player=state.lead_player,
        lead_type=new_lead_type,
        slots=new_slots,
        played=new_played,
        cur=new_cur,
        trump_suit=state.trump_suit,
        trump_rank=state.trump_rank,
        defender_points=state.defender_points,
        declarer_team=state.declarer_team,
        hands=new_hands,
        result=state.result,
    )

    # Resolve when all 4 have played
    if new_played == 4:
        _resolve(new_state)

    return new_state


def _resolve(state: TrickState) -> None:
    """Resolve the trick: determine winner, count points, build result.

    Mutates state in-place to set phase and result.
    """
    # Get lead suit for comparison
    lead_slot = state.slots[state.lead_player]
    lead_cards = lead_slot.cards
    if len(lead_cards) == 0:
        raise ValueError("Lead cards must exist at resolution")
    lead_suit_raw = effective_suit(lead_cards[0], state.trump_suit, state.trump_rank)
    lead_suit_obj: Suit | None = None if not isinstance(lead_suit_raw, Suit) else lead_suit_raw

    # Find winner by comparing each play against current best
    winner = state.lead_player
    best_cards = state.slots[winner].cards
    if len(best_cards) == 0:
        raise ValueError("Winner's cards must exist at resolution")

    for slot in state.slots:
        p = slot.player
        if p == winner:
            continue
        p_cards = slot.cards
        if len(p_cards) == 0:
            raise ValueError(f"Player {p}'s cards must exist at resolution")
        cmp = compare_plays(
            p_cards, best_cards,
            state.trump_suit, state.trump_rank,
            lead_suit_obj,
        )
        if cmp > 0:
            winner = p
            best_cards = p_cards

    # Count points from all played cards
    total_points = 0
    for slot in state.slots:
        for c in slot.cards:
            total_points += c.points

    # Update defender points
    winner_team = get_team_index(winner)
    defender_team = 1 - state.declarer_team
    updated_defender_points = state.defender_points
    if winner_team == defender_team:
        updated_defender_points = state.defender_points + total_points

    # Build CompletedTrick
    completed = CompletedTrick(
        lead_player=state.lead_player,
        lead_type=state.lead_type,  # type: ignore[arg-type]
        slots=list(state.slots),
        winner=winner,
        points=total_points,
    )

    state.result = TrickResult(
        winner=winner,
        points=total_points,
        updated_defender_points=updated_defender_points,
        completed_trick=completed,
    )
    state.phase = "RESOLVED"
