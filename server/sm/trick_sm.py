"""Trick (one-trick) state machine for Shengji/Tractor.

Manages one trick: leading player plays, then 3 followers in CCW order.
After all 4 play, determine winner and points.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .card_model import Card, Suit, Rank
from .comparator import effective_suit
from .constants import next_player_ccw, get_team_index
from .play_rules import (
    compare_plays,
    is_legal_follow,
    resolve_lead_throw,
)
from .result import Ok, Rejected, StateResult
from .types import CompletedTrick, CompletedTrickSlot, FailedThrow


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

    model_config = ConfigDict(frozen=True)

    phase: Literal["LEADING", "FOLLOWING", "RESOLVED"]
    lead_player: int
    slots: list[CompletedTrickSlot]
    played: int
    cur: int
    trump_suit: Suit | None
    trump_rank: Rank
    defender_points: int
    declarer_team: int
    hands: list[list[Card]]
    result: TrickResult | None
    failed_throw: FailedThrow | None


# ---- Public API ----


def create_trick(input: TrickInput) -> TrickState:
    """Create a new trick in LEADING phase."""
    return TrickState(
        phase="LEADING",
        lead_player=input.lead_player,
        slots=[CompletedTrickSlot(player=i, cards=[]) for i in range(4)],
        played=0,
        cur=input.lead_player,
        trump_suit=input.trump_suit,
        trump_rank=input.trump_rank,
        defender_points=input.defender_points,
        declarer_team=input.declarer_team,
        hands=[list(h) for h in input.hands],  # copy hands
        result=None,
        failed_throw=None,
    )


def play(state: TrickState, player: int, cards: list[Card]) -> StateResult[TrickState]:
    """Play cards for the current player.

    Validates:
    - player == cur (right player)
    - cards are in player's hand
    - following players follow suit if possible

    Returns Ok(new_state) on success, Rejected(reason) on invalid input.
    """
    # Validate it's this player's turn
    if player != state.cur:
        return Rejected(f"不是你的回合，当前是玩家 {state.cur} 的回合")

    # Validate phase is not already resolved
    if state.phase == "RESOLVED":
        return Rejected("该轮已结束")

    hand = state.hands[player]

    # Validate at least one card is being played
    if not cards:
        return Rejected("必须至少出一张牌")

    # Validate cards are in player's hand
    played_ids = {c.id for c in cards}
    hand_ids = {c.id for c in hand}
    if not played_ids.issubset(hand_ids):
        return Rejected("出的牌不在手牌中")

    actual_cards = list(cards)
    failed_throw: FailedThrow | None = None

    # Validate lead legality
    if state.phase == "LEADING":
        other_players_hands: list[list[Card]] = []
        for i in range(4):
            if i != player:
                other_players_hands.append(list(state.hands[i]))
        match resolve_lead_throw(
            hand,
            cards,
            state.trump_suit,
            state.trump_rank,
            other_players_hands,
        ):
            case Ok(value=resolution):
                actual_cards = list(resolution.played_cards)
            case Rejected(reason=reason):
                return Rejected(reason)

        attempted_ids = {c.id for c in cards}
        actual_ids = {c.id for c in actual_cards}
        if attempted_ids != actual_ids:
            failed_throw = FailedThrow(
                player=player,
                attempted_cards=list(cards),
                forced_cards=list(actual_cards),
            )

    # Validate follow-suit if following
    if state.phase == "FOLLOWING":
        lead_cards = state.slots[state.lead_player].cards
        if len(lead_cards) == 0:
            # Internal invariant: should never happen if play() is called correctly.
            # Kept as raise because it signals a code bug, not a race condition.
            raise ValueError("Lead cards must exist in FOLLOWING phase")
        if not is_legal_follow(hand, actual_cards, lead_cards, state.trump_suit, state.trump_rank):
            return Rejected(
                "必须跟牌"
            )

    # Build new state (immutable)
    new_phase = state.phase
    new_played = state.played + 1
    new_cur = next_player_ccw(player)

    # Update slots: copy and set player's slot
    new_slots = list(state.slots)
    new_slots[player] = CompletedTrickSlot(player=player, cards=list(actual_cards))

    # Remove cards from hand
    actual_played_ids = {c.id for c in actual_cards}
    new_hands = [list(h) for h in state.hands]
    new_hands[player] = [c for c in hand if c.id not in actual_played_ids]

    # Transition to FOLLOWING on first play
    if state.played == 0:
        new_phase = "FOLLOWING"

    new_state = TrickState(
        phase=new_phase,
        lead_player=state.lead_player,
        slots=new_slots,
        played=new_played,
        cur=new_cur,
        trump_suit=state.trump_suit,
        trump_rank=state.trump_rank,
        defender_points=state.defender_points,
        declarer_team=state.declarer_team,
        hands=new_hands,
        result=state.result,
        failed_throw=failed_throw,
    )

    # Resolve when all 4 have played
    if new_played == 4:
        return Ok(_resolve(new_state))

    return Ok(new_state)


def _trick_play_order(lead_player: int) -> list[int]:
    """Return the four players in the actual trick play order."""
    order = [lead_player]
    cur = lead_player
    for _ in range(3):
        cur = next_player_ccw(cur)
        order.append(cur)
    return order


def _resolve(state: TrickState) -> TrickState:
    """Resolve the trick: determine winner, count points, build result.

    Returns a new TrickState in RESOLVED phase (immutable).
    """
    slots_by_player: dict[int, CompletedTrickSlot] = {slot.player: slot for slot in state.slots}

    # Get lead cards for comparison
    lead_slot = slots_by_player.get(state.lead_player)
    if lead_slot is None:
        raise ValueError("Lead player's slot must exist at resolution")
    lead_cards = lead_slot.cards
    if len(lead_cards) == 0:
        raise ValueError("Lead cards must exist at resolution")
    lead_eff = effective_suit(lead_cards[0], state.trump_suit, state.trump_rank)

    # Find winner in real play order so equal-ranked plays keep the earlier winner.
    winner = state.lead_player
    best_cards = lead_slot.cards
    if len(best_cards) == 0:
        raise ValueError("Winner's cards must exist at resolution")

    for p in _trick_play_order(state.lead_player)[1:]:
        slot = slots_by_player.get(p)
        if slot is None:
            raise ValueError(f"Player {p}'s slot must exist at resolution")
        p_cards = slot.cards
        if len(p_cards) == 0:
            raise ValueError(f"Player {p}'s cards must exist at resolution")
        cmp = compare_plays(
            p_cards, best_cards,
            lead_eff,
            state.trump_suit, state.trump_rank,
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
        slots=list(state.slots),
        winner=winner,
        points=total_points,
    )

    result = TrickResult(
        winner=winner,
        points=total_points,
        updated_defender_points=updated_defender_points,
        completed_trick=completed,
    )

    return state.model_copy(update={"phase": "RESOLVED", "result": result})
