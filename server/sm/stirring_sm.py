"""Stirring (炒地皮) state machine for Shengji/Tractor.

After the declarer is determined, other players can change the trump suit
by revealing pairs of trump-rank cards or jokers.

Rules:
- Only pairs are valid (singles cannot stir).
- Joker pair > pair_♠ > pair_♥ > pair_♣ > pair_♦.
- Joker pair always accepted (sets trump_suit=None).
- For non-trump-suit pairs: must have higher priority than current trump.
- For 空主 (trump_suit=None): any trump-rank pair is accepted.
- Declarer and team never change during stirring.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from server.sm.card_model import Card, Suit, Rank
from server.sm.comparator import bid_value
from server.sm.constants import next_player_ccw
from server.sm.result import Ok, Rejected, StateResult
from server.sm.types import StirAction


# ---- Priority Mapping ----

# Priority order for stirring: higher value = higher priority
# Matches bid_value's suit ordering: ♦=0, ♣=1, ♥=2, ♠=3, 小王=4, 大王=5
# For stirring, we compare pair values directly using bid_value.


# ---- Input / Output Models ----


class StirInput(BaseModel):
    """Input to create a stirring phase."""

    model_config = ConfigDict(frozen=True)

    trump_suit: Suit | None
    trump_rank: Rank
    declarer_player: int


class StirResult(BaseModel):
    """Output from a completed stirring phase."""

    model_config = ConfigDict(frozen=True)

    final_trump_suit: Suit | None
    stir_count: int


class StirringState(BaseModel):
    """Internal state of the stirring phase."""

    model_config = ConfigDict(frozen=True)

    phase: Literal["WAITING", "COMPLETE"]
    trump_suit: Suit | None
    trump_rank: Rank
    declarer_player: int
    current_player: int
    pass_set: frozenset[int]
    actions: tuple[StirAction, ...]
    last_stir_player: int | None = None
    current_priority: int = 0


# ---- Operations ----


def create_stirring(input: StirInput) -> StirringState:
    """Create initial stirring state.

    Starts with current_player = CCW_next(declarer_player).
    current_priority is set to bid_value of the current trump pair,
    or 0 for empty trump (so any pair can stir).
    """
    if input.trump_suit is not None:
        initial_priority = bid_value(
            _make_trump_pair(input.trump_suit, input.trump_rank),
            input.trump_rank,
        )
    else:
        initial_priority = 0

    return StirringState(
        phase="WAITING",
        trump_suit=input.trump_suit,
        trump_rank=input.trump_rank,
        declarer_player=input.declarer_player,
        current_player=next_player_ccw(input.declarer_player),
        pass_set=frozenset(),
        actions=(),
        current_priority=initial_priority,
    )


def pass_stir(state: StirringState, player: int) -> StateResult[StirringState]:
    """Player passes. Add to pass_set and advance current_player.

    If all 4 players have passed, phase becomes COMPLETE.
    Returns Rejected if player is not the current player.
    """
    if player != state.current_player:
        return Rejected("不是你的回合")

    new_pass_set = state.pass_set | {player}
    new_action = StirAction(player=player, kind="pass", new_suit=None)

    if len(new_pass_set) == 4:
        return Ok(StirringState(
            phase="COMPLETE",
            trump_suit=state.trump_suit,
            trump_rank=state.trump_rank,
            declarer_player=state.declarer_player,
            current_player=state.current_player,
            pass_set=new_pass_set,
            actions=state.actions + (new_action,),
            last_stir_player=state.last_stir_player,
            current_priority=state.current_priority,
        ))

    return Ok(StirringState(
        phase="WAITING",
        trump_suit=state.trump_suit,
        trump_rank=state.trump_rank,
        declarer_player=state.declarer_player,
        current_player=next_player_ccw(state.current_player),
        pass_set=new_pass_set,
        actions=state.actions + (new_action,),
        last_stir_player=state.last_stir_player,
        current_priority=state.current_priority,
    ))


def stir(
    state: StirringState, player: int, cards: list[Card]
) -> StateResult[StirringState]:
    """Attempt to stir (change trump suit) with a pair of cards.

    Validation:
    1. Must be current_player.
    2. Must be exactly 2 cards (pair).
    3. Must be a valid pair (same suit, or both same joker type).
    4. Cards must be trump rank or jokers.
    5. Priority must be higher than current trump suit priority.

    If valid: returns Ok(new_state) with updated trump_suit, reset pass_set.
    If invalid: returns Rejected(reason) with a specific reason.
    """
    # 1. Wrong player
    if player != state.current_player:
        return Rejected("不是你的回合")

    # 1b. Cannot stir one's own trump (prevents infinite stir loops)
    if state.last_stir_player == player:
        return Rejected("不能连续反主")

    # 2. Must be exactly 2 cards
    if len(cards) != 2:
        return Rejected("反主必须出对子")

    # 3. Must be a valid pair
    # Joker pair: both jokers of same type
    if cards[0].is_joker and cards[1].is_joker:
        if cards[0].rank != cards[1].rank:
            return Rejected("两种王不能配对")
        # Valid joker pair
        new_suit: Suit | None = None  # Joker pair → 无主
    else:
        # Non-joker: both must be trump rank, same suit
        if cards[0].is_joker or cards[1].is_joker:
            return Rejected("王和普通牌不能配对")
        if cards[0].rank != state.trump_rank or cards[1].rank != state.trump_rank:
            return Rejected("牌不是主牌等级")
        if cards[0].suit != cards[1].suit:
            return Rejected("对子必须同花色")
        new_suit = cards[0].suit

    # 4. Priority check (unified for all cases, including empty trump + joker)
    new_priority = bid_value(cards, state.trump_rank)
    if new_priority <= state.current_priority:
        return Rejected("优先级不足，不能反主")

    # 5. Valid stir: update state
    new_action = StirAction(player=player, kind="stir", new_suit=new_suit)
    return Ok(StirringState(
        phase="WAITING",
        trump_suit=new_suit,
        trump_rank=state.trump_rank,
        declarer_player=state.declarer_player,
        current_player=next_player_ccw(state.current_player),
        pass_set=frozenset(),
        actions=state.actions + (new_action,),
        last_stir_player=player,
        current_priority=new_priority,
    ))


def get_stir_result(state: StirringState) -> StirResult:
    """Extract the result from a completed stirring phase.

    Returns the final trump suit and the number of stir actions taken.
    """
    stir_count = sum(1 for a in state.actions if a.kind == "stir")
    return StirResult(final_trump_suit=state.trump_suit, stir_count=stir_count)


def _make_trump_pair(suit: Suit, rank: Rank) -> list[Card]:
    """Create a dummy pair of cards for priority comparison."""
    return [
        Card(
            id=f"dummy-1-{suit.value}-{rank.value}",
            suit=suit,
            rank=rank,
            is_joker=False,
            is_big_joker=False,
            points=0,
            deck=1,
        ),
        Card(
            id=f"dummy-2-{suit.value}-{rank.value}",
            suit=suit,
            rank=rank,
            is_joker=False,
            is_big_joker=False,
            points=0,
            deck=2,
        ),
    ]
