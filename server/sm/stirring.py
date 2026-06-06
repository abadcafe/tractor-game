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

from pydantic import BaseModel, ConfigDict

from server.sm.card_model import Card, Suit, Rank
from server.sm.comparator import bid_value
from server.sm.constants import next_player_ccw
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

    phase: str  # "WAITING" | "COMPLETE"
    trump_suit: Suit | None
    trump_rank: Rank
    declarer_player: int
    current_player: int
    pass_set: frozenset[int]
    actions: tuple[StirAction, ...]


# ---- Operations ----


def create_stirring(input: StirInput) -> StirringState:
    """Create initial stirring state.

    Starts with current_player = CCW_next(declarer_player).
    """
    return StirringState(
        phase="WAITING",
        trump_suit=input.trump_suit,
        trump_rank=input.trump_rank,
        declarer_player=input.declarer_player,
        current_player=next_player_ccw(input.declarer_player),
        pass_set=frozenset(),
        actions=(),
    )


def pass_stir(state: StirringState, player: int) -> StirringState:
    """Player passes. Add to pass_set and advance current_player.

    If all 4 players have passed, phase becomes COMPLETE.
    """
    new_pass_set = state.pass_set | {player}
    new_action = StirAction(player=player, kind="pass", new_suit=None)

    if len(new_pass_set) == 4:
        return StirringState(
            phase="COMPLETE",
            trump_suit=state.trump_suit,
            trump_rank=state.trump_rank,
            declarer_player=state.declarer_player,
            current_player=state.current_player,
            pass_set=new_pass_set,
            actions=state.actions + (new_action,),
        )

    return StirringState(
        phase="WAITING",
        trump_suit=state.trump_suit,
        trump_rank=state.trump_rank,
        declarer_player=state.declarer_player,
        current_player=next_player_ccw(state.current_player),
        pass_set=new_pass_set,
        actions=state.actions + (new_action,),
    )


def _pair_priority(cards: list[Card], trump_rank: Rank) -> int:
    """Calculate the priority value for a pair of cards.

    Uses bid_value from comparator for priority comparison.
    Pair values: ♦=200, ♣=201, ♥=202, ♠=203, small_joker=204, big_joker=205.
    """
    return bid_value(cards, trump_rank)


def stir(
    state: StirringState, player: int, cards: list[Card]
) -> StirringState:
    """Attempt to stir (change trump suit) with a pair of cards.

    Validation:
    1. Must be current_player.
    2. Must be exactly 2 cards (pair).
    3. Must be a valid pair (same suit, or both same joker type).
    4. Cards must be trump rank or jokers.
    5. Priority must be higher than current trump suit priority.

    If valid: update trump_suit, reset pass_set, advance current_player.
    If invalid: state unchanged.
    """
    # 1. Wrong player
    if player != state.current_player:
        return state

    # 2. Must be exactly 2 cards
    if len(cards) != 2:
        return state

    # 3. Must be a valid pair
    # Joker pair: both jokers of same type
    if cards[0].is_joker and cards[1].is_joker:
        if cards[0].rank != cards[1].rank:
            return state  # Different joker types, not a valid pair
        # Valid joker pair
        new_suit: Suit | None = None  # Joker pair → 无主
    else:
        # Non-joker: both must be trump rank, same suit
        if cards[0].is_joker or cards[1].is_joker:
            return state  # Mixed joker + non-joker
        if cards[0].rank != state.trump_rank or cards[1].rank != state.trump_rank:
            return state  # Not trump rank cards
        if cards[0].suit != cards[1].suit:
            return state  # Different suits
        new_suit = cards[0].suit

    # 4. Priority check
    new_priority = _pair_priority(cards, state.trump_rank)

    if state.trump_suit is None:
        # 空主: any trump-rank pair is accepted
        # But joker pair on empty trump → stays None (no effective change)
        if cards[0].is_joker and cards[1].is_joker:
            # Joker pair on 空主: record action but trump stays None
            new_action = StirAction(player=player, kind="stir", new_suit=None)
            return StirringState(
                phase="WAITING",
                trump_suit=None,
                trump_rank=state.trump_rank,
                declarer_player=state.declarer_player,
                current_player=next_player_ccw(state.current_player),
                pass_set=frozenset(),
                actions=state.actions + (new_action,),
            )
        # Non-joker pair on 空主: always accepted
    else:
        # Non-empty trump: priority must be higher
        current_priority = _pair_priority(
            _make_trump_pair(state.trump_suit, state.trump_rank),
            state.trump_rank,
        )
        if new_priority <= current_priority:
            return state  # Priority too low

    # 5. Valid stir: update state
    new_action = StirAction(player=player, kind="stir", new_suit=new_suit)
    return StirringState(
        phase="WAITING",
        trump_suit=new_suit,
        trump_rank=state.trump_rank,
        declarer_player=state.declarer_player,
        current_player=next_player_ccw(state.current_player),
        pass_set=frozenset(),
        actions=state.actions + (new_action,),
    )


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
