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
- Each time trump suit is established or changed, the stirring player must
  pick up bottom cards and discard the same number back (EXCHANGING sub-phase).
- After exchange, stirring continues (WAITING sub-phase).
- After an exchange, the exchanging player is skipped; if the other 3 pass,
  stirring completes and play starts.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .card_model import Card, Suit, Rank
from .comparator import bid_value
from .constants import next_player_ccw
from .result import Ok, Rejected, StateResult
from .types import StirAction
from . import exchange_sm as exc


# ---- Priority Mapping ----

# Priority order for stirring: higher value = higher priority
# Matches bid_value's suit ordering: ♦=0, ♣=1, ♥=2, ♠=3, 小王=4, 大王=5
# For stirring, we compare pair values directly using bid_value.

# Big joker pair = 2*100 + 5 = 205; nothing can beat it.
_MAX_STIR_PRIORITY: int = 205


def _empty_cards() -> list[Card]:
    return []


# ---- Input / Output Models ----


class StirInput(BaseModel):
    """Input to create a stirring phase."""

    model_config = ConfigDict(frozen=True)

    trump_suit: Suit | None
    trump_rank: Rank
    initial_bid_cards: list[Card] = Field(default_factory=_empty_cards)
    declarer_player: int
    bottom_cards: list[Card]
    players_hand: list[list[Card]]


class StirResult(BaseModel):
    """Output from a completed stirring phase."""

    model_config = ConfigDict(frozen=True)

    final_trump_suit: Suit | None
    stir_count: int
    final_bottom_cards: list[Card]
    final_players_hand: list[list[Card]]


class StirringState(BaseModel):
    """Internal state of the stirring phase."""

    model_config = ConfigDict(frozen=True)

    phase: Literal["WAITING", "EXCHANGING", "COMPLETE"]
    trump_suit: Suit | None
    trump_rank: Rank
    declarer_player: int
    current_player: int
    pass_set: frozenset[int]
    actions: tuple[StirAction, ...]
    last_stir_player: int | None = None
    current_priority: int = 0
    bottom_cards: list[Card]
    players_hand: list[list[Card]]
    exchange_state: exc.ExchangeState | None = None
    exchanging_player: int | None = None


# ---- Operations ----


def create_stirring(input: StirInput) -> StirringState:
    """Create initial stirring state.

    Starts in EXCHANGING sub-phase: the declarer must first pick up bottom
    cards and discard the same number back (first exchange after bid).
    current_player is set to declarer_player for the exchange.
    """
    initial_priority = _initial_priority(
        input.initial_bid_cards,
        input.trump_suit,
        input.trump_rank,
    )

    # Create exchange state for the initial declarer
    declarer_hand = list(input.players_hand[input.declarer_player])
    exchange_input = exc.ExchangeInput(
        declarer_player=input.declarer_player,
        bottom_cards=list(input.bottom_cards),
        declarer_hand=declarer_hand,
    )
    exchange_state = exc.create_exchange(exchange_input)

    return StirringState(
        phase="EXCHANGING",
        trump_suit=input.trump_suit,
        trump_rank=input.trump_rank,
        declarer_player=input.declarer_player,
        current_player=input.declarer_player,
        pass_set=frozenset(),
        actions=(),
        current_priority=initial_priority,
        bottom_cards=list(input.bottom_cards),
        players_hand=[list(h) for h in input.players_hand],
        exchange_state=exchange_state,
        exchanging_player=input.declarer_player,
    )


def pass_stir(state: StirringState, player: int) -> StateResult[StirringState]:
    """Player passes. Add to pass_set and advance current_player.

    If every player has passed or been skipped for this stir cycle, phase becomes COMPLETE.
    Returns Rejected if player is not the current player or if in EXCHANGING sub-phase.
    """
    if state.phase == "EXCHANGING":
        return Rejected("正在换底牌，不能跳过反主")

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
            bottom_cards=state.bottom_cards,
            players_hand=state.players_hand,
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
        bottom_cards=state.bottom_cards,
        players_hand=state.players_hand,
    ))


def stir(
    state: StirringState, player: int, cards: list[Card]
) -> StateResult[StirringState]:
    """Attempt to stir (change trump suit) with a pair of cards.

    Validation:
    1. Must be in WAITING sub-phase.
    2. Must be current_player.
    3. Must be exactly 2 cards (pair).
    4. Must be a valid pair (same suit, or both same joker type).
    5. Cards must be trump rank or jokers.
    6. Priority must be higher than current trump suit priority.

    If valid: returns Ok(new_state) with updated trump_suit, reset pass_set,
    phase transitions to EXCHANGING (stirring player must exchange bottom cards).
    """
    if state.phase != "WAITING":
        return Rejected("当前不能反主")

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

    # 5. Valid stir: create exchange state for the stirring player
    new_action = StirAction(player=player, kind="stir", new_suit=new_suit)

    stirring_player_hand = list(state.players_hand[player])
    exchange_input = exc.ExchangeInput(
        declarer_player=player,
        bottom_cards=list(state.bottom_cards),
        declarer_hand=stirring_player_hand,
    )
    new_exchange_state = exc.create_exchange(exchange_input)

    return Ok(StirringState(
        phase="EXCHANGING",
        trump_suit=new_suit,
        trump_rank=state.trump_rank,
        declarer_player=state.declarer_player,
        current_player=player,
        pass_set=frozenset(),
        actions=state.actions + (new_action,),
        last_stir_player=player,
        current_priority=new_priority,
        bottom_cards=list(state.bottom_cards),
        players_hand=[list(h) for h in state.players_hand],
        exchange_state=new_exchange_state,
        exchanging_player=player,
    ))


def stir_discard(
    state: StirringState, player: int, cards: list[Card]
) -> StateResult[StirringState]:
    """Discard cards during EXCHANGING sub-phase.

    The player who just stirred (or the declarer on initial exchange) must
    pick up the bottom cards and discard the same number back.

    After successful discard, transitions to WAITING sub-phase with
    updated hands and bottom cards. The next player is CCW after the
    exchanging player.
    """
    if state.phase != "EXCHANGING":
        return Rejected("当前不在换底牌阶段")

    if player != state.exchanging_player:
        return Rejected("只有炒主者可以换底牌")

    if state.exchange_state is None:
        return Rejected("换底牌状态异常")

    match exc.discard(state.exchange_state, cards):
        case Ok(value=new_exc):
            pass
        case Rejected(reason=reason):
            return Rejected(reason)

    if new_exc.phase == "COMPLETE" and new_exc.result is not None:
        # Update hands and bottom cards
        new_hands = [list(h) for h in state.players_hand]
        assert state.exchanging_player is not None
        exchanging = state.exchanging_player
        new_hands[exchanging] = list(new_exc.result.new_hand)
        new_bottom_cards = list(new_exc.result.new_bottom_cards)

        # If current_priority is already the maximum (big joker pair),
        # no higher stir is possible — skip WAITING, go directly to COMPLETE.
        if state.current_priority >= _MAX_STIR_PRIORITY:
            return Ok(StirringState(
                phase="COMPLETE",
                trump_suit=state.trump_suit,
                trump_rank=state.trump_rank,
                declarer_player=state.declarer_player,
                current_player=exchanging,
                pass_set=frozenset(),
                actions=state.actions,
                last_stir_player=state.last_stir_player,
                current_priority=state.current_priority,
                bottom_cards=new_bottom_cards,
                players_hand=new_hands,
            ))

        # Next player is CCW after the exchanging player
        next_player = next_player_ccw(exchanging)

        return Ok(StirringState(
            phase="WAITING",
            trump_suit=state.trump_suit,
            trump_rank=state.trump_rank,
            declarer_player=state.declarer_player,
            current_player=next_player,
            pass_set=frozenset({exchanging}),
            actions=state.actions,
            last_stir_player=state.last_stir_player,
            current_priority=state.current_priority,
            bottom_cards=new_bottom_cards,
            players_hand=new_hands,
        ))

    # Should not reach here (exchange discard always completes in one step)
    return Ok(state.model_copy(update={"exchange_state": new_exc}))


def get_stir_result(state: StirringState) -> StirResult:
    """Extract the result from a completed stirring phase.

    Returns the final trump suit, the number of stir actions taken,
    the final bottom cards, and the final player hands.
    """
    stir_count = sum(1 for a in state.actions if a.kind == "stir")
    return StirResult(
        final_trump_suit=state.trump_suit,
        stir_count=stir_count,
        final_bottom_cards=state.bottom_cards,
        final_players_hand=state.players_hand,
    )


def _initial_priority(
    initial_bid_cards: list[Card],
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> int:
    """Return the priority that later stir actions must beat."""
    if initial_bid_cards:
        return bid_value(initial_bid_cards, trump_rank)
    if trump_suit is None:
        return 0
    return bid_value(
        _make_trump_cards(trump_suit, trump_rank, 2),
        trump_rank,
    )


def _make_trump_cards(suit: Suit, rank: Rank, count: int) -> list[Card]:
    """Create dummy trump-rank cards for priority comparison."""
    result: list[Card] = []
    deck_values: tuple[Literal[1], Literal[2]] = (1, 2)
    for deck in deck_values[:count]:
        result.append(
            Card(
                id=f"dummy-{deck}-{suit.value}-{rank.value}",
                suit=suit,
                rank=rank,
                is_joker=False,
                is_big_joker=False,
                points=0,
                deck=deck,
            )
        )
    return result
