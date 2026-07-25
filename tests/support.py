"""Typed black-box fixtures shared by server tests."""

from __future__ import annotations

from typing import Literal

from server.foundation.result import Ok
from server.game import (
    GameConfig,
    GameSeed,
    GameState,
    Partnership,
    PartnershipMap,
    Seat,
    SeatMap,
    apply,
    commands,
    create,
    observe,
    seats,
    snapshots,
)
from server.game.rules import play
from server.game.rules.cards import (
    Card,
    CardId,
    Rank,
    Suit,
    card_points,
)
from server.game.snapshots.contract import (
    NoTrump,
    PendingTrump,
    SuitedTrump,
    TrumpSnapshot,
)
from server.game.snapshots.events import (
    BidEventSnapshot,
    BottomExchangeSnapshot,
    StirDeclarationEventSnapshot,
    StirringStateSnapshot,
)
from server.game.snapshots.player import AwaitingAction, RoundPhase
from server.game.snapshots.review import ScoringSnapshot
from server.game.snapshots.tricks import (
    CompletedTrickSnapshot,
    TrickSnapshot,
)

type TestSuit = Literal[
    "hearts", "spades", "diamonds", "clubs", "joker"
]
type TestRank = Literal[
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "J",
    "Q",
    "K",
    "A",
    "SJ",
    "BJ",
]

GAME_SEATS = seats()
_DECISION_ACTIONS = frozenset(("bid", "discard", "stir", "play"))
_EMPTY_REMAINING_CARDS = SeatMap(a=0, b=0, c=0, d=0)
_INITIAL_LEVELS = PartnershipMap(
    first=Rank.TWO,
    second=Rank.TWO,
)


def card(
    suit: TestSuit,
    rank: TestRank,
    deck: Literal[1, 2] = 1,
) -> Card:
    """Create one valid physical card."""
    value = Rank(rank)
    return Card(
        id=f"D{deck}-{suit}-{rank}",
        suit=Suit(suit),
        rank=value,
        points=card_points(value),
    )


def seat_values[T](a: T, b: T, c: T, d: T) -> SeatMap[T]:
    """Build an explicitly keyed total seat map for a test."""

    return SeatMap(a=a, b=b, c=c, d=d)


def partnership_levels(
    first: TestRank,
    second: TestRank,
) -> PartnershipMap[Rank]:
    """Build explicitly keyed partnership levels for a test."""

    return PartnershipMap(
        first=Rank(first),
        second=Rank(second),
    )


def accepted_game_command(
    state: GameState,
    actor: Seat,
    command: commands.Command,
) -> GameState:
    result = apply(state, actor, command)
    assert isinstance(result, Ok)
    return result.value


def started_game(*, seed: int = 0) -> GameState:
    state = create(GameConfig(), GameSeed(seed))
    for seat in GAME_SEATS:
        state = accepted_game_command(
            state,
            seat,
            commands.ConfirmRound(),
        )
    return state


def game_decision(
    state: GameState,
) -> tuple[Seat, snapshots.PlayerSnapshot]:
    awaiting = tuple(
        (seat, current)
        for seat in GAME_SEATS
        if (current := observe(state, seat)).awaiting_action
        in _DECISION_ACTIONS
    )
    assert len(awaiting) == 1
    return awaiting[0]


def automatic_game_command(
    current: snapshots.PlayerSnapshot,
) -> commands.Command:
    action = current.awaiting_action
    if action == "bid":
        return commands.PassBid()
    if action == "discard":
        return commands.Bury(
            tuple(
                CardId(item.id)
                for item in current.hand[: len(current.bottom_cards)]
            )
        )
    if action == "stir":
        return commands.PassStir()
    assert action == "play"
    lead = None
    if current.trick is not None and current.trick.slots:
        lead = current.trick.slots[0].cards
    selected = play.choose_legal_play(
        current.hand,
        lead,
        current.trump_suit,
        current.trump_rank,
    )
    return commands.Play(tuple(CardId(item.id) for item in selected))


def advance_game_to_action(
    state: GameState,
    target: AwaitingAction,
) -> tuple[GameState, Seat, snapshots.PlayerSnapshot]:
    for _ in range(500):
        actor, current = game_decision(state)
        if current.awaiting_action == target:
            return state, actor, current
        state = accepted_game_command(
            state,
            actor,
            automatic_game_command(current),
        )
    assert False, f"game did not reach {target}"


def snapshot(
    *,
    phase: RoundPhase = "PLAYING",
    awaiting_action: AwaitingAction | None = "play",
    trump_rank: TestRank = "2",
    trump_suit: TestSuit | None = None,
    hand: list[Card] | tuple[Card, ...] = (),
    remaining_cards: SeatMap[int] = _EMPTY_REMAINING_CARDS,
    bottom_cards: list[Card] | tuple[Card, ...] = (),
    declarer: Seat | None = None,
    defender_points: int = 0,
    trick: TrickSnapshot | None = None,
    last_completed_trick: CompletedTrickSnapshot | None = None,
    defender_point_cards: list[Card] | tuple[Card, ...] = (),
    bid_events: (
        list[BidEventSnapshot] | tuple[BidEventSnapshot, ...]
    ) = (),
    bid_winner: BidEventSnapshot | None = None,
    own_initial_bottom_exchange: BottomExchangeSnapshot | None = None,
    stir_events: (
        list[StirDeclarationEventSnapshot]
        | tuple[StirDeclarationEventSnapshot, ...]
    ) = (),
    stirring_state: StirringStateSnapshot | None = None,
    scoring: ScoringSnapshot | None = None,
    winning_partnership: Partnership | None = None,
    partnership_levels: PartnershipMap[Rank] = _INITIAL_LEVELS,
    next_round_confirmed: frozenset[Seat] = frozenset(),
    mandatory_levels: tuple[Rank, ...] = (Rank.ACE,),
) -> snapshots.PlayerSnapshot:
    """Create one complete player snapshot through its public type."""
    rank = Rank(trump_rank)
    trump: TrumpSnapshot
    if phase == "DEAL_BID":
        trump = PendingTrump(rank=rank)
    elif trump_suit is None:
        trump = NoTrump(rank=rank)
    else:
        trump = SuitedTrump(
            rank=rank,
            suit=Suit(trump_suit),
        )
    return snapshots.PlayerSnapshot(
        phase=phase,
        round_number=1,
        hand=tuple(hand),
        remaining_cards=remaining_cards,
        bottom_cards=tuple(bottom_cards),
        trump=trump,
        declarer=declarer,
        defender_points=defender_points,
        trick=trick,
        last_completed_trick=last_completed_trick,
        defender_point_cards=tuple(defender_point_cards),
        awaiting_action=awaiting_action,
        scoring=scoring,
        winning_partnership=winning_partnership,
        partnership_levels=partnership_levels,
        mandatory_levels=mandatory_levels,
        bid_events=tuple(bid_events),
        bid_winner=bid_winner,
        own_initial_bottom_exchange=own_initial_bottom_exchange,
        stir_events=tuple(stir_events),
        stirring_state=stirring_state,
        next_round_confirmed=next_round_confirmed,
    )


__all__ = (
    "card",
    "partnership_levels",
    "seat_values",
    "snapshot",
)
