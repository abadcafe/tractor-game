"""Black-box tests for the complete public game aggregate."""

from __future__ import annotations

from server.foundation.result import Ok
from server.game import (
    CommandRejected,
    GameConfig,
    GameSeed,
    GameState,
    Seat,
    SeatMap,
    apply,
    commands,
    create,
    next_seat,
    observe,
    seats,
    snapshots,
)
from server.game.rules import bidding, play
from server.game.rules.cards import CardId
from server.game.snapshots.contract import NoTrump, SuitedTrump

_SEATS = seats()
_DECISION_ACTIONS = frozenset(("bid", "discard", "stir", "play"))


def test_seeded_game_is_deterministic_and_state_is_opaque() -> None:
    first = _start_round(seed=41)
    second = _start_round(seed=41)

    assert not hasattr(first, "phase")
    assert observe(first, Seat.A) == observe(
        second,
        Seat.A,
    )


def test_rejected_command_preserves_current_state() -> None:
    state = _start_round(seed=7)
    before = observe(state, Seat.A)

    result = apply(state, Seat.B, commands.PassBid())

    assert isinstance(result, CommandRejected)
    assert observe(state, Seat.A) == before


def test_complete_round_returns_reconnect_safe_snapshot() -> None:
    state = _start_round(seed=17)
    step_count = 0

    while observe(state, Seat.A).scoring is None:
        actor, snapshot = _decision(state)
        state = _accepted(
            state,
            actor,
            _automatic_command(snapshot),
        )
        step_count += 1
        assert step_count < 500

    seat_a_snapshot = observe(state, Seat.A)
    seat_b_snapshot = observe(state, Seat.B)
    assert seat_a_snapshot.phase == "WAITING"
    assert seat_a_snapshot.scoring is not None
    assert seat_a_snapshot.last_completed_trick is not None
    assert seat_a_snapshot.trick is None
    assert seat_a_snapshot.remaining_cards == SeatMap(
        a=0, b=0, c=0, d=0
    )
    assert len(seat_a_snapshot.bottom_cards) == 8
    assert seat_a_snapshot.scoring == seat_b_snapshot.scoring
    assert seat_a_snapshot.bottom_cards == seat_b_snapshot.bottom_cards
    assert seat_a_snapshot.awaiting_action == "next_round"
    assert seat_b_snapshot.awaiting_action == "next_round"
    assert isinstance(seat_a_snapshot.trump, NoTrump)


def test_bid_and_private_exchange_are_reconnect_safe() -> None:
    state = _start_round(seed=23)
    winning_actor: Seat | None = None
    winning_cards: tuple[CardId, ...] = ()
    deal_ordinal = 0

    while True:
        actor, snapshot = _decision(state)
        assert snapshot.awaiting_action == "bid"
        reveals = bidding.legal_reveals(
            snapshot.hand,
            snapshot.trump_rank,
            current=None,
        )
        if reveals:
            winning_actor = actor
            winning_cards = tuple(
                CardId(card.id) for card in reveals[0].cards
            )
            deal_ordinal = sum(snapshot.remaining_cards.values())
            state = _accepted(
                state,
                actor,
                commands.RevealBid(winning_cards),
            )
            break
        state = _accepted(state, actor, commands.PassBid())

    public = observe(state, Seat.D)
    assert len(public.bid_events) == 1
    assert public.bid_events[0].actor == winning_actor
    assert public.bid_events[0].deal_ordinal == deal_ordinal
    assert public.bid_winner == public.bid_events[0]

    state, actor, exchange = _reach_initial_exchange(state)
    assert actor == winning_actor
    assert exchange.awaiting_action == "discard"
    assert len(exchange.hand) == 33
    assert len(exchange.bottom_cards) == 8
    buried = tuple(CardId(card.id) for card in exchange.hand[:8])
    state = _accepted(
        state,
        actor,
        commands.Bury(buried),
    )

    owner = observe(state, actor)
    opponent = observe(state, next_seat(actor))
    assert isinstance(owner.trump, SuitedTrump)
    assert owner.own_initial_bottom_exchange is not None
    assert owner.own_initial_bottom_exchange.discarded_bottom_cards
    assert opponent.own_initial_bottom_exchange is None
    assert opponent.bottom_cards == ()


def test_stir_moves_bottom_visibility_without_leaking_history() -> None:
    state = _start_round(seed=0)
    state, declarer, exchange = _reach_initial_exchange(state)
    state = _accepted(
        state,
        declarer,
        commands.Bury(
            tuple(CardId(card.id) for card in exchange.hand[:8])
        ),
    )
    stirrer, stir_snapshot = _decision(state)
    assert stir_snapshot.awaiting_action == "stir"
    reveals = tuple(
        declaration
        for declaration in bidding.legal_reveals(
            stir_snapshot.hand,
            stir_snapshot.trump_rank,
            current=None,
        )
        if len(declaration.cards) == 2
    )
    assert reveals
    state = _accepted(
        state,
        stirrer,
        commands.Stir(
            tuple(CardId(card.id) for card in reveals[0].cards)
        ),
    )

    stir_exchange = observe(state, stirrer)
    prior_owner = observe(state, declarer)
    assert stir_exchange.awaiting_action == "discard"
    assert len(stir_exchange.bottom_cards) == 8
    assert isinstance(
        stir_exchange.trump,
        SuitedTrump,
    )
    assert prior_owner.bottom_cards == ()
    assert prior_owner.own_initial_bottom_exchange is not None
    assert stir_exchange.stir_events[-1].kind == "stir"
    assert stir_exchange.stir_events[-1].own_bottom_exchange is None

    state = _accepted(
        state,
        stirrer,
        commands.Bury(
            tuple(CardId(card.id) for card in stir_exchange.hand[:8])
        ),
    )
    stirrer_after = observe(state, stirrer)
    declarer_after = observe(state, declarer)
    assert stirrer_after.stir_events[-1].own_bottom_exchange is not None
    assert declarer_after.stir_events[-1].own_bottom_exchange is None


def _start_round(*, seed: int) -> GameState:
    state = create(GameConfig(), GameSeed(seed))
    for seat in _SEATS:
        state = _accepted(
            state,
            seat,
            commands.ConfirmRound(),
        )
    return state


def _reach_initial_exchange(
    state: GameState,
) -> tuple[GameState, Seat, snapshots.PlayerSnapshot]:
    while True:
        actor, snapshot = _decision(state)
        if snapshot.awaiting_action == "discard":
            return state, actor, snapshot
        assert snapshot.awaiting_action == "bid"
        state = _accepted(state, actor, commands.PassBid())


def _accepted(
    state: GameState,
    actor: Seat,
    command: commands.Command,
) -> GameState:
    result = apply(state, actor, command)
    assert isinstance(result, Ok)
    return result.value


def _decision(
    state: GameState,
) -> tuple[Seat, snapshots.PlayerSnapshot]:
    awaiting = tuple(
        (seat, snapshot)
        for seat in _SEATS
        if (snapshot := observe(state, seat)).awaiting_action
        in _DECISION_ACTIONS
    )
    assert len(awaiting) == 1
    return awaiting[0]


def _automatic_command(
    snapshot: snapshots.PlayerSnapshot,
) -> commands.Command:
    action = snapshot.awaiting_action
    if action == "bid":
        return commands.PassBid()
    if action == "discard":
        return commands.Bury(
            tuple(
                CardId(card.id)
                for card in snapshot.hand[: len(snapshot.bottom_cards)]
            )
        )
    if action == "stir":
        return commands.PassStir()
    assert action == "play"
    lead = None
    if snapshot.trick is not None and snapshot.trick.slots:
        lead = snapshot.trick.slots[0].cards
    cards = play.choose_legal_play(
        snapshot.hand,
        lead,
        snapshot.trump_suit,
        snapshot.trump_rank,
    )
    return commands.Play(tuple(CardId(card.id) for card in cards))
