"""Black-box bottom-exchange and stirring lifecycle tests."""

from server.game import (
    CommandRejected,
    Seat,
    apply,
    commands,
    next_seat,
    observe,
)
from server.game.rules import bidding
from server.game.rules.cards import CardId, Rank
from server.game.snapshots.contract import NoTrump
from tests.support import (
    GAME_SEATS,
    accepted_game_command,
    advance_game_to_action,
    game_decision,
    started_game,
)


def test_bottom_is_visible_only_to_current_exchange_owner() -> None:
    state, actor, current = advance_game_to_action(
        started_game(seed=31),
        "discard",
    )

    assert len(current.bottom_cards) == 8
    assert len(current.hand) == 33
    for seat in GAME_SEATS:
        if seat != actor:
            assert observe(state, seat).bottom_cards == ()
            assert (
                observe(state, seat).own_initial_bottom_exchange is None
            )


def test_bury_requires_exact_distinct_owned_cards() -> None:
    state, actor, current = advance_game_to_action(
        started_game(seed=37),
        "discard",
    )
    one = CardId(current.hand[0].id)
    too_few = apply(state, actor, commands.Bury((one,)))
    repeated = apply(
        state,
        actor,
        commands.Bury((one,) * len(current.bottom_cards)),
    )
    unknown_ids = tuple(
        CardId(f"unknown-{index}")
        for index in range(len(current.bottom_cards))
    )
    unknown = apply(state, actor, commands.Bury(unknown_ids))

    assert isinstance(too_few, CommandRejected)
    assert isinstance(repeated, CommandRejected)
    assert isinstance(unknown, CommandRejected)
    assert observe(state, actor) == current


def test_initial_exchange_is_private_and_reconnect_safe() -> None:
    state, actor, current = advance_game_to_action(
        started_game(seed=41),
        "discard",
    )
    buried = tuple(
        CardId(card.id)
        for card in current.hand[: len(current.bottom_cards)]
    )
    state = accepted_game_command(
        state,
        actor,
        commands.Bury(buried),
    )

    owner = observe(state, actor)
    opponent = observe(state, next_seat(actor))
    assert owner.own_initial_bottom_exchange is not None
    assert (
        owner.own_initial_bottom_exchange.discarded_bottom_cards
        == tuple(current.hand[:8])
    )
    assert opponent.own_initial_bottom_exchange is None
    assert opponent.bottom_cards == ()


def test_stir_passes_advance_once_and_begin_play_after_full_cycle() -> (
    None
):
    state, owner, exchange = advance_game_to_action(
        started_game(seed=43),
        "discard",
    )
    state = accepted_game_command(
        state,
        owner,
        commands.Bury(
            tuple(CardId(card.id) for card in exchange.hand[:8])
        ),
    )

    passers: list[Seat] = []
    for _ in range(3):
        actor, current = game_decision(state)
        assert current.awaiting_action == "stir"
        passers.append(actor)
        state = accepted_game_command(
            state,
            actor,
            commands.PassStir(),
        )

    assert owner not in passers
    playing = observe(state, owner)
    assert playing.phase == "PLAYING"
    assert playing.awaiting_action == "play"
    assert tuple(event.kind for event in playing.stir_events) == (
        "pass",
        "pass",
        "pass",
    )


def test_stir_rejects_single_and_non_owned_cards() -> None:
    state, owner, exchange = advance_game_to_action(
        started_game(seed=47),
        "discard",
    )
    state = accepted_game_command(
        state,
        owner,
        commands.Bury(
            tuple(CardId(card.id) for card in exchange.hand[:8])
        ),
    )
    actor, current = game_decision(state)
    single = apply(
        state,
        actor,
        commands.Stir((CardId(current.hand[0].id),)),
    )
    unknown = apply(
        state,
        actor,
        commands.Stir((CardId("unknown-1"), CardId("unknown-2"))),
    )

    assert isinstance(single, CommandRejected)
    assert isinstance(unknown, CommandRejected)


def test_big_joker_pair_skips_redundant_stir_waiting() -> None:
    state, owner, exchange = advance_game_to_action(
        started_game(seed=83),
        "discard",
    )
    buried = tuple(
        CardId(card.id)
        for card in exchange.hand
        if card.rank != Rank.BIG_JOKER
    )[:8]
    state = accepted_game_command(
        state,
        owner,
        commands.Bury(buried),
    )
    stirrer, stirring = game_decision(state)
    big_jokers = tuple(
        CardId(card.id)
        for card in stirring.hand
        if card.rank == Rank.BIG_JOKER
    )
    assert len(big_jokers) == 2
    state = accepted_game_command(
        state,
        stirrer,
        commands.Stir(big_jokers),
    )
    exchange_after_stir = observe(state, stirrer)
    state = accepted_game_command(
        state,
        stirrer,
        commands.Bury(
            tuple(
                CardId(card.id) for card in exchange_after_stir.hand[:8]
            )
        ),
    )

    snapshots = tuple(observe(state, seat) for seat in GAME_SEATS)
    assert all(current.phase == "PLAYING" for current in snapshots)
    assert all(
        current.awaiting_action != "stir" for current in snapshots
    )
    assert isinstance(observe(state, owner).trump, NoTrump)
    declaration = bidding.Declaration(
        cards=tuple(
            card
            for card in exchange_after_stir.hand
            if card.rank == Rank.BIG_JOKER
        )
    )
    assert bidding.is_maximum(declaration)
