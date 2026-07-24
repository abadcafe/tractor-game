"""Black-box lifecycle tests for the public game aggregate."""

from server.foundation.result import Ok
from server.game import (
    CommandRejected,
    GameConfig,
    GameSeed,
    Seat,
    apply,
    commands,
    create,
    observe,
)
from server.game.snapshots.contract import PendingTrump
from tests.support import (
    GAME_SEATS,
    accepted_game_command,
    automatic_game_command,
    game_decision,
    started_game,
)


def test_create_returns_complete_waiting_observation() -> None:
    state = create(GameConfig(), GameSeed(7))

    for seat in GAME_SEATS:
        current = observe(state, seat)
        assert current.phase == "WAITING"
        assert current.round_number == 0
        assert current.awaiting_action == "next_round"
        assert current.player_hand == ()
        assert current.player_hand_counts == (0, 0, 0, 0)
        assert current.bottom_cards == ()
        assert current.team_levels == (
            current.trump_rank,
            current.trump_rank,
        )
        assert isinstance(current.trump, PendingTrump)


def test_confirm_tracks_each_seat_without_mutating_old_state() -> None:
    original = create(GameConfig(), GameSeed(11))

    result = apply(
        original,
        Seat.NORTH,
        commands.ConfirmRound(),
    )

    assert isinstance(result, Ok)
    assert observe(original, Seat.NORTH).awaiting_action == "next_round"
    assert observe(result.value, Seat.NORTH).awaiting_action is None
    assert (
        observe(result.value, Seat.WEST).awaiting_action == "next_round"
    )
    assert observe(
        result.value, Seat.EAST
    ).next_round_confirmed == frozenset((Seat.NORTH,))


def test_confirm_rejects_duplicate_and_wrong_phase() -> None:
    state = create(GameConfig(), GameSeed(13))
    state = accepted_game_command(
        state,
        Seat.NORTH,
        commands.ConfirmRound(),
    )

    duplicate = apply(state, Seat.NORTH, commands.ConfirmRound())
    wrong_command = apply(state, Seat.WEST, commands.PassBid())

    assert isinstance(duplicate, CommandRejected)
    assert isinstance(wrong_command, CommandRejected)
    assert observe(state, Seat.NORTH).phase == "WAITING"


def test_four_confirmations_start_dealing_at_north() -> None:
    state = started_game(seed=17)

    north = observe(state, Seat.NORTH)
    assert north.phase == "DEAL_BID"
    assert north.round_number == 1
    assert north.awaiting_action == "bid"
    assert north.player_hand_counts == (1, 0, 0, 0)
    assert north.declarer_player is None


def test_seed_controls_shuffle_without_global_random_state() -> None:
    first = started_game(seed=23)
    second = started_game(seed=23)
    different = started_game(seed=24)

    assert observe(first, Seat.NORTH) == observe(second, Seat.NORTH)
    assert (
        observe(first, Seat.NORTH).player_hand
        != observe(different, Seat.NORTH).player_hand
    )


def test_round_review_requires_all_players_before_next_deal() -> None:
    state = started_game(seed=29)
    for _ in range(500):
        actor, current = game_decision(state)
        state = accepted_game_command(
            state,
            actor,
            automatic_game_command(current),
        )
        if observe(state, Seat.NORTH).scoring is not None:
            break
    review = observe(state, Seat.NORTH)
    assert review.scoring is not None

    for seat in GAME_SEATS[:-1]:
        state = accepted_game_command(
            state,
            seat,
            commands.ConfirmRound(),
        )
        assert observe(state, Seat.EAST).round_number == 1
    state = accepted_game_command(
        state,
        GAME_SEATS[-1],
        commands.ConfirmRound(),
    )

    next_round = observe(state, GAME_SEATS[-1])
    assert next_round.phase == "DEAL_BID"
    assert next_round.round_number == 2
    assert next_round.declarer_player is not None
