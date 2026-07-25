"""Black-box trick-play tests for the public game aggregate."""

from itertools import combinations

from server.foundation.result import Ok
from server.game import (
    CommandRejected,
    Seat,
    apply,
    commands,
    next_seat,
    observe,
)
from server.game.rules import play
from server.game.rules.cards import Card, CardId
from server.game.snapshots.tricks import CompletedTrickSnapshot
from tests.support import (
    GAME_SEATS,
    accepted_game_command,
    advance_game_to_action,
    automatic_game_command,
    game_decision,
    started_game,
)


def test_play_rejects_wrong_turn_empty_and_unknown_cards() -> None:
    state, actor, current = advance_game_to_action(
        started_game(seed=53),
        "play",
    )
    wrong_actor = next_seat(actor)

    wrong_turn = apply(
        state,
        wrong_actor,
        commands.Play((CardId(current.hand[0].id),)),
    )
    empty = apply(state, actor, commands.Play(()))
    unknown = apply(
        state,
        actor,
        commands.Play((CardId("unknown"),)),
    )

    assert isinstance(wrong_turn, CommandRejected)
    assert isinstance(empty, CommandRejected)
    assert isinstance(unknown, CommandRejected)
    assert observe(state, actor) == current


def test_lead_records_public_slot_and_advances_counterclockwise() -> (
    None
):
    state, actor, current = advance_game_to_action(
        started_game(seed=59),
        "play",
    )
    card = current.hand[0]

    state = accepted_game_command(
        state,
        actor,
        commands.Play((CardId(card.id),)),
    )

    next_actor, next_view = game_decision(state)
    assert next_actor == next_seat(actor)
    assert next_view.trick is not None
    assert next_view.trick.lead_actor == actor
    assert len(next_view.trick.slots) == 1
    assert next_view.trick.slots[0].actor == actor
    assert next_view.trick.slots[0].cards == (card,)
    assert card not in observe(state, actor).hand


def test_completed_trick_is_public_and_next_winner_leads() -> None:
    state, _, _ = advance_game_to_action(
        started_game(seed=61),
        "play",
    )

    for _ in range(4):
        actor, current = game_decision(state)
        state = accepted_game_command(
            state,
            actor,
            automatic_game_command(current),
        )

    public = tuple(observe(state, seat) for seat in GAME_SEATS)
    completed = public[0].last_completed_trick
    assert completed is not None
    assert len(completed.slots) == 4
    assert completed.points == sum(
        card.points for slot in completed.slots for card in slot.cards
    )
    assert all(
        item.last_completed_trick == completed for item in public
    )
    winner_view = observe(state, completed.winner)
    assert winner_view.awaiting_action == "play"
    assert public[0].trick is not None
    assert public[0].trick.slots == ()
    assert public[0].trick.lead_actor == completed.winner


def test_defender_points_equal_visible_collected_point_cards() -> None:
    state, _, _ = advance_game_to_action(
        started_game(seed=67),
        "play",
    )

    for _ in range(80):
        actor, current = game_decision(state)
        state = accepted_game_command(
            state,
            actor,
            automatic_game_command(current),
        )
        snapshot = observe(state, Seat.A)
        if snapshot.defender_point_cards:
            assert snapshot.defender_points == sum(
                card.points for card in snapshot.defender_point_cards
            )
            assert all(
                observe(state, seat).defender_point_cards
                == snapshot.defender_point_cards
                for seat in GAME_SEATS
            )
            return
    assert False, "seeded game produced no defender point card"


def test_only_latest_completed_trick_is_in_snapshot() -> None:
    state, _, _ = advance_game_to_action(
        started_game(seed=71),
        "play",
    )
    completed: list[CompletedTrickSnapshot] = []

    while len(completed) < 2:
        actor, current = game_decision(state)
        state = accepted_game_command(
            state,
            actor,
            automatic_game_command(current),
        )
        latest = observe(state, Seat.A).last_completed_trick
        if latest is not None and (
            not completed or latest != completed[-1]
        ):
            completed.append(latest)

    assert completed[0] != completed[1]
    assert observe(state, Seat.A).last_completed_trick == completed[1]
    assert not hasattr(observe(state, Seat.A), "trick_history")


def test_failed_throw_exposes_attempt_and_forced_lead() -> None:
    state, actor, current = advance_game_to_action(
        started_game(seed=0),
        "play",
    )
    other_hands = tuple(
        observe(state, seat).hand
        for seat in GAME_SEATS
        if seat != actor
    )
    failed_attempt: tuple[Card, ...] | None = None
    forced: tuple[Card, ...] | None = None
    for count in range(2, 6):
        for attempted in combinations(current.hand, count):
            result = play.resolve_lead(
                current.hand,
                attempted,
                current.trump_suit,
                current.trump_rank,
                other_hands,
            )
            if isinstance(result, Ok) and {
                card.id for card in result.value.cards
            } != {card.id for card in attempted}:
                failed_attempt = attempted
                forced = result.value.cards
                break
        if failed_attempt is not None:
            break
    assert failed_attempt is not None
    assert forced is not None

    state = accepted_game_command(
        state,
        actor,
        commands.Play(
            tuple(CardId(card.id) for card in failed_attempt)
        ),
    )

    trick = observe(state, Seat.B).trick
    assert trick is not None
    assert trick.failed_throw is not None
    assert trick.failed_throw.actor == actor
    assert trick.failed_throw.attempted_cards == failed_attempt
    assert trick.failed_throw.forced_cards == forced
