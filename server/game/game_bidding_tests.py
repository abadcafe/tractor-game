"""Black-box deal-time bidding tests for the game aggregate."""

from server.foundation.result import Ok
from server.game import (
    CommandRejected,
    Seat,
    SeatMap,
    apply,
    commands,
    next_seat,
    observe,
)
from server.game.rules import bidding
from server.game.rules.cards import CardId, Rank
from tests.support import (
    accepted_game_command,
    game_decision,
    started_game,
)


def test_pass_deals_one_card_to_next_seat() -> None:
    state = started_game(seed=3)
    actor, before = game_decision(state)

    state = accepted_game_command(
        state,
        actor,
        commands.PassBid(),
    )

    after_actor = next_seat(actor)
    after = observe(state, after_actor)
    assert before.remaining_cards.at(actor) == 1
    assert before.remaining_cards.at(after_actor) == 0
    assert after.remaining_cards.at(actor) == 1
    assert after.remaining_cards.at(after_actor) == 1
    assert observe(state, actor).awaiting_action is None
    assert after.awaiting_action == "bid"


def test_bid_rejects_wrong_turn_unknown_duplicate_and_non_level() -> (
    None
):
    state = started_game(seed=5)
    actor, snapshot = game_decision(state)
    first = CardId(snapshot.hand[0].id)

    wrong_turn = apply(
        state,
        next_seat(actor),
        commands.RevealBid((first,)),
    )
    unknown = apply(
        state,
        actor,
        commands.RevealBid((CardId("unknown"),)),
    )
    duplicate = apply(
        state,
        actor,
        commands.RevealBid((first, first)),
    )
    non_level = apply(
        state,
        actor,
        commands.RevealBid((first,)),
    )

    assert isinstance(wrong_turn, CommandRejected)
    assert isinstance(unknown, CommandRejected)
    assert isinstance(duplicate, CommandRejected)
    if snapshot.hand[0].rank != Rank.TWO:
        assert isinstance(non_level, CommandRejected)


def test_successful_bid_records_actor_cards_and_deal_ordinal() -> None:
    state = started_game(seed=7)
    while True:
        actor, current = game_decision(state)
        reveals = bidding.legal_reveals(
            current.hand,
            current.trump_rank,
            current=None,
        )
        if reveals:
            reveal = reveals[0]
            ordinal = sum(current.remaining_cards.values())
            state = accepted_game_command(
                state,
                actor,
                commands.RevealBid(
                    tuple(CardId(card.id) for card in reveal.cards)
                ),
            )
            break
        state = accepted_game_command(
            state,
            actor,
            commands.PassBid(),
        )

    public = observe(state, Seat.D)
    assert len(public.bid_events) == 1
    assert public.bid_events[0].actor == actor
    assert public.bid_events[0].cards == reveal.cards
    assert public.bid_events[0].deal_ordinal == ordinal
    assert public.bid_winner == public.bid_events[0]


def test_current_bid_winner_cannot_raise_again_without_override() -> (
    None
):
    state = started_game(seed=11)
    while True:
        actor, current = game_decision(state)
        reveals = bidding.legal_reveals(
            current.hand,
            current.trump_rank,
            current=None,
        )
        if reveals:
            state = accepted_game_command(
                state,
                actor,
                commands.RevealBid(
                    tuple(CardId(card.id) for card in reveals[0].cards)
                ),
            )
            winner = actor
            break
        state = accepted_game_command(
            state,
            actor,
            commands.PassBid(),
        )
    while True:
        actor, current = game_decision(state)
        if actor == winner:
            break
        state = accepted_game_command(
            state,
            actor,
            commands.PassBid(),
        )

    result = apply(
        state,
        winner,
        commands.RevealBid((CardId(current.hand[0].id),)),
    )

    assert isinstance(result, CommandRejected)
    assert "领先者" in result.reason


def test_overridden_bidder_can_bid_again() -> None:
    state = started_game(seed=0)
    first: Seat | None = None
    overrider: Seat | None = None

    for _ in range(100):
        actor, current = game_decision(state)
        current_declaration = (
            None
            if current.bid_winner is None
            else bidding.Declaration(cards=current.bid_winner.cards)
        )
        reveals = bidding.legal_reveals(
            current.hand,
            current.trump_rank,
            current=current_declaration,
        )
        if first is None and reveals:
            first = actor
            command = commands.RevealBid(
                tuple(CardId(card.id) for card in reveals[0].cards)
            )
        elif first is not None and overrider is None:
            if actor != first and reveals:
                overrider = actor
                command = commands.RevealBid(
                    tuple(CardId(card.id) for card in reveals[0].cards)
                )
            else:
                command = commands.PassBid()
        elif actor == first and reveals:
            result = apply(
                state,
                actor,
                commands.RevealBid(
                    tuple(CardId(card.id) for card in reveals[0].cards)
                ),
            )
            assert isinstance(result, Ok)
            return
        else:
            command = commands.PassBid()
        state = accepted_game_command(state, actor, command)
    assert False, "seeded deal did not exercise an overridden bidder"


def test_last_dealt_card_still_has_a_bid_decision() -> None:
    state = started_game(seed=0)
    while True:
        actor, current = game_decision(state)
        if sum(current.remaining_cards.values()) == 100:
            break
        state = accepted_game_command(
            state,
            actor,
            commands.PassBid(),
        )

    assert current.awaiting_action == "bid"
    assert current.remaining_cards == SeatMap(
        a=25,
        b=25,
        c=25,
        d=25,
    )
    reveals = bidding.legal_reveals(
        current.hand,
        current.trump_rank,
        current=None,
    )
    assert reveals
    state = accepted_game_command(
        state,
        actor,
        commands.RevealBid(
            tuple(CardId(card.id) for card in reveals[0].cards)
        ),
    )
    exchange = observe(state, actor)
    assert exchange.awaiting_action == "discard"
