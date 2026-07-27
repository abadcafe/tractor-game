"""Black-box tests for independent complete-position construction."""

from __future__ import annotations

from server.foundation.result import Ok, Rejected
from server.game import (
    GameConfig,
    GameSeed,
    Partnership,
    PartnershipMap,
    RoundDeal,
    Seat,
    apply,
    commands,
    instantiate,
    observe,
)
from server.game.rules.cards import Rank, create_decks


def _position() -> RoundDeal:
    return RoundDeal(
        config=GameConfig(),
        continuation_seed=GameSeed(91),
        round_number=4,
        partnership_levels=PartnershipMap(
            first=Rank.FIVE,
            second=Rank.SEVEN,
        ),
        fixed_declarer=Seat.C,
        start_actor=Seat.C,
        shuffled_cards=tuple(create_decks()),
    )


def test_instantiate_creates_requested_independent_round() -> None:
    position = _position()

    result = instantiate(position)

    assert isinstance(result, Ok)
    snapshot = observe(result.value, Seat.C)
    assert snapshot.round_number == 4
    assert snapshot.declarer == Seat.C
    assert snapshot.awaiting_action == "bid"
    assert snapshot.remaining_cards.at(Seat.C) == 1
    assert (
        snapshot.partnership_levels.at(Partnership.FIRST) == Rank.FIVE
    )


def test_instantiate_state_branches_without_mutating_root() -> None:
    result = instantiate(_position())
    assert isinstance(result, Ok)
    root = result.value

    child = apply(root, Seat.C, commands.PassBid())

    assert isinstance(child, Ok)
    assert observe(root, Seat.C).remaining_cards.at(Seat.D) == 0
    assert observe(child.value, Seat.C).remaining_cards.at(Seat.D) == 1


def test_instantiate_rejects_incomplete_or_duplicate_deck() -> None:
    position = _position()
    missing = RoundDeal(
        config=position.config,
        continuation_seed=position.continuation_seed,
        round_number=position.round_number,
        partnership_levels=position.partnership_levels,
        fixed_declarer=position.fixed_declarer,
        start_actor=position.start_actor,
        shuffled_cards=position.shuffled_cards[:-1],
    )
    duplicated = RoundDeal(
        config=position.config,
        continuation_seed=position.continuation_seed,
        round_number=position.round_number,
        partnership_levels=position.partnership_levels,
        fixed_declarer=position.fixed_declarer,
        start_actor=position.start_actor,
        shuffled_cards=(
            position.shuffled_cards[:-1] + (position.shuffled_cards[0],)
        ),
    )

    assert isinstance(instantiate(missing), Rejected)
    assert isinstance(instantiate(duplicated), Rejected)
