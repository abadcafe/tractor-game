"""Deal cards and establish the round contract."""

from __future__ import annotations

import random
from dataclasses import replace

from server.foundation.result import Ok, Rejected
from server.game import commands
from server.game.rules import bidding
from server.game.rules.cards import Card, create_decks
from server.game.seating import Seat, SeatMap, next_seat, seats

from ..rejections import wrong_turn
from . import phases
from .contracts import declaration_suit, rule_declaration, trump_rank
from .hands import replace_hand, resolve_cards
from .round_values import (
    BottomExchangeCause,
    Contract,
    Declaration,
    TeamLevels,
)
from .state import GameState, replace_phase, seed_of

_BOTTOM_COUNT = 8


def start_round(
    state: GameState,
    *,
    round_number: int,
    levels: TeamLevels,
    fixed_declarer: Seat | None,
) -> GameState:
    cards = create_decks()
    random_source = random.Random(
        f"{seed_of(state).value}:{round_number}"
    )
    random_source.shuffle(cards)
    bottom = tuple(cards[:_BOTTOM_COUNT])
    deck = tuple(cards[_BOTTOM_COUNT:])
    start = (
        random.Random(
            f"{seed_of(state).value}:{round_number}:opening-seat"
        ).choice(seats())
        if fixed_declarer is None
        else fixed_declarer
    )
    empty: tuple[Card, ...] = ()
    phase = phases.DealBid(
        round_number=round_number,
        levels=levels,
        fixed_declarer=fixed_declarer,
        start_actor=start,
        deck=deck,
        bottom_cards=bottom,
        hands=SeatMap(a=empty, b=empty, c=empty, d=empty),
        deal_cursor=0,
        current_actor=start,
        declaration=None,
        bid_reveals=(),
    )
    return replace_phase(state, _deal_one(phase, start))


def start_round_from_cards(
    state: GameState,
    *,
    round_number: int,
    levels: TeamLevels,
    fixed_declarer: Seat | None,
    start_actor: Seat,
    shuffled_cards: tuple[Card, ...],
) -> GameState:
    """Start one round from an already validated complete shuffle."""
    bottom = shuffled_cards[:_BOTTOM_COUNT]
    deck = shuffled_cards[_BOTTOM_COUNT:]
    empty: tuple[Card, ...] = ()
    phase = phases.DealBid(
        round_number=round_number,
        levels=levels,
        fixed_declarer=fixed_declarer,
        start_actor=start_actor,
        deck=deck,
        bottom_cards=bottom,
        hands=SeatMap(a=empty, b=empty, c=empty, d=empty),
        deal_cursor=0,
        current_actor=start_actor,
        declaration=None,
        bid_reveals=(),
    )
    return replace_phase(state, _deal_one(phase, start_actor))


def pass_bid(
    state: GameState,
    phase: phases.DealBid,
    actor: Seat,
) -> Ok[GameState] | Rejected:
    if actor != phase.current_actor:
        return wrong_turn(phase.current_actor)
    return Ok(replace_phase(state, _advance_deal(phase)))


def reveal_bid(
    state: GameState,
    phase: phases.DealBid,
    actor: Seat,
    command: commands.RevealBid,
) -> Ok[GameState] | Rejected:
    if actor != phase.current_actor:
        return wrong_turn(phase.current_actor)
    if (
        phase.declaration is not None
        and phase.declaration.actor == actor
    ):
        return Rejected("当前亮牌领先者不能连续抬高自己的亮牌")
    match resolve_cards(
        phase.hands.at(actor),
        command.card_ids,
    ):
        case Ok(value=cards):
            pass
        case Rejected() as rejected:
            return rejected
    current = rule_declaration(phase.declaration)
    match bidding.validate_reveal(
        cards,
        trump_rank(phase.levels, phase.fixed_declarer),
        current,
    ):
        case Ok(value=validated):
            declaration = Declaration(
                actor=actor,
                cards=validated.cards,
                deal_ordinal=phase.deal_cursor,
            )
        case Rejected() as rejected:
            return rejected
    next_phase = replace(
        phase,
        declaration=declaration,
        bid_reveals=phase.bid_reveals + (declaration,),
    )
    return Ok(replace_phase(state, _advance_deal(next_phase)))


def _advance_deal(
    phase: phases.DealBid,
) -> phases.DealBid | phases.BottomExchange:
    if phase.deal_cursor == len(phase.deck):
        return _finish_deal(phase)
    target = next_seat(phase.current_actor)
    return _deal_one(phase, target)


def _deal_one(
    phase: phases.DealBid,
    target: Seat,
) -> phases.DealBid:
    assert phase.deal_cursor < len(phase.deck)
    card = phase.deck[phase.deal_cursor]
    hand = phase.hands.at(target) + (card,)
    return replace(
        phase,
        hands=replace_hand(phase.hands, target, hand),
        deal_cursor=phase.deal_cursor + 1,
        current_actor=target,
    )


def _finish_deal(
    phase: phases.DealBid,
) -> phases.BottomExchange:
    declarer = phase.fixed_declarer
    if declarer is None and phase.declaration is not None:
        declarer = phase.declaration.actor
    if declarer is None:
        declarer = phase.start_actor
    rank = trump_rank(phase.levels, declarer)
    contract = Contract(
        declarer=declarer,
        trump_rank=rank,
        trump_suit=declaration_suit(phase.declaration),
    )
    return phases.BottomExchange(
        round_number=phase.round_number,
        levels=phase.levels,
        contract=contract,
        declaration=phase.declaration,
        hands=phase.hands,
        bottom_cards=phase.bottom_cards,
        current_actor=declarer,
        hand_after_pickup=(
            phase.hands.at(declarer) + phase.bottom_cards
        ),
        cause=BottomExchangeCause.INITIAL,
        bid_reveals=phase.bid_reveals,
        stir_events=(),
        exchange_events=(),
    )
