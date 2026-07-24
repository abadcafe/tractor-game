"""Bottom exchange and overcall lifecycle."""

from __future__ import annotations

from dataclasses import replace

from server.foundation.result import Ok, Rejected
from server.game import commands
from server.game.config import Seat
from server.game.rules import bidding

from . import phases
from .contracts import rule_declaration
from .hands import replace_hand, resolve_cards
from .round_values import (
    BottomExchangeCause,
    BottomExchangeEvent,
    Declaration,
    OpenTrick,
    StirEvent,
)
from .state import GameState, replace_phase
from .topology import SEATS, next_seat, wrong_turn


def bury(
    state: GameState,
    phase: phases.BottomExchange,
    actor: Seat,
    command: commands.Bury,
) -> Ok[GameState] | Rejected:
    if actor != phase.current_actor:
        return wrong_turn(phase.current_actor)
    if len(command.card_ids) != len(phase.bottom_cards):
        return Rejected(f"必须埋下 {len(phase.bottom_cards)} 张底牌")
    match resolve_cards(phase.hand_after_pickup, command.card_ids):
        case Ok(value=discarded):
            pass
        case Rejected() as rejected:
            return rejected
    discarded_ids = {card.id for card in discarded}
    hand = tuple(
        card
        for card in phase.hand_after_pickup
        if card.id not in discarded_ids
    )
    hands = replace_hand(phase.hands, actor, hand)
    stir_index = (
        len(phase.stir_events) - 1
        if phase.cause == BottomExchangeCause.STIR
        else None
    )
    exchange = BottomExchangeEvent(
        actor=actor,
        cause=phase.cause,
        stir_event_index=stir_index,
        picked_up=phase.bottom_cards,
        discarded=discarded,
        resulting_bottom=discarded,
    )
    events = phase.exchange_events + (exchange,)
    decision = phases.StirDecision(
        round_number=phase.round_number,
        levels=phase.levels,
        contract=phase.contract,
        declaration=phase.declaration,
        hands=hands,
        bottom_cards=discarded,
        bottom_owner=actor,
        current_actor=next_seat(actor),
        passed=frozenset((actor,)),
        bid_reveals=phase.bid_reveals,
        stir_events=phase.stir_events,
        exchange_events=events,
    )
    declaration = rule_declaration(phase.declaration)
    if declaration is not None and bidding.is_maximum(declaration):
        return Ok(
            replace_phase(
                state,
                begin_playing(
                    decision,
                    stir_events=phase.stir_events,
                ),
            )
        )
    return Ok(replace_phase(state, decision))


def pass_stir(
    state: GameState,
    phase: phases.StirDecision,
    actor: Seat,
) -> Ok[GameState] | Rejected:
    if actor != phase.current_actor:
        return wrong_turn(phase.current_actor)
    passed = phase.passed | {actor}
    events = phase.stir_events + (
        StirEvent(actor=actor, cards=(), priority=None),
    )
    if len(passed) == len(SEATS):
        return Ok(
            replace_phase(
                state,
                begin_playing(
                    phase,
                    stir_events=events,
                ),
            )
        )
    return Ok(
        replace_phase(
            state,
            replace(
                phase,
                current_actor=next_seat(actor),
                passed=passed,
                stir_events=events,
            ),
        )
    )


def stir(
    state: GameState,
    phase: phases.StirDecision,
    actor: Seat,
    command: commands.Stir,
) -> Ok[GameState] | Rejected:
    if actor != phase.current_actor:
        return wrong_turn(phase.current_actor)
    match resolve_cards(
        phase.hands[int(actor)],
        command.card_ids,
    ):
        case Ok(value=cards):
            pass
        case Rejected() as rejected:
            return rejected
    current = rule_declaration(phase.declaration)
    match bidding.validate_stir(
        cards,
        phase.contract.trump_rank,
        current,
    ):
        case Ok(value=validated):
            pass
        case Rejected() as rejected:
            return rejected
    declaration = Declaration(
        actor=actor,
        cards=validated.cards,
        deal_ordinal=len(phase.bid_reveals),
    )
    contract = replace(
        phase.contract,
        trump_suit=validated.suit,
    )
    event = StirEvent(
        actor=actor,
        cards=validated.cards,
        priority=bidding.priority(
            validated,
            phase.contract.trump_rank,
        ),
    )
    return Ok(
        replace_phase(
            state,
            phases.BottomExchange(
                round_number=phase.round_number,
                levels=phase.levels,
                contract=contract,
                declaration=declaration,
                hands=phase.hands,
                bottom_cards=phase.bottom_cards,
                current_actor=actor,
                hand_after_pickup=(
                    phase.hands[int(actor)] + phase.bottom_cards
                ),
                cause=BottomExchangeCause.STIR,
                bid_reveals=phase.bid_reveals,
                stir_events=phase.stir_events + (event,),
                exchange_events=phase.exchange_events,
            ),
        )
    )


def begin_playing(
    phase: phases.StirDecision,
    *,
    stir_events: tuple[StirEvent, ...],
) -> phases.Playing:
    declarer = phase.contract.declarer
    return phases.Playing(
        round_number=phase.round_number,
        levels=phase.levels,
        contract=phase.contract,
        declaration=phase.declaration,
        hands=phase.hands,
        bottom_cards=phase.bottom_cards,
        bottom_owner=phase.bottom_owner,
        defender_point_cards=(),
        current_trick=OpenTrick(
            lead_actor=declarer,
            current_actor=declarer,
            slots=(),
            failed_throw=None,
        ),
        previous_trick=None,
        bid_reveals=phase.bid_reveals,
        stir_events=stir_events,
        exchange_events=phase.exchange_events,
    )
