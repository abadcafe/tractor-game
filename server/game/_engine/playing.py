"""Validate plays and resolve tricks."""

from __future__ import annotations

from dataclasses import replace

import server.game.commands as commands
import server.game.rules.play as play
from server.foundation.result import Ok, Rejected
from server.game.rules.cards import CardId
from server.game.seating import (
    Seat,
    next_seat,
    partnership_of,
    seats,
)

from ..rejections import wrong_turn
from . import phases
from .completion import finish_round
from .hands import replace_hand, resolve_cards
from .round_values import (
    Hands,
    InternalCompletedTrick,
    InternalFailedThrow,
    InternalTrickSlot,
    OpenTrick,
)
from .state import GameState, phase_of, replace_phase


def play_cards(
    state: GameState,
    phase: phases.Playing,
    actor: Seat,
    command: commands.Play,
) -> Ok[GameState] | Rejected:
    if actor != phase.current_trick.current_actor:
        return wrong_turn(phase.current_trick.current_actor)
    match resolve_cards(
        phase.hands.at(actor),
        command.card_ids,
    ):
        case Ok(value=attempted):
            pass
        case Rejected() as rejected:
            return rejected
    if not attempted:
        return Rejected("出牌不能为空")

    failed = phase.current_trick.failed_throw
    if not phase.current_trick.slots:
        other_hands = tuple(
            phase.hands.at(seat) for seat in seats() if seat != actor
        )
        match play.resolve_lead(
            phase.hands.at(actor),
            attempted,
            phase.contract.trump_suit,
            phase.contract.trump_rank,
            other_hands,
        ):
            case Ok(value=resolution):
                actual = resolution.cards
            case Rejected() as rejected:
                return rejected
        if {card.id for card in actual} != {
            card.id for card in attempted
        }:
            failed = InternalFailedThrow(
                actor=actor,
                attempted_cards=attempted,
                forced_cards=actual,
            )
    else:
        lead = phase.current_trick.slots[0].cards
        match play.validate_follow(
            phase.hands.at(actor),
            attempted,
            lead,
            phase.contract.trump_suit,
            phase.contract.trump_rank,
        ):
            case Ok():
                actual = attempted
            case Rejected() as rejected:
                return rejected

    actual_ids = {card.id for card in actual}
    hand = tuple(
        card
        for card in phase.hands.at(actor)
        if card.id not in actual_ids
    )
    slots = phase.current_trick.slots + (
        InternalTrickSlot(actor=actor, cards=actual),
    )
    hands = replace_hand(phase.hands, actor, hand)
    if len(slots) == len(seats()):
        return Ok(_complete_trick(state, phase, hands, slots, failed))

    next_phase = replace(
        phase,
        hands=hands,
        current_trick=OpenTrick(
            lead_actor=phase.current_trick.lead_actor,
            current_actor=next_seat(actor),
            slots=slots,
            failed_throw=failed,
        ),
    )
    next_state = replace_phase(state, next_phase)
    if len(slots) == 1 and _should_auto_complete(next_phase):
        return Ok(_auto_complete(next_state, next_phase))
    return Ok(next_state)


def _complete_trick(
    state: GameState,
    phase: phases.Playing,
    hands: Hands,
    slots: tuple[InternalTrickSlot, ...],
    failed: InternalFailedThrow | None,
) -> GameState:
    lead = slots[0].cards
    winner_slot = slots[0]
    for slot in slots[1:]:
        if (
            play.compare(
                slot.cards,
                winner_slot.cards,
                lead,
                phase.contract.trump_suit,
                phase.contract.trump_rank,
            )
            > 0
        ):
            winner_slot = slot
    points = sum(card.points for slot in slots for card in slot.cards)
    completed = InternalCompletedTrick(
        lead_actor=phase.current_trick.lead_actor,
        slots=slots,
        winner=winner_slot.actor,
        points=points,
        failed_throw=failed,
    )
    point_cards = phase.defender_point_cards
    if partnership_of(winner_slot.actor) != partnership_of(
        phase.contract.declarer
    ):
        point_cards = point_cards + tuple(
            card
            for slot in slots
            for card in slot.cards
            if card.points > 0
        )
    next_phase = replace(
        phase,
        hands=hands,
        defender_point_cards=point_cards,
        current_trick=OpenTrick(
            lead_actor=winner_slot.actor,
            current_actor=winner_slot.actor,
            slots=(),
            failed_throw=None,
        ),
        previous_trick=completed,
    )
    if all(not hand for hand in hands.values()):
        return finish_round(state, next_phase, completed)
    return replace_phase(state, next_phase)


def _should_auto_complete(phase: phases.Playing) -> bool:
    slots = phase.current_trick.slots
    if len(slots) != 1:
        return False
    lead_count = len(slots[0].cards)
    if phase.hands.at(slots[0].actor):
        return False
    return all(
        len(phase.hands.at(seat)) == lead_count
        for seat in seats()
        if seat != slots[0].actor
    )


def _auto_complete(
    state: GameState,
    phase: phases.Playing,
) -> GameState:
    next_state = state
    next_phase = phase
    while len(next_phase.current_trick.slots) < len(seats()):
        actor = next_phase.current_trick.current_actor
        ids = tuple(
            CardId(card.id) for card in next_phase.hands.at(actor)
        )
        result = play_cards(
            next_state,
            next_phase,
            actor,
            commands.Play(card_ids=ids),
        )
        assert isinstance(result, Ok)
        next_state = result.value
        candidate = phase_of(next_state)
        if not isinstance(candidate, phases.Playing):
            return next_state
        next_phase = candidate
    return next_state
