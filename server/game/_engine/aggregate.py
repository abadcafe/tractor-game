"""Create and dispatch commands for the game aggregate."""

from __future__ import annotations

from dataclasses import replace

import server.game.commands as commands
from server.foundation.result import Ok, Rejected
from server.game.config import GameConfig, GameSeed
from server.game.positions import RoundDeal
from server.game.rejections import CommandRejected
from server.game.rules.cards import Rank, create_decks
from server.game.seating import PartnershipMap, Seat, seats

from . import deal, phases, playing, stirring
from .state import (
    GameState,
    make_state,
    phase_of,
    replace_phase,
)


def create_game(config: GameConfig, seed: GameSeed) -> GameState:
    """Create a game awaiting four confirmations."""
    return make_state(
        config,
        seed,
        phases.AwaitingStart(
            levels=PartnershipMap(
                first=Rank.TWO,
                second=Rank.TWO,
            ),
            confirmed=frozenset(),
        ),
    )


def create_round_from_deal(
    deal_position: RoundDeal,
) -> Ok[GameState] | Rejected:
    """Create an independent state from one complete round deal."""
    expected = tuple(create_decks())
    cards = deal_position.shuffled_cards
    if len(cards) != len(expected):
        return Rejected("完整局面必须包含两副牌")
    expected_by_id = {card.id: card for card in expected}
    actual_by_id = {card.id: card for card in cards}
    if len(actual_by_id) != len(cards):
        return Rejected("完整局面包含重复的物理牌")
    if actual_by_id != expected_by_id:
        return Rejected("完整局面的物理牌集合不正确")
    initial = make_state(
        deal_position.config,
        deal_position.continuation_seed,
        phases.AwaitingStart(
            levels=deal_position.partnership_levels,
            confirmed=frozenset(),
        ),
    )
    return Ok(
        deal.start_round_from_cards(
            initial,
            round_number=deal_position.round_number,
            levels=deal_position.partnership_levels,
            fixed_declarer=deal_position.fixed_declarer,
            start_actor=deal_position.start_actor,
            shuffled_cards=cards,
        )
    )


def apply_command(
    state: GameState,
    actor: Seat,
    command: commands.Command,
) -> Ok[GameState] | CommandRejected:
    """Apply one command without I/O or hidden mutation."""
    result = _apply_phase(state, actor, command)
    if isinstance(result, Rejected):
        return CommandRejected(result.reason)
    return result


def _apply_phase(
    state: GameState,
    actor: Seat,
    command: commands.Command,
) -> Ok[GameState] | Rejected:
    phase = phase_of(state)
    if isinstance(command, commands.ConfirmRound):
        return _confirm(state, phase, actor)
    if isinstance(phase, phases.DealBid):
        if isinstance(command, commands.PassBid):
            return deal.pass_bid(state, phase, actor)
        if isinstance(command, commands.RevealBid):
            return deal.reveal_bid(state, phase, actor, command)
        return _wrong_phase()
    if isinstance(phase, phases.BottomExchange):
        if isinstance(command, commands.Bury):
            return stirring.bury(state, phase, actor, command)
        return _wrong_phase()
    if isinstance(phase, phases.StirDecision):
        if isinstance(command, commands.PassStir):
            return stirring.pass_stir(state, phase, actor)
        if isinstance(command, commands.Stir):
            return stirring.stir(state, phase, actor, command)
        return _wrong_phase()
    if isinstance(phase, phases.Playing):
        if isinstance(command, commands.Play):
            return playing.play_cards(state, phase, actor, command)
        return _wrong_phase()
    assert isinstance(
        phase,
        phases.AwaitingStart | phases.RoundReview | phases.Finished,
    )
    return _wrong_phase()


def _confirm(
    state: GameState,
    phase: phases.GamePhase,
    actor: Seat,
) -> Ok[GameState] | Rejected:
    if isinstance(phase, phases.AwaitingStart):
        if actor in phase.confirmed:
            return Rejected("不能重复确认开始")
        confirmed = phase.confirmed | {actor}
        if len(confirmed) < len(seats()):
            return Ok(
                replace_phase(
                    state,
                    replace(phase, confirmed=confirmed),
                )
            )
        return Ok(
            deal.start_round(
                state,
                round_number=1,
                levels=phase.levels,
                fixed_declarer=None,
            )
        )
    if isinstance(phase, phases.RoundReview):
        if actor in phase.confirmed:
            return Rejected("不能重复确认下一局")
        confirmed = phase.confirmed | {actor}
        if len(confirmed) < len(seats()):
            return Ok(
                replace_phase(
                    state,
                    replace(phase, confirmed=confirmed),
                )
            )
        completed = phase.completed
        return Ok(
            deal.start_round(
                state,
                round_number=completed.round_number + 1,
                levels=completed.levels_after,
                fixed_declarer=completed.next_declarer,
            )
        )
    if isinstance(phase, phases.Finished):
        return Rejected("游戏已经结束")
    return _wrong_phase()


def _wrong_phase() -> Rejected:
    return Rejected("当前阶段不接受这个操作")
