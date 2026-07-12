"""Text helpers shared by state-machine rejection reasons."""

from __future__ import annotations

from typing import assert_never

from server.game.room.actions import GameActionKind
from server.game.state_machine.types import DealBidPhase, RoundPhase


def game_action_text(action: GameActionKind) -> str:
    match action:
        case "bid":
            return "抢主"
        case "skip_bid":
            return "不抢"
        case "stir":
            return "反主"
        case "skip_stir":
            return "不反"
        case "discard":
            return "换底牌"
        case "play":
            return "出牌"
        case "next_round":
            return "下一轮"
    assert_never(action)


def round_phase_text(phase: RoundPhase) -> str:
    match phase:
        case "DEAL_BID":
            return "抓牌抢主"
        case "STIRRING":
            return "炒地皮"
        case "PLAYING":
            return "出牌"
        case "SCORING":
            return "结算"
        case "WAITING":
            return "等待下一轮"
    assert_never(phase)


def deal_bid_phase_text(phase: DealBidPhase) -> str:
    match phase:
        case "DEALING":
            return "发牌"
        case "COMPLETE":
            return "发牌完成"
        case "NO_BID":
            return "无人抢主"
    assert_never(phase)


def round_phase_rejection(action_text: str, phase: RoundPhase) -> str:
    return f"不能在{round_phase_text(phase)}阶段{action_text}。"


def deal_bid_phase_rejection(
    action_text: str, phase: DealBidPhase
) -> str:
    return f"不能在{deal_bid_phase_text(phase)}阶段{action_text}。"
