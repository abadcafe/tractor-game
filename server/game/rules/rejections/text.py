"""Text helpers shared by rule rejection reasons."""

from __future__ import annotations

from typing import assert_never

from server.game.rules.cards import Suit
from server.game.rules.types import EffectiveSuit, PlayShapeInfo


def effective_suit_name(suit: EffectiveSuit) -> str:
    if suit == "trump":
        return "主牌"
    if suit == Suit.HEARTS:
        return "红桃"
    if suit == Suit.SPADES:
        return "黑桃"
    if suit == Suit.CLUBS:
        return "梅花"
    if suit == Suit.DIAMONDS:
        return "方片"
    if suit == Suit.JOKER:
        return "王牌"
    assert_never(suit)


def play_shape_text(shape: PlayShapeInfo) -> str:
    match shape.kind:
        case "empty":
            return "空牌"
        case "single":
            assert shape.suit is not None
            return f"{effective_suit_name(shape.suit)}单张"
        case "pair":
            assert shape.suit is not None
            return f"{effective_suit_name(shape.suit)}对子"
        case "tractor":
            assert shape.suit is not None
            assert shape.pair_count is not None
            suit_name = effective_suit_name(shape.suit)
            return f"{suit_name}{shape.pair_count}连对"
        case "cards":
            assert shape.suit is not None
            suit_name = effective_suit_name(shape.suit)
            return f"{shape.card_count}张{suit_name}牌"
    assert_never(shape.kind)
