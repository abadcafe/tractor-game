"""Semantic argument ids used by the model action head."""

from __future__ import annotations

from dataclasses import dataclass

from server.result import Ok, Rejected
from server.rules.card_faces import MAX_FACE_COUNT, CardFace, FaceCount
from server.rules.cards import Rank, Suit
from server.training.semantic_actions.arguments import (
    InvalidSemanticActionRejected,
    SemanticArgument,
)


@dataclass(frozen=True, slots=True)
class SemanticCodecSchema:
    """Stable model-id schema for semantic arguments."""

    max_argument_tokens: int
    argument_bos_id: int
    argument_pass_id: int
    argument_stop_id: int
    argument_select_base_id: int
    semantic_face_count_count: int
    argument_vocab_size: int


SEMANTIC_CODEC = SemanticCodecSchema(
    max_argument_tokens=36,
    argument_bos_id=1,
    argument_pass_id=2,
    argument_stop_id=3,
    argument_select_base_id=4,
    semantic_face_count_count=54 * MAX_FACE_COUNT,
    argument_vocab_size=4 + 54 * MAX_FACE_COUNT,
)

__all__ = (
    "SEMANTIC_CODEC",
    "SemanticCodecSchema",
    "semantic_argument_from_id",
    "semantic_argument_id",
    "semantic_argument_name",
)


def semantic_argument_name(argument: SemanticArgument) -> str:
    """Return a stable human-readable argument name."""
    if argument.kind == "pass":
        return "PASS"
    if argument.kind == "stop":
        return "STOP"
    assert argument.face_count is not None
    face = argument.face_count.face
    return (
        f"SELECT_{face.suit.value}_{face.rank.value}_"
        f"X{argument.face_count.count}"
    )


def semantic_argument_id(argument: SemanticArgument) -> int:
    """Return the model vocab id for a semantic argument."""
    if argument.kind == "pass":
        return SEMANTIC_CODEC.argument_pass_id
    if argument.kind == "stop":
        return SEMANTIC_CODEC.argument_stop_id
    assert argument.face_count is not None
    return SEMANTIC_CODEC.argument_select_base_id + (
        _face_count_choice_index(argument.face_count)
    )


def semantic_argument_from_id(
    argument_id: int,
) -> Ok[SemanticArgument] | Rejected:
    """Return the semantic argument represented by a vocab id."""
    if argument_id == SEMANTIC_CODEC.argument_pass_id:
        return Ok(value=SemanticArgument("pass"))
    if argument_id == SEMANTIC_CODEC.argument_stop_id:
        return Ok(value=SemanticArgument("stop"))
    if argument_id < SEMANTIC_CODEC.argument_select_base_id:
        return InvalidSemanticActionRejected("不能生成 BOS/PAD")
    index = argument_id - SEMANTIC_CODEC.argument_select_base_id
    if index < 0 or index >= SEMANTIC_CODEC.semantic_face_count_count:
        return InvalidSemanticActionRejected("语义参数 id 超出词表")
    face_index = index // MAX_FACE_COUNT
    count = index % MAX_FACE_COUNT + 1
    face_result = _face_from_index(face_index)
    if isinstance(face_result, Rejected):
        return face_result
    return Ok(
        value=SemanticArgument(
            "select_face_count",
            FaceCount(face=face_result.value, count=count),
        )
    )


def _face_count_choice_index(face_count: FaceCount) -> int:
    face_index = _face_index(face_count.face)
    return face_index * MAX_FACE_COUNT + face_count.count - 1


def _face_index(face: CardFace) -> int:
    suited_count = 4 * 13
    if face.suit == Suit.JOKER:
        if face.rank == Rank.SMALL_JOKER:
            return suited_count
        if face.rank == Rank.BIG_JOKER:
            return suited_count + 1
        assert False
    suit_index = (
        (Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS, Suit.CLUBS)
    ).index(face.suit)
    rank_index = (
        Rank.TWO,
        Rank.THREE,
        Rank.FOUR,
        Rank.FIVE,
        Rank.SIX,
        Rank.SEVEN,
        Rank.EIGHT,
        Rank.NINE,
        Rank.TEN,
        Rank.JACK,
        Rank.QUEEN,
        Rank.KING,
        Rank.ACE,
    ).index(face.rank)
    return suit_index * 13 + rank_index


def _face_from_index(index: int) -> Ok[CardFace] | Rejected:
    suited_count = 4 * 13
    if index == suited_count:
        return Ok(value=CardFace(Suit.JOKER, Rank.SMALL_JOKER))
    if index == suited_count + 1:
        return Ok(value=CardFace(Suit.JOKER, Rank.BIG_JOKER))
    if index < 0 or index >= suited_count:
        return InvalidSemanticActionRejected("牌面 id 超出范围")
    suits = (Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS, Suit.CLUBS)
    ranks = (
        Rank.TWO,
        Rank.THREE,
        Rank.FOUR,
        Rank.FIVE,
        Rank.SIX,
        Rank.SEVEN,
        Rank.EIGHT,
        Rank.NINE,
        Rank.TEN,
        Rank.JACK,
        Rank.QUEEN,
        Rank.KING,
        Rank.ACE,
    )
    return Ok(value=CardFace(suits[index // 13], ranks[index % 13]))
