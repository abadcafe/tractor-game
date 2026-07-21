"""Closed output vocabulary shared by policy, rules, and replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from server.foundation.result import Ok, Rejected
from server.game.rules.card_faces import (
    MAX_FACE_COUNT,
    CardFace,
    FaceCount,
    face_sort_key,
)
from server.game.rules.cards import Rank, Suit

type ActionChoiceKind = Literal["pass", "finish", "card"]

PASS_CHOICE_ID: int = 0
FINISH_CHOICE_ID: int = 1
CARD_CHOICE_BASE_ID: int = 2
CARD_FACE_COUNT: int = 54
CARD_CHOICE_COUNT: int = CARD_FACE_COUNT * MAX_FACE_COUNT
ACTION_CHOICE_COUNT: int = CARD_CHOICE_BASE_ID + CARD_CHOICE_COUNT
MAX_ACTION_STEPS: int = 26

_SUITED_RANKS: tuple[Rank, ...] = (
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
_SUITS: tuple[Suit, ...] = (
    Suit.HEARTS,
    Suit.SPADES,
    Suit.DIAMONDS,
    Suit.CLUBS,
)
CARD_FACES: tuple[CardFace, ...] = (
    *(
        CardFace(suit, rank)
        for suit in _SUITS
        for rank in _SUITED_RANKS
    ),
    CardFace(Suit.JOKER, Rank.SMALL_JOKER),
    CardFace(Suit.JOKER, Rank.BIG_JOKER),
)


class InvalidActionRejected(Rejected):
    """An action choice sequence violated the generation grammar."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"动作非法：{reason}")


@dataclass(frozen=True, slots=True)
class ActionChoice:
    """One member of the fixed 110-choice policy vocabulary."""

    kind: ActionChoiceKind
    face_count: FaceCount | None = None

    def __post_init__(self) -> None:
        if self.kind == "card":
            assert self.face_count is not None
            return
        assert self.face_count is None


@dataclass(frozen=True, slots=True)
class ActionPrefix:
    """A partially generated action choice sequence."""

    choices: tuple[ActionChoice, ...]


@dataclass(frozen=True, slots=True)
class ActionTrace:
    """A complete policy action choice sequence."""

    choices: tuple[ActionChoice, ...]


def action_choice_id(choice: ActionChoice) -> int:
    """Return the dense fixed-vocabulary id for one action choice."""
    if choice.kind == "pass":
        return PASS_CHOICE_ID
    if choice.kind == "finish":
        return FINISH_CHOICE_ID
    assert choice.face_count is not None
    return CARD_CHOICE_BASE_ID + (
        face_index(choice.face_count.face) * MAX_FACE_COUNT
        + choice.face_count.count
        - 1
    )


def action_choice_from_id(
    choice_id: int,
) -> Ok[ActionChoice] | Rejected:
    """Decode one dense fixed-vocabulary id."""
    if choice_id == PASS_CHOICE_ID:
        return Ok(value=ActionChoice("pass"))
    if choice_id == FINISH_CHOICE_ID:
        return Ok(value=ActionChoice("finish"))
    card_index = choice_id - CARD_CHOICE_BASE_ID
    if card_index < 0 or card_index >= CARD_CHOICE_COUNT:
        return InvalidActionRejected("候选 id 超出固定词表")
    return Ok(
        value=ActionChoice(
            "card",
            FaceCount(
                face=CARD_FACES[card_index // MAX_FACE_COUNT],
                count=card_index % MAX_FACE_COUNT + 1,
            ),
        )
    )


def action_choice_name(choice: ActionChoice) -> str:
    """Return the stable diagnostics name for one choice."""
    if choice.kind == "pass":
        return "PASS"
    if choice.kind == "finish":
        return "FINISH"
    assert choice.face_count is not None
    face = choice.face_count.face
    return (
        f"CARD_{face.suit.value}_{face.rank.value}_"
        f"X{choice.face_count.count}"
    )


def face_index(face: CardFace) -> int:
    """Return the canonical zero-based index of a physical card face."""
    return CARD_FACES.index(face)


def action_prefix_cards(
    prefix: ActionPrefix,
) -> Ok[tuple[FaceCount, ...]] | Rejected:
    """Validate a prefix and return its canonically selected cards."""
    selected: list[FaceCount] = []
    terminated = False
    for choice in prefix.choices:
        if terminated:
            return InvalidActionRejected("终止选择后还有额外选择")
        if choice.kind in ("pass", "finish"):
            terminated = True
            continue
        assert choice.face_count is not None
        if any(
            item.face == choice.face_count.face for item in selected
        ):
            return InvalidActionRejected("同一牌面重复选择")
        if selected and face_sort_key(choice.face_count.face) <= (
            face_sort_key(selected[-1].face)
        ):
            return InvalidActionRejected("牌面未按规范顺序选择")
        selected.append(choice.face_count)
    return Ok(value=tuple(selected))


__all__ = (
    "ACTION_CHOICE_COUNT",
    "ActionChoice",
    "ActionChoiceKind",
    "ActionPrefix",
    "ActionTrace",
    "CARD_CHOICE_BASE_ID",
    "CARD_CHOICE_COUNT",
    "CARD_FACE_COUNT",
    "CARD_FACES",
    "FINISH_CHOICE_ID",
    "InvalidActionRejected",
    "MAX_ACTION_STEPS",
    "PASS_CHOICE_ID",
    "action_choice_from_id",
    "action_choice_id",
    "action_choice_name",
    "action_prefix_cards",
    "face_index",
)
