"""Face-count selection grammar for semantic legal actions."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from server.result import Ok, Rejected
from server.rules.card_faces import (
    CardFace,
    FaceCount,
    face_count_width,
    face_sort_key,
)
from server.rules.cards import Card
from server.training.semantic_actions.arguments import (
    InvalidSemanticActionRejected,
    SemanticArgument,
    SemanticArgumentTrace,
)
from server.training.semantic_actions.query import ActionQuery

type CanCompleteSelection = Callable[[tuple[FaceCount, ...]], bool]


def select_arguments(
    *,
    query: ActionQuery,
    selected: tuple[FaceCount, ...],
    can_complete: CanCompleteSelection,
) -> tuple[SemanticArgument, ...]:
    """Return next face-count arguments that can complete later."""
    return tuple(
        argument
        for argument in _raw_select_arguments(
            query=query, selected=selected
        )
        if can_complete((*selected, required_face_count(argument)))
    )


def required_face_count(argument: SemanticArgument) -> FaceCount:
    """Return the face count carried by a selection argument."""
    assert argument.kind == "select_face_count"
    assert argument.face_count is not None
    return argument.face_count


def cards_for_face_counts(
    face_counts: tuple[FaceCount, ...],
    hand_cards: Sequence[Card],
) -> Ok[list[Card]] | Rejected:
    """Bind selected semantic faces to concrete hand cards."""
    result: list[Card] = []
    used_ids: set[str] = set()
    for requested in face_counts:
        matching = [
            card
            for card in hand_cards
            if card.id not in used_ids
            and card.suit == requested.face.suit
            and card.rank == requested.face.rank
        ]
        if len(matching) < requested.count:
            return InvalidSemanticActionRejected(
                "当前手牌没有足够的指定牌面"
            )
        selected = matching[: requested.count]
        result.extend(selected)
        used_ids.update(card.id for card in selected)
    return Ok(value=result)


def remaining_count_after_selected(
    *,
    hand_faces: tuple[FaceCount, ...],
    selected: tuple[FaceCount, ...],
) -> int:
    """Return selectable card count after the canonical prefix."""
    remaining = 0
    last_face = selected[-1].face if selected else None
    for available in hand_faces:
        if last_face is not None and face_sort_key(
            available.face
        ) <= face_sort_key(last_face):
            continue
        selected_count = 0
        for item in selected:
            if item.face == available.face:
                selected_count = item.count
                break
        remaining += max(available.count - selected_count, 0)
    return remaining


def trace_is_selection_only(trace: SemanticArgumentTrace) -> bool:
    """Return whether a trace contains only face-count selections."""
    return all(
        argument.kind == "select_face_count"
        for argument in trace.arguments
    )


def _raw_select_arguments(
    *,
    query: ActionQuery,
    selected: tuple[FaceCount, ...],
) -> tuple[SemanticArgument, ...]:
    selected_count = face_count_width(selected)
    if selected_count >= query.max_select:
        return ()
    last_face = selected[-1].face if selected else None
    result: list[SemanticArgument] = []
    for available in query.hand_faces:
        if _face_already_selected(selected, available.face):
            continue
        if last_face is not None and face_sort_key(
            available.face
        ) <= face_sort_key(last_face):
            continue
        for count in range(1, available.count + 1):
            if selected_count + count > query.max_select:
                continue
            result.append(
                SemanticArgument(
                    "select_face_count",
                    FaceCount(face=available.face, count=count),
                )
            )
    return tuple(result)


def _face_already_selected(
    selected: tuple[FaceCount, ...],
    face: CardFace,
) -> bool:
    return any(item.face == face for item in selected)
