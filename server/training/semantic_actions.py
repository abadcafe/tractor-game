"""Semantic action grammar for model-generated player actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from server.protocol import StateSnapshot
from server.result import Ok, Rejected
from server.rules.card_faces import (
    CardFace,
    FaceCount,
    canonical_face_counts,
    face_count_width,
    face_sort_key,
)
from server.rules.cards import Card, Rank, Suit
from server.training.tokens import RelativeRole, relative_role

type PlayerActionKind = Literal["bid", "stir", "discard", "play"]
type DecisionKind = Literal[
    "bid", "stir", "discard", "lead_play", "follow_play"
]
type SemanticArgumentKind = Literal["pass", "stop", "select_face_count"]

__all__ = (
    "ActionQuery",
    "BoundAction",
    "DecisionKind",
    "GeneratedAction",
    "InvalidSemanticActionRejected",
    "PlayerActionKind",
    "SemanticArgument",
    "SemanticArgumentKind",
    "SemanticArgumentPrefix",
    "SemanticArgumentTrace",
    "bind_generated_action",
    "build_action_query",
    "semantic_prefix_state",
)


@dataclass(frozen=True, slots=True)
class ActionQuery:
    """Player-visible decision shape for semantic action decoding."""

    kind: DecisionKind | None
    hand_faces: tuple[FaceCount, ...]
    pass_allowed: bool
    min_select: int
    max_select: int
    exact_select: int | None
    action_play_order: int | None
    current_trick_width: int | None
    lead_actor: RelativeRole | None
    discard_count: int | None
    trump_suit: Suit | None
    level_rank: Rank
    current_best_bid_role: RelativeRole | None


@dataclass(frozen=True, slots=True)
class SemanticArgumentPrefix:
    """Current generated semantic argument prefix."""

    arguments: tuple[SemanticArgument, ...]


@dataclass(frozen=True, slots=True)
class SemanticArgument:
    """One incremental semantic action argument."""

    kind: SemanticArgumentKind
    face_count: FaceCount | None = None


@dataclass(frozen=True, slots=True)
class SemanticArgumentTrace:
    """Full generated semantic argument trace for one action."""

    arguments: tuple[SemanticArgument, ...]


@dataclass(frozen=True, slots=True)
class GeneratedAction:
    """One model-generated semantic action."""

    action_kind: PlayerActionKind | Literal["pass"]
    message_type: PlayerActionKind
    face_counts: tuple[FaceCount, ...]
    semantic_trace: SemanticArgumentTrace
    is_pass: bool


@dataclass(frozen=True, slots=True)
class BoundAction:
    """Semantic action bound to physical ids for Game.receive()."""

    raw: dict[str, object]


class InvalidSemanticActionRejected(Rejected):
    """Semantic action sequence violated the model grammar."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"语义动作非法：{reason}")


def build_action_query(
    *,
    player_index: int,
    snapshot: StateSnapshot,
) -> ActionQuery:
    """Build the structured player-visible decision request."""
    kind = _decision_kind(snapshot)
    hand_faces = canonical_face_counts(snapshot.player_hand)
    hand_size = face_count_width(hand_faces)
    pass_allowed = kind in ("bid", "stir")
    min_select, max_select, exact_select = _selection_shape(
        kind=kind,
        hand_size=hand_size,
        snapshot=snapshot,
    )
    action_play_order = _action_play_order(snapshot)
    current_trick_width = _current_trick_width(snapshot)
    return ActionQuery(
        kind=kind,
        hand_faces=hand_faces,
        pass_allowed=pass_allowed,
        min_select=min_select,
        max_select=max_select,
        exact_select=exact_select,
        action_play_order=action_play_order,
        current_trick_width=current_trick_width,
        lead_actor=_lead_actor(player_index, snapshot),
        discard_count=_discard_count(snapshot)
        if kind == "discard"
        else None,
        trump_suit=snapshot.trump_suit,
        level_rank=snapshot.trump_rank,
        current_best_bid_role=_current_best_bid_role(
            player_index, snapshot
        ),
    )


def semantic_prefix_state(
    prefix: SemanticArgumentPrefix,
) -> Ok[tuple[FaceCount, ...]] | Rejected:
    """Return selected face counts after applying a prefix."""
    selected: list[FaceCount] = []
    terminated = False
    for argument in prefix.arguments:
        if terminated:
            return InvalidSemanticActionRejected(
                "终止参数后还有额外参数"
            )
        if argument.kind in ("pass", "stop"):
            terminated = True
            continue
        if argument.face_count is None:
            return InvalidSemanticActionRejected(
                "select_face_count 缺少牌面"
            )
        if _face_already_selected(
            tuple(selected), argument.face_count.face
        ):
            return InvalidSemanticActionRejected("同一牌面重复选择")
        if selected and face_sort_key(argument.face_count.face) <= (
            face_sort_key(selected[-1].face)
        ):
            return InvalidSemanticActionRejected("牌面未按规范顺序选择")
        selected.append(argument.face_count)
    return Ok(value=tuple(selected))


def bind_generated_action(
    action: GeneratedAction,
    hand_cards: list[Card] | tuple[Card, ...],
) -> Ok[BoundAction] | Rejected:
    """Bind a semantic action to current physical card ids."""
    if action.is_pass:
        return Ok(
            value=BoundAction(
                raw={"type": action.message_type, "pass": True},
            )
        )
    card_ids_result = _card_ids_for_face_counts(
        action.face_counts, hand_cards
    )
    if isinstance(card_ids_result, Rejected):
        return card_ids_result
    return Ok(
        value=BoundAction(
            raw={
                "type": action.message_type,
                "cards": list(card_ids_result.value),
            },
        )
    )


def _card_ids_for_face_counts(
    face_counts: tuple[FaceCount, ...],
    hand_cards: list[Card] | tuple[Card, ...],
) -> Ok[tuple[str, ...]] | Rejected:
    result: list[str] = []
    for requested in face_counts:
        matching = [
            card
            for card in hand_cards
            if card.suit == requested.face.suit
            and card.rank == requested.face.rank
        ]
        if len(matching) < requested.count:
            return InvalidSemanticActionRejected(
                "当前手牌没有足够的指定牌面"
            )
        result.extend(card.id for card in matching[: requested.count])
    return Ok(value=tuple(result))


def _decision_kind(snapshot: StateSnapshot) -> DecisionKind | None:
    if snapshot.awaiting_action == "bid":
        return "bid"
    if snapshot.awaiting_action == "stir":
        return "stir"
    if snapshot.awaiting_action == "discard":
        return "discard"
    if snapshot.awaiting_action == "play":
        order = _action_play_order(snapshot)
        if order is None or order == 0:
            return "lead_play"
        return "follow_play"
    return None


def _selection_shape(
    *,
    kind: DecisionKind | None,
    hand_size: int,
    snapshot: StateSnapshot,
) -> tuple[int, int, int | None]:
    if kind is None:
        return 0, 0, None
    if kind in ("bid", "stir"):
        return (1 if hand_size > 0 else 0), min(4, hand_size), None
    if kind == "discard":
        count = min(_discard_count(snapshot), hand_size)
        return count, count, count
    if kind == "follow_play":
        width = _current_trick_width(snapshot)
        assert width is not None
        exact = min(width, hand_size)
        return exact, exact, exact
    return (1 if hand_size > 0 else 0), hand_size, None


def _discard_count(snapshot: StateSnapshot) -> int:
    stir = snapshot.stirring_state
    if stir is not None and stir.exchange_count is not None:
        return stir.exchange_count
    return 8


def _current_trick_width(snapshot: StateSnapshot) -> int | None:
    trick = snapshot.trick
    if trick is None:
        return None
    for slot in trick.slots:
        if slot.player == trick.lead_player and slot.cards:
            return len(slot.cards)
    return None


def _action_play_order(snapshot: StateSnapshot) -> int | None:
    trick = snapshot.trick
    if snapshot.awaiting_action != "play" or trick is None:
        return None
    return _play_order(
        lead_player=trick.lead_player, player=trick.current_player
    )


def _lead_actor(
    player_index: int, snapshot: StateSnapshot
) -> RelativeRole | None:
    trick = snapshot.trick
    if snapshot.awaiting_action != "play" or trick is None:
        return None
    return relative_role(player_index, trick.lead_player)


def _current_best_bid_role(
    player_index: int, snapshot: StateSnapshot
) -> RelativeRole | None:
    winner = snapshot.bid_winner
    if winner is None:
        return None
    return relative_role(player_index, winner.player)


def _play_order(*, lead_player: int, player: int) -> int:
    if player >= lead_player:
        return player - lead_player
    return 4 - lead_player + player


def _face_already_selected(
    selected: tuple[FaceCount, ...],
    face: CardFace,
) -> bool:
    return any(item.face == face for item in selected)
