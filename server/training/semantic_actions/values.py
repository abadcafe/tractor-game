"""Semantic action values before and after physical-card binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from server.rules.card_faces import FaceCount
from server.training.semantic_actions.arguments import (
    SemanticArgumentTrace,
)

type PlayerActionKind = Literal["bid", "stir", "discard", "play"]


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
