"""Viewer-relative public and private action facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from server.game.rules.card_faces import FaceCount
from server.training.observation_structure import RoundEventOrdinal
from server.training.relative_state.relations import (
    RelativeActor,
    TrickPosition,
)

type ActionDisposition = Literal["pass", "reveal"]


@dataclass(frozen=True, slots=True)
class RelativeBidAction:
    """One bid decision observed during dealing."""

    actor: RelativeActor
    disposition: ActionDisposition
    revealed: tuple[FaceCount, ...]
    event_ordinal: RoundEventOrdinal


@dataclass(frozen=True, slots=True)
class RelativeStirAction:
    """One public stir decision after dealing."""

    actor: RelativeActor
    disposition: ActionDisposition
    revealed: tuple[FaceCount, ...]
    event_ordinal: RoundEventOrdinal


@dataclass(frozen=True, slots=True)
class RelativeExchangeAction:
    """One viewer-private bottom exchange memory."""

    picked_up: tuple[FaceCount, ...]
    discarded: tuple[FaceCount, ...]
    event_ordinal: RoundEventOrdinal


@dataclass(frozen=True, slots=True)
class RelativePlayAction:
    """One public play with failed-throw extras normalized."""

    actor: RelativeActor
    trick_position: TrickPosition
    played: tuple[FaceCount, ...]
    revealed_extra: tuple[FaceCount, ...]


type RelativeRoundAction = (
    RelativeBidAction | RelativeStirAction | RelativeExchangeAction
)


__all__ = (
    "RelativeBidAction",
    "RelativeExchangeAction",
    "RelativePlayAction",
    "RelativeRoundAction",
    "RelativeStirAction",
)
