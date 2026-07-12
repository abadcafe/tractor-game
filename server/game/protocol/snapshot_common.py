"""Shared types for player-facing snapshot models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

type RoundPhase = Literal[
    "DEAL_BID",
    "STIRRING",
    "PLAYING",
    "SCORING",
    "WAITING",
]
type StirringPhase = Literal["WAITING", "EXCHANGING", "COMPLETE"]
type AwaitingAction = Literal[
    "bid", "stir", "discard", "play", "next_round"
]
type BidEventKind = Literal["trump_rank", "joker"]
type JokerType = Literal["big", "small"]


class SnapshotModel(BaseModel):
    """Frozen base model for player-facing snapshots."""

    model_config = ConfigDict(frozen=True)
