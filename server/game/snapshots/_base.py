"""Private stable values used by player snapshots."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from server.game.rules.cards import Rank
from server.game.seating import PartnershipMap, SeatMap

type RoundPhase = Literal[
    "DEAL_BID",
    "STIRRING",
    "PLAYING",
    "WAITING",
]
type AwaitingAction = Literal[
    "bid",
    "discard",
    "stir",
    "play",
    "next_round",
]
type RemainingCards = SeatMap[int]
type PartnershipLevels = PartnershipMap[Rank]


class SnapshotModel(BaseModel):
    """Immutable strict base for domain snapshots."""

    model_config = ConfigDict(frozen=True, extra="forbid")
