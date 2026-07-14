"""Closed public contract for persisted training events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type EventName = Literal[
    "initialize",
    "training",
    "process.start",
    "process.stop",
    "rollout",
    "sampling",
    "round",
    "update",
    "update.rank",
    "checkpoint",
    "inference.batch",
    "decision",
    "logging.drop",
]
type ProcessKind = Literal[
    "initializer", "coordinator", "worker", "model_rank"
]

EVENT_NAMES: tuple[EventName, ...] = (
    "initialize",
    "training",
    "process.start",
    "process.stop",
    "rollout",
    "sampling",
    "round",
    "update",
    "update.rank",
    "checkpoint",
    "inference.batch",
    "decision",
    "logging.drop",
)
PROCESS_KINDS: tuple[ProcessKind, ...] = (
    "initializer",
    "coordinator",
    "worker",
    "model_rank",
)


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """Stable process dimensions attached to every event."""

    kind: ProcessKind
    index: int | None = None

    def __post_init__(self) -> None:
        assert self.kind in PROCESS_KINDS
        assert self.index is None or self.index >= 0


@dataclass(frozen=True, slots=True)
class EventContext:
    """Correlation identifiers shared by related events."""

    policy_version: int | None = None
    rollout_id: str | None = None
    worker_index: int | None = None
    model_rank_index: int | None = None
    game_env_index: int | None = None
    episode_id: int | None = None
    player_index: int | None = None
    decision_index: int | None = None
    request_id: int | None = None
    batch_id: int | None = None

    def __post_init__(self) -> None:
        assert self.rollout_id is None or (
            self.rollout_id.strip() == self.rollout_id
            and self.rollout_id
        )
        for value in (
            self.policy_version,
            self.worker_index,
            self.model_rank_index,
            self.game_env_index,
            self.episode_id,
            self.player_index,
            self.decision_index,
            self.request_id,
            self.batch_id,
        ):
            assert value is None or value >= 0
