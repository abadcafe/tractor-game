"""Shared-memory rollout sample arenas for training runtime."""

from server.training.runtime.shared_rollout_arena.group import (
    RolloutSampleTargetReached,
    RolloutStopRequested,
    RolloutWaitOutcome,
    SharedRolloutArenaGroup,
    close_shared_rollout_arenas,
    create_shared_rollout_arena_group,
    reset_rollout_arenas,
    snapshot_rollout_arenas,
    wait_rollout_sample_target_or_stop,
)
from server.training.runtime.shared_rollout_arena.reader import (
    SharedRolloutArenaReader,
    attach_rollout_arena_reader,
)
from server.training.runtime.shared_rollout_arena.types import (
    RolloutArenaAppendResult,
    RolloutArenaHandle,
    RolloutArenaSnapshot,
    RolloutRoundMetrics,
)
from server.training.runtime.shared_rollout_arena.writer import (
    SharedRolloutArenaWriter,
    attach_rollout_arena_writer,
)

__all__ = (
    "RolloutArenaAppendResult",
    "RolloutArenaHandle",
    "RolloutArenaSnapshot",
    "RolloutRoundMetrics",
    "RolloutSampleTargetReached",
    "RolloutStopRequested",
    "RolloutWaitOutcome",
    "SharedRolloutArenaGroup",
    "SharedRolloutArenaReader",
    "SharedRolloutArenaWriter",
    "attach_rollout_arena_reader",
    "attach_rollout_arena_writer",
    "close_shared_rollout_arenas",
    "create_shared_rollout_arena_group",
    "reset_rollout_arenas",
    "snapshot_rollout_arenas",
    "wait_rollout_sample_target_or_stop",
)
