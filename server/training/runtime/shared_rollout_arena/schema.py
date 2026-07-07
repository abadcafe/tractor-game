"""Binary layout for shared rollout arenas."""

from __future__ import annotations

import struct
from dataclasses import dataclass

HEADER_STRUCT = struct.Struct("<qqqqqqqqqqqddd")
ROW_STRUCT = struct.Struct("<qqqd")


@dataclass(frozen=True, slots=True)
class RolloutArenaHeader:
    """Shared counters for one worker-owned rollout arena."""

    policy_version: int
    sample_count: int
    capacity: int
    full: bool
    round_count: int
    generated_action_count: int
    accepted_action_count: int
    action_choice_count: int
    game_over_count: int
    dropped_sample_count: int
    cancelled_env_count: int
    team0_reward_sum: float
    team1_reward_sum: float
    elapsed_seconds_max: float

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.sample_count >= 0
        assert self.capacity > 0
        assert self.sample_count <= self.capacity
        assert self.round_count >= 0
        assert self.generated_action_count >= 0
        assert self.accepted_action_count >= 0
        assert self.action_choice_count >= 0
        assert self.game_over_count >= 0
        assert self.dropped_sample_count >= 0
        assert self.cancelled_env_count >= 0


@dataclass(frozen=True, slots=True)
class RolloutArenaRow:
    """One accepted training sample reference."""

    model_rank_index: int
    slot_index: int
    slot_generation: int
    return_value: float

    def __post_init__(self) -> None:
        assert self.model_rank_index >= 0
        assert self.slot_index >= 0
        assert self.slot_generation >= 0


def arena_byte_size(*, capacity: int) -> int:
    """Return the shared memory bytes needed for one arena."""
    assert capacity > 0
    return HEADER_STRUCT.size + capacity * ROW_STRUCT.size


def row_offset(index: int) -> int:
    """Return the byte offset for one row index."""
    assert index >= 0
    return HEADER_STRUCT.size + index * ROW_STRUCT.size


def pack_header(
    buffer: memoryview, *, header: RolloutArenaHeader
) -> None:
    """Write one header into a shared memory buffer."""
    HEADER_STRUCT.pack_into(
        buffer,
        0,
        header.policy_version,
        header.sample_count,
        header.capacity,
        1 if header.full else 0,
        header.round_count,
        header.generated_action_count,
        header.accepted_action_count,
        header.action_choice_count,
        header.game_over_count,
        header.dropped_sample_count,
        header.cancelled_env_count,
        header.team0_reward_sum,
        header.team1_reward_sum,
        header.elapsed_seconds_max,
    )


def unpack_header(buffer: memoryview) -> RolloutArenaHeader:
    """Read one header from a shared memory buffer."""
    values = HEADER_STRUCT.unpack_from(buffer, 0)
    return RolloutArenaHeader(
        policy_version=values[0],
        sample_count=values[1],
        capacity=values[2],
        full=values[3] != 0,
        round_count=values[4],
        generated_action_count=values[5],
        accepted_action_count=values[6],
        action_choice_count=values[7],
        game_over_count=values[8],
        dropped_sample_count=values[9],
        cancelled_env_count=values[10],
        team0_reward_sum=values[11],
        team1_reward_sum=values[12],
        elapsed_seconds_max=values[13],
    )


def empty_header(
    *, policy_version: int, capacity: int
) -> RolloutArenaHeader:
    """Return an empty arena header for one policy version."""
    return RolloutArenaHeader(
        policy_version=policy_version,
        sample_count=0,
        capacity=capacity,
        full=False,
        round_count=0,
        generated_action_count=0,
        accepted_action_count=0,
        action_choice_count=0,
        game_over_count=0,
        dropped_sample_count=0,
        cancelled_env_count=0,
        team0_reward_sum=0.0,
        team1_reward_sum=0.0,
        elapsed_seconds_max=0.0,
    )


def pack_row(
    buffer: memoryview, *, index: int, row: RolloutArenaRow
) -> None:
    """Write one sample row into a shared memory buffer."""
    ROW_STRUCT.pack_into(
        buffer,
        row_offset(index),
        row.model_rank_index,
        row.slot_index,
        row.slot_generation,
        row.return_value,
    )


def unpack_row(buffer: memoryview, *, index: int) -> RolloutArenaRow:
    """Read one sample row from a shared memory buffer."""
    values = ROW_STRUCT.unpack_from(buffer, row_offset(index))
    return RolloutArenaRow(
        model_rank_index=values[0],
        slot_index=values[1],
        slot_generation=values[2],
        return_value=values[3],
    )
