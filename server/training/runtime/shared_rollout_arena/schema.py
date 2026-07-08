"""Binary layout for shared rollout arenas."""

from __future__ import annotations

import struct
from dataclasses import dataclass

HEADER_STRUCT = struct.Struct("<qqqqqqqqqqqddd")
_I64 = struct.Struct("<q")
_F32 = struct.Struct("<f")


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


def arena_byte_size(*, capacity: int) -> int:
    """Return the shared memory bytes needed for one arena."""
    assert capacity > 0
    return _return_values_offset(capacity) + capacity * _F32.size


def pack_sample_reference(
    *,
    buffer: memoryview,
    capacity: int,
    index: int,
    model_rank_index: int,
    slot_index: int,
    slot_generation: int,
    return_value: float,
) -> None:
    """Write one sample reference into the columnar arena."""
    assert capacity > 0
    assert index >= 0
    assert index < capacity
    assert model_rank_index >= 0
    assert slot_index >= 0
    assert slot_generation >= 0
    _I64.pack_into(
        buffer,
        _model_rank_indices_offset() + index * _I64.size,
        model_rank_index,
    )
    _I64.pack_into(
        buffer,
        _slot_indices_offset(capacity) + index * _I64.size,
        slot_index,
    )
    _I64.pack_into(
        buffer,
        _slot_generations_offset(capacity) + index * _I64.size,
        slot_generation,
    )
    _F32.pack_into(
        buffer,
        _return_values_offset(capacity) + index * _F32.size,
        return_value,
    )


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


def unpack_model_rank_index(*, buffer: memoryview, index: int) -> int:
    """Read one model-rank index from the arena columns."""
    return _read_i64(buffer, _model_rank_indices_offset(), index)


def unpack_slot_index(
    *, buffer: memoryview, capacity: int, index: int
) -> int:
    """Read one slot index from the arena columns."""
    return _read_i64(buffer, _slot_indices_offset(capacity), index)


def unpack_slot_generation(
    *, buffer: memoryview, capacity: int, index: int
) -> int:
    """Read one slot generation from the arena columns."""
    return _read_i64(buffer, _slot_generations_offset(capacity), index)


def unpack_return_value(
    *, buffer: memoryview, capacity: int, index: int
) -> float:
    """Read one return value from the arena columns."""
    values = _F32.unpack_from(
        buffer, _return_values_offset(capacity) + index * _F32.size
    )
    return float(values[0])


def _read_i64(buffer: memoryview, offset: int, index: int) -> int:
    values = _I64.unpack_from(buffer, offset + index * _I64.size)
    return int(values[0])


def _model_rank_indices_offset() -> int:
    return HEADER_STRUCT.size


def _slot_indices_offset(capacity: int) -> int:
    return _model_rank_indices_offset() + capacity * _I64.size


def _slot_generations_offset(capacity: int) -> int:
    return _slot_indices_offset(capacity) + capacity * _I64.size


def _return_values_offset(capacity: int) -> int:
    return _slot_generations_offset(capacity) + capacity * _I64.size
