"""Binary layout for atomic completed-round rollout arenas."""

from __future__ import annotations

import struct
from dataclasses import dataclass

from pydantic import ConfigDict, TypeAdapter

MAX_TRAINABLE_DECISIONS_PER_ROUND = 1024

_HEADER = struct.Struct("<" + "q" * 20 + "d" * 3)
_I64 = struct.Struct("<q")
_F32 = struct.Struct("<f")
type _HeaderValues = tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    float,
    float,
]
_HEADER_VALUES: TypeAdapter[_HeaderValues] = TypeAdapter(
    _HeaderValues,
    config=ConfigDict(strict=True),
)


@dataclass(frozen=True, slots=True)
class RolloutArenaHeader:
    """Shared counters for one worker-owned arena."""

    policy_version: int
    decision_count: int
    decision_capacity: int
    trajectory_count: int
    trajectory_capacity: int
    round_count: int
    round_capacity: int
    model_action_count: int
    auto_action_count: int
    forced_action_count: int
    trainable_decision_count: int
    scored_choice_step_count: int
    game_over_count: int
    rejected_round_count: int
    cancelled_env_count: int
    total_trace_step_count: int
    max_trace_step_count: int
    model_win_count: int
    auto_win_count: int
    model_declarer_round_count: int
    model_reward_sum: float
    auto_reward_sum: float
    elapsed_seconds_max: float

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert 0 <= self.decision_count <= self.decision_capacity
        assert self.decision_capacity > 0
        assert 0 <= self.trajectory_count <= self.trajectory_capacity
        assert self.trajectory_capacity > 0
        assert 0 <= self.round_count <= self.round_capacity
        assert self.round_capacity > 0
        assert self.model_action_count >= 0
        assert self.auto_action_count >= 0
        assert self.forced_action_count >= 0
        assert self.trainable_decision_count >= 0
        assert self.scored_choice_step_count >= 0
        assert self.game_over_count >= 0
        assert self.rejected_round_count >= 0
        assert self.cancelled_env_count >= 0
        assert self.total_trace_step_count >= 0
        assert self.max_trace_step_count >= 0
        assert self.model_win_count >= 0
        assert self.auto_win_count >= 0
        assert self.model_declarer_round_count >= 0


@dataclass(frozen=True, slots=True)
class RolloutColumnViews:
    """Direct byte views of committed trajectory columns."""

    row_indices: memoryview[int]
    step_counts: memoryview[int]
    trajectory_offsets: memoryview[int]
    trajectory_lengths: memoryview[int]
    terminal_rewards: memoryview[int]


def arena_byte_size(
    *, decision_capacity: int, trajectory_capacity: int
) -> int:
    """Return bytes required by one fixed-capacity arena."""
    assert decision_capacity > 0
    assert trajectory_capacity > 0
    return (
        _terminal_rewards_offset(
            decision_capacity=decision_capacity,
            trajectory_capacity=trajectory_capacity,
        )
        + trajectory_capacity * _F32.size
    )


def empty_header(
    *, policy_version: int, round_capacity: int
) -> RolloutArenaHeader:
    """Return an empty header for one update policy version."""
    assert round_capacity > 0
    return RolloutArenaHeader(
        policy_version=policy_version,
        decision_count=0,
        decision_capacity=(
            round_capacity * MAX_TRAINABLE_DECISIONS_PER_ROUND
        ),
        trajectory_count=0,
        trajectory_capacity=round_capacity * 2,
        round_count=0,
        round_capacity=round_capacity,
        model_action_count=0,
        auto_action_count=0,
        forced_action_count=0,
        trainable_decision_count=0,
        scored_choice_step_count=0,
        game_over_count=0,
        rejected_round_count=0,
        cancelled_env_count=0,
        total_trace_step_count=0,
        max_trace_step_count=0,
        model_win_count=0,
        auto_win_count=0,
        model_declarer_round_count=0,
        model_reward_sum=0.0,
        auto_reward_sum=0.0,
        elapsed_seconds_max=0.0,
    )


def pack_header(
    buffer: memoryview, *, header: RolloutArenaHeader
) -> None:
    """Write one header into shared memory."""
    _HEADER.pack_into(
        buffer,
        0,
        header.policy_version,
        header.decision_count,
        header.decision_capacity,
        header.trajectory_count,
        header.trajectory_capacity,
        header.round_count,
        header.round_capacity,
        header.model_action_count,
        header.auto_action_count,
        header.forced_action_count,
        header.trainable_decision_count,
        header.scored_choice_step_count,
        header.game_over_count,
        header.rejected_round_count,
        header.cancelled_env_count,
        header.total_trace_step_count,
        header.max_trace_step_count,
        header.model_win_count,
        header.auto_win_count,
        header.model_declarer_round_count,
        header.model_reward_sum,
        header.auto_reward_sum,
        header.elapsed_seconds_max,
    )


def unpack_header(buffer: memoryview) -> RolloutArenaHeader:
    """Read one header from shared memory."""
    values = _HEADER_VALUES.validate_python(
        _HEADER.unpack_from(buffer, 0)
    )
    return RolloutArenaHeader(
        policy_version=values[0],
        decision_count=values[1],
        decision_capacity=values[2],
        trajectory_count=values[3],
        trajectory_capacity=values[4],
        round_count=values[5],
        round_capacity=values[6],
        model_action_count=values[7],
        auto_action_count=values[8],
        forced_action_count=values[9],
        trainable_decision_count=values[10],
        scored_choice_step_count=values[11],
        game_over_count=values[12],
        rejected_round_count=values[13],
        cancelled_env_count=values[14],
        total_trace_step_count=values[15],
        max_trace_step_count=values[16],
        model_win_count=values[17],
        auto_win_count=values[18],
        model_declarer_round_count=values[19],
        model_reward_sum=values[20],
        auto_reward_sum=values[21],
        elapsed_seconds_max=values[22],
    )


def pack_decisions(
    *,
    buffer: memoryview,
    decision_capacity: int,
    start: int,
    row_indices: tuple[int, ...],
    step_counts: tuple[int, ...],
) -> None:
    """Write replay references for an atomic round."""
    assert len(row_indices) == len(step_counts)
    assert 0 <= start <= decision_capacity
    assert start + len(row_indices) <= decision_capacity
    assert all(value >= 0 for value in row_indices)
    assert all(value > 0 for value in step_counts)
    _pack_i64(
        buffer=buffer,
        offset=_row_indices_offset() + start * _I64.size,
        values=row_indices,
    )
    _pack_i64(
        buffer=buffer,
        offset=(
            _step_counts_offset(decision_capacity) + start * _I64.size
        ),
        values=step_counts,
    )


def pack_trajectories(
    *,
    buffer: memoryview,
    decision_capacity: int,
    trajectory_capacity: int,
    start: int,
    offsets: tuple[int, ...],
    lengths: tuple[int, ...],
    terminal_rewards: tuple[float, ...],
) -> None:
    """Write complete seat-trajectory descriptors."""
    count = len(offsets)
    assert len(lengths) == count
    assert len(terminal_rewards) == count
    assert 0 <= start <= trajectory_capacity
    assert start + count <= trajectory_capacity
    assert all(value >= 0 for value in offsets)
    assert all(value > 0 for value in lengths)
    _pack_i64(
        buffer=buffer,
        offset=(
            _trajectory_offsets_offset(decision_capacity)
            + start * _I64.size
        ),
        values=offsets,
    )
    _pack_i64(
        buffer=buffer,
        offset=(
            _trajectory_lengths_offset(
                decision_capacity=decision_capacity,
                trajectory_capacity=trajectory_capacity,
            )
            + start * _I64.size
        ),
        values=lengths,
    )
    _pack_f32(
        buffer=buffer,
        offset=(
            _terminal_rewards_offset(
                decision_capacity=decision_capacity,
                trajectory_capacity=trajectory_capacity,
            )
            + start * _F32.size
        ),
        values=terminal_rewards,
    )


def rollout_column_views(
    *, buffer: memoryview[int], header: RolloutArenaHeader
) -> RolloutColumnViews:
    """Return views of committed decision and trajectory rows."""
    decision_i64_bytes = header.decision_count * _I64.size
    trajectory_i64_bytes = header.trajectory_count * _I64.size
    trajectory_f32_bytes = header.trajectory_count * _F32.size
    return RolloutColumnViews(
        row_indices=_view(
            buffer,
            _row_indices_offset(),
            decision_i64_bytes,
        ),
        step_counts=_view(
            buffer,
            _step_counts_offset(header.decision_capacity),
            decision_i64_bytes,
        ),
        trajectory_offsets=_view(
            buffer,
            _trajectory_offsets_offset(header.decision_capacity),
            trajectory_i64_bytes,
        ),
        trajectory_lengths=_view(
            buffer,
            _trajectory_lengths_offset(
                decision_capacity=header.decision_capacity,
                trajectory_capacity=header.trajectory_capacity,
            ),
            trajectory_i64_bytes,
        ),
        terminal_rewards=_view(
            buffer,
            _terminal_rewards_offset(
                decision_capacity=header.decision_capacity,
                trajectory_capacity=header.trajectory_capacity,
            ),
            trajectory_f32_bytes,
        ),
    )


def _pack_i64(
    *, buffer: memoryview, offset: int, values: tuple[int, ...]
) -> None:
    if values:
        data = struct.pack(f"<{len(values)}q", *values)
        buffer[offset : offset + len(data)] = data


def _pack_f32(
    *, buffer: memoryview, offset: int, values: tuple[float, ...]
) -> None:
    if values:
        data = struct.pack(f"<{len(values)}f", *values)
        buffer[offset : offset + len(data)] = data


def _view(
    buffer: memoryview[int], offset: int, byte_count: int
) -> memoryview[int]:
    return buffer[offset : offset + byte_count]


def _row_indices_offset() -> int:
    return _HEADER.size


def _step_counts_offset(decision_capacity: int) -> int:
    return _row_indices_offset() + decision_capacity * _I64.size


def _trajectory_offsets_offset(decision_capacity: int) -> int:
    return _step_counts_offset(decision_capacity) + (
        decision_capacity * _I64.size
    )


def _trajectory_lengths_offset(
    *, decision_capacity: int, trajectory_capacity: int
) -> int:
    return _trajectory_offsets_offset(decision_capacity) + (
        trajectory_capacity * _I64.size
    )


def _terminal_rewards_offset(
    *, decision_capacity: int, trajectory_capacity: int
) -> int:
    return (
        _trajectory_lengths_offset(
            decision_capacity=decision_capacity,
            trajectory_capacity=trajectory_capacity,
        )
        + trajectory_capacity * _I64.size
    )


__all__ = (
    "MAX_TRAINABLE_DECISIONS_PER_ROUND",
    "RolloutArenaHeader",
    "RolloutColumnViews",
    "arena_byte_size",
    "empty_header",
    "pack_decisions",
    "pack_header",
    "pack_trajectories",
    "rollout_column_views",
    "unpack_header",
)
