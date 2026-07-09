"""Binary layout for shared rollout arenas."""

from __future__ import annotations

import struct
from dataclasses import dataclass

HEADER_STRUCT = struct.Struct("<qqqqqqqqqqqqqqddd")
_I64 = struct.Struct("<q")
_F32 = struct.Struct("<f")


@dataclass(frozen=True, slots=True)
class RolloutArenaHeader:
    """Shared counters for one worker-owned rollout arena."""

    policy_version: int
    sample_count: int
    capacity: int
    target_sample_count: int
    full: bool
    round_count: int
    generated_action_count: int
    accepted_action_count: int
    action_choice_count: int
    game_over_count: int
    dropped_sample_count: int
    cancelled_env_count: int
    total_step_count: int
    max_step_count: int
    team0_reward_sum: float
    team1_reward_sum: float
    elapsed_seconds_max: float

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.sample_count >= 0
        assert self.capacity > 0
        assert self.target_sample_count > 0
        assert self.target_sample_count <= self.capacity
        assert self.sample_count <= self.capacity
        assert self.round_count >= 0
        assert self.generated_action_count >= 0
        assert self.accepted_action_count >= 0
        assert self.action_choice_count >= 0
        assert self.game_over_count >= 0
        assert self.dropped_sample_count >= 0
        assert self.cancelled_env_count >= 0
        assert self.total_step_count >= 0
        assert self.max_step_count >= 0


@dataclass(frozen=True, slots=True)
class SampleReferenceColumnViews:
    """Byte views for committed sample reference columns."""

    row_indices: memoryview[int]
    step_counts: memoryview[int]
    return_values: memoryview[int]


@dataclass(frozen=True, slots=True)
class SampleReferenceColumnWriter:
    """Append sample reference columns into one arena range."""

    buffer: memoryview[int]
    capacity: int
    start_index: int
    count: int

    def __post_init__(self) -> None:
        assert self.capacity > 0
        assert self.start_index >= 0
        assert self.count >= 0
        assert self.start_index + self.count <= self.capacity

    def write(
        self,
        *,
        row_indices: tuple[int, ...],
        step_counts: tuple[int, ...],
        return_values: tuple[float, ...],
    ) -> None:
        """Write all sample reference columns for this range."""
        assert len(row_indices) == self.count
        assert len(step_counts) == self.count
        assert len(return_values) == self.count
        assert all(value >= 0 for value in row_indices)
        assert all(value > 0 for value in step_counts)
        self.write_row_indices(row_indices)
        self.write_step_counts(step_counts)
        self.write_return_values(return_values)

    def write_row_indices(self, values: tuple[int, ...]) -> None:
        """Write replay row indices."""
        assert len(values) == self.count
        _pack_i64_values(
            buffer=self.buffer,
            offset=(
                _row_indices_offset(self.capacity)
                + self.start_index * _I64.size
            ),
            values=values,
        )

    def write_step_counts(self, values: tuple[int, ...]) -> None:
        """Write replay step counts."""
        assert len(values) == self.count
        _pack_i64_values(
            buffer=self.buffer,
            offset=(
                _step_counts_offset(self.capacity)
                + self.start_index * _I64.size
            ),
            values=values,
        )

    def write_return_values(self, values: tuple[float, ...]) -> None:
        """Write return values."""
        assert len(values) == self.count
        _pack_f32_values(
            buffer=self.buffer,
            offset=(
                _return_values_offset(self.capacity)
                + self.start_index * _F32.size
            ),
            values=values,
        )


def arena_byte_size(*, capacity: int) -> int:
    """Return the shared memory bytes needed for one arena."""
    assert capacity > 0
    return _return_values_offset(capacity) + capacity * _F32.size


def pack_sample_references(
    *,
    buffer: memoryview,
    capacity: int,
    start_index: int,
    row_indices: tuple[int, ...],
    step_counts: tuple[int, ...],
    return_values: tuple[float, ...],
) -> None:
    """Write sample reference columns into one arena range."""
    assert capacity > 0
    assert start_index >= 0
    count = len(return_values)
    assert len(row_indices) == count
    assert len(step_counts) == count
    assert start_index + count <= capacity
    SampleReferenceColumnWriter(
        buffer=buffer,
        capacity=capacity,
        start_index=start_index,
        count=count,
    ).write(
        row_indices=row_indices,
        step_counts=step_counts,
        return_values=return_values,
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
        header.target_sample_count,
        1 if header.full else 0,
        header.round_count,
        header.generated_action_count,
        header.accepted_action_count,
        header.action_choice_count,
        header.game_over_count,
        header.dropped_sample_count,
        header.cancelled_env_count,
        header.total_step_count,
        header.max_step_count,
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
        target_sample_count=values[3],
        full=values[4] != 0,
        round_count=values[5],
        generated_action_count=values[6],
        accepted_action_count=values[7],
        action_choice_count=values[8],
        game_over_count=values[9],
        dropped_sample_count=values[10],
        cancelled_env_count=values[11],
        total_step_count=values[12],
        max_step_count=values[13],
        team0_reward_sum=values[14],
        team1_reward_sum=values[15],
        elapsed_seconds_max=values[16],
    )


def empty_header(
    *, policy_version: int, capacity: int, target_sample_count: int
) -> RolloutArenaHeader:
    """Return an empty arena header for one policy version."""
    return RolloutArenaHeader(
        policy_version=policy_version,
        sample_count=0,
        capacity=capacity,
        target_sample_count=target_sample_count,
        full=False,
        round_count=0,
        generated_action_count=0,
        accepted_action_count=0,
        action_choice_count=0,
        game_over_count=0,
        dropped_sample_count=0,
        cancelled_env_count=0,
        total_step_count=0,
        max_step_count=0,
        team0_reward_sum=0.0,
        team1_reward_sum=0.0,
        elapsed_seconds_max=0.0,
    )


def sample_reference_column_views(
    *, buffer: memoryview[int], capacity: int, count: int
) -> SampleReferenceColumnViews:
    """Return direct byte views for committed sample columns."""
    assert capacity > 0
    assert count >= 0
    assert count <= capacity
    i64_bytes = count * _I64.size
    f32_bytes = count * _F32.size
    return SampleReferenceColumnViews(
        row_indices=_column_view(
            buffer=buffer,
            offset=_row_indices_offset(capacity),
            byte_count=i64_bytes,
        ),
        step_counts=_column_view(
            buffer=buffer,
            offset=_step_counts_offset(capacity),
            byte_count=i64_bytes,
        ),
        return_values=_column_view(
            buffer=buffer,
            offset=_return_values_offset(capacity),
            byte_count=f32_bytes,
        ),
    )


def unpack_row_indices(
    *, buffer: memoryview, capacity: int, count: int
) -> tuple[int, ...]:
    """Read committed sample row indices from the arena columns."""
    assert count >= 0
    assert count <= capacity
    return _unpack_i64_values(
        buffer=buffer,
        offset=_row_indices_offset(capacity),
        count=count,
    )


def unpack_step_counts(
    *, buffer: memoryview, capacity: int, count: int
) -> tuple[int, ...]:
    """Read committed replay step counts from the arena columns."""
    assert count >= 0
    assert count <= capacity
    return _unpack_i64_values(
        buffer=buffer,
        offset=_step_counts_offset(capacity),
        count=count,
    )


def unpack_return_values(
    *, buffer: memoryview, capacity: int, count: int
) -> tuple[float, ...]:
    """Read committed return values from the arena columns."""
    assert count >= 0
    assert count <= capacity
    return _unpack_f32_values(
        buffer=buffer,
        offset=_return_values_offset(capacity),
        count=count,
    )


def _pack_i64_values(
    *, buffer: memoryview, offset: int, values: tuple[int, ...]
) -> None:
    if not values:
        return
    data = struct.pack(f"<{len(values)}q", *values)
    buffer[offset : offset + len(data)] = data


def _pack_f32_values(
    *, buffer: memoryview, offset: int, values: tuple[float, ...]
) -> None:
    if not values:
        return
    data = struct.pack(f"<{len(values)}f", *values)
    buffer[offset : offset + len(data)] = data


def _unpack_i64_values(
    *, buffer: memoryview, offset: int, count: int
) -> tuple[int, ...]:
    if count == 0:
        return ()
    values = struct.unpack_from(f"<{count}q", buffer, offset)
    return tuple(int(value) for value in values)


def _unpack_f32_values(
    *, buffer: memoryview, offset: int, count: int
) -> tuple[float, ...]:
    if count == 0:
        return ()
    values = struct.unpack_from(f"<{count}f", buffer, offset)
    return tuple(float(value) for value in values)


def _column_view(
    *, buffer: memoryview[int], offset: int, byte_count: int
) -> memoryview[int]:
    assert offset >= 0
    assert byte_count >= 0
    return buffer[offset : offset + byte_count]


def _row_indices_offset(capacity: int) -> int:
    assert capacity > 0
    return HEADER_STRUCT.size


def _return_values_offset(capacity: int) -> int:
    return _step_counts_offset(capacity) + capacity * _I64.size


def _step_counts_offset(capacity: int) -> int:
    return _row_indices_offset(capacity) + capacity * _I64.size
