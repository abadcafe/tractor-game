"""Columnar request frame encoding and decoding."""

from __future__ import annotations

from dataclasses import dataclass

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
from server.training.packed_observation import (
    MAX_LOSSLESS_OBSERVATION_TOKENS,
)
from server.training.policy_inference_batch.schema import (
    BATCH_CAPACITY_INDEX,
    HEADER_BYTES,
    I64,
    MAGIC_INDEX,
    MAX_PAIR_PLAN_COUNT,
    MAX_TRACE_COUNT,
    OBSERVATION_TOKEN_CAPACITY_INDEX,
    PADDED_GENERATION_STEPS_INDEX,
    PAIR_PLAN_COUNT_INDEX,
    REQUEST_BATCH_MAGIC,
    ROW_COUNT_INDEX,
    TOTAL_BYTES_INDEX,
    TRACE_COUNT_INDEX,
    PolicyRequestBatchLayout,
    policy_request_batch_layout,
)
from server.training.policy_inference_batch.types import (
    PolicyRequestFrameMetadata,
    PolicyRequestRoute,
)
from server.training.semantic_actions.choices import MAX_ACTION_STEPS

type ReadableFrameBuffer = bytes | bytearray | memoryview
type WritableFrameBuffer = bytearray | memoryview


@dataclass(frozen=True, slots=True)
class PolicyRequestFrameShape:
    """Static shape declared by a request frame header."""

    row_count: int
    batch_capacity: int
    observation_token_capacity: int
    padded_generation_steps: int
    total_bytes: int
    layout: PolicyRequestBatchLayout

    def __post_init__(self) -> None:
        assert self.row_count > 0
        assert self.batch_capacity >= self.row_count
        assert self.observation_token_capacity > 0
        assert self.padded_generation_steps > 0
        assert self.total_bytes > HEADER_BYTES
        assert self.layout.batch_capacity == self.batch_capacity
        assert self.layout.observation_token_capacity == (
            self.observation_token_capacity
        )
        assert self.layout.padded_generation_steps == (
            self.padded_generation_steps
        )
        assert self.layout.total_bytes == self.total_bytes


def initialize_policy_request_frame(
    buffer: WritableFrameBuffer,
    *,
    row_count: int,
    layout: PolicyRequestBatchLayout,
) -> None:
    """Write a request frame header into a caller-owned buffer."""
    assert row_count >= 0
    assert row_count <= layout.batch_capacity
    assert layout.padded_generation_steps <= (MAX_ACTION_STEPS)
    assert len(buffer) == layout.total_bytes
    words = (
        REQUEST_BATCH_MAGIC,
        layout.total_bytes,
        row_count,
        layout.batch_capacity,
        layout.observation_token_capacity,
        MAX_TRACE_COUNT,
        layout.padded_generation_steps,
        MAX_PAIR_PLAN_COUNT,
    )
    for index, word in enumerate(words):
        I64.pack_into(buffer, index * I64.size, word)


def decode_policy_request_frame_metadata(
    data: ReadableFrameBuffer,
) -> _result.Ok[PolicyRequestFrameMetadata] | _result.Rejected:
    """Decode and validate a columnar request frame header."""
    shape_result = decode_policy_request_frame_shape(data)
    if isinstance(shape_result, Rejected):
        return shape_result
    shape = shape_result.value
    layout = shape.layout
    routes: list[PolicyRequestRoute] = []
    policy_versions: list[int] = []
    generation_step_counts: list[int] = []
    for row_index in range(shape.row_count):
        worker_index = _read_column_i64(
            data, layout.route_worker_indices.offset, row_index
        )
        request_id = _read_column_i64(
            data, layout.route_request_ids.offset, row_index
        )
        policy_version = _read_column_i64(
            data, layout.policy_versions.offset, row_index
        )
        generation_step_count = _read_column_i64(
            data, layout.generation_step_counts.offset, row_index
        )
        if worker_index < 0 or request_id < 0:
            return Rejected(reason="policy request route is invalid")
        if policy_version < 0:
            return Rejected(reason="policy request version is invalid")
        if (
            generation_step_count <= 0
            or generation_step_count > shape.padded_generation_steps
        ):
            return Rejected(
                reason="policy request generation width is invalid"
            )
        routes.append(
            PolicyRequestRoute(
                worker_index=worker_index,
                request_id=request_id,
            )
        )
        policy_versions.append(policy_version)
        generation_step_counts.append(generation_step_count)
    return Ok(
        value=PolicyRequestFrameMetadata(
            row_count=shape.row_count,
            batch_capacity=shape.batch_capacity,
            observation_token_capacity=shape.observation_token_capacity,
            padded_generation_steps=shape.padded_generation_steps,
            generation_step_counts=tuple(generation_step_counts),
            routes=tuple(routes),
            policy_versions=tuple(policy_versions),
            byte_count=shape.total_bytes,
            layout=shape.layout,
        )
    )


def decode_policy_request_frame_shape(
    data: ReadableFrameBuffer,
) -> _result.Ok[PolicyRequestFrameShape] | _result.Rejected:
    """Decode and validate only the static frame shape."""
    if len(data) < HEADER_BYTES:
        return Rejected(reason="policy request frame is truncated")
    if _read_i64(data, MAGIC_INDEX) != REQUEST_BATCH_MAGIC:
        return Rejected(reason="policy request frame schema is invalid")
    total_bytes = _read_i64(data, TOTAL_BYTES_INDEX)
    if total_bytes != len(data):
        return Rejected(reason="policy request frame length mismatch")
    row_count = _read_i64(data, ROW_COUNT_INDEX)
    batch_capacity = _read_i64(data, BATCH_CAPACITY_INDEX)
    observation_token_capacity = _read_i64(
        data, OBSERVATION_TOKEN_CAPACITY_INDEX
    )
    trace_count = _read_i64(data, TRACE_COUNT_INDEX)
    padded_generation_steps = _read_i64(
        data, PADDED_GENERATION_STEPS_INDEX
    )
    pair_plan_count = _read_i64(data, PAIR_PLAN_COUNT_INDEX)
    if row_count <= 0:
        return Rejected(reason="policy request frame is empty")
    if batch_capacity <= 0 or row_count > batch_capacity:
        return Rejected(
            reason="policy request frame capacity is invalid"
        )
    if (
        observation_token_capacity <= 0
        or observation_token_capacity > MAX_LOSSLESS_OBSERVATION_TOKENS
    ):
        return Rejected(
            reason="policy request observation layout is invalid"
        )
    if trace_count != MAX_TRACE_COUNT:
        return Rejected(reason="policy request trace layout mismatch")
    if (
        padded_generation_steps <= 0
        or padded_generation_steps > MAX_ACTION_STEPS
    ):
        return Rejected(reason="policy request trace width mismatch")
    if pair_plan_count != MAX_PAIR_PLAN_COUNT:
        return Rejected(
            reason="policy request pair plan layout mismatch"
        )
    layout = policy_request_batch_layout(
        batch_capacity=batch_capacity,
        observation_token_capacity=observation_token_capacity,
        padded_generation_steps=padded_generation_steps,
    )
    if layout.total_bytes != total_bytes:
        return Rejected(reason="policy request frame layout mismatch")
    return Ok(
        value=PolicyRequestFrameShape(
            row_count=row_count,
            batch_capacity=batch_capacity,
            observation_token_capacity=observation_token_capacity,
            padded_generation_steps=padded_generation_steps,
            total_bytes=total_bytes,
            layout=layout,
        )
    )


def _read_column_i64(
    data: ReadableFrameBuffer, column_offset: int, row_index: int
) -> int:
    return _read_i64_at(data, column_offset + row_index * I64.size)


def _read_i64(data: ReadableFrameBuffer, word_index: int) -> int:
    return _read_i64_at(data, word_index * I64.size)


def _read_i64_at(data: ReadableFrameBuffer, offset: int) -> int:
    values = I64.unpack_from(data, offset)
    return int(values[0])
