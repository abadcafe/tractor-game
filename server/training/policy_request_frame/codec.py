"""Binary codec for policy request and response frames."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import cast

from server import result as _result
from server.result import Ok, Rejected
from server.training.policy_request_frame.frame import (
    CompletedPolicyResponseFrame,
    PolicyRequestBatchFrame,
    PolicyRequestFrame,
    PolicyResponseFrame,
    RejectedPolicyResponseFrame,
)
from server.training.policy_sampling import DecisionHandle
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_action_plan import ActionPlanFrame

_FLOAT32 = struct.Struct("<f")
_RESPONSE_COMPLETED = 1
_RESPONSE_REJECTED = 2


def encode_policy_request_frame(frame: PolicyRequestFrame) -> bytes:
    """Encode one request frame as binary bytes."""
    writer = _BinaryWriter()
    _write_int_rows(writer, frame.component_rows)
    _write_float_rows(writer, frame.numeric_value_rows)
    _write_float_rows(writer, frame.numeric_mask_rows)
    _write_action_plan_frame(writer, frame.action_plan)
    _write_decision_key(writer, frame.decision_key)
    return writer.bytes()


def decode_policy_request_frame(
    data: bytes,
) -> _result.Ok[PolicyRequestFrame] | _result.Rejected:
    """Decode one binary request frame."""
    reader = _BinaryReader(data=data)
    component_rows_result = _read_int_rows(reader)
    if isinstance(component_rows_result, Rejected):
        return component_rows_result
    numeric_values_result = _read_float_rows(reader)
    if isinstance(numeric_values_result, Rejected):
        return numeric_values_result
    numeric_masks_result = _read_float_rows(reader)
    if isinstance(numeric_masks_result, Rejected):
        return numeric_masks_result
    action_plan_result = _read_action_plan_frame(reader)
    if isinstance(action_plan_result, Rejected):
        return action_plan_result
    key_result = _read_decision_key(reader)
    if isinstance(key_result, Rejected):
        return key_result
    finish_result = reader.finish()
    if isinstance(finish_result, Rejected):
        return finish_result
    return Ok(
        value=PolicyRequestFrame(
            component_rows=component_rows_result.value,
            numeric_value_rows=numeric_values_result.value,
            numeric_mask_rows=numeric_masks_result.value,
            action_plan=action_plan_result.value,
            decision_key=key_result.value,
        )
    )


def encode_policy_request_batch_frame(
    batch: PolicyRequestBatchFrame,
) -> bytes:
    """Encode one request batch frame as binary bytes."""
    writer = _BinaryWriter()
    writer.write_i64(batch.batch_size())
    for frame in batch.frames:
        writer.write_bytes(encode_policy_request_frame(frame))
    return writer.bytes()


def decode_policy_request_batch_frame(
    data: bytes,
) -> _result.Ok[PolicyRequestBatchFrame] | _result.Rejected:
    """Decode one binary request batch frame."""
    reader = _BinaryReader(data=data)
    count_result = reader.read_i64()
    if isinstance(count_result, Rejected):
        return count_result
    count = count_result.value
    if count <= 0:
        return Rejected(reason="policy request batch is empty")
    frames: list[PolicyRequestFrame] = []
    for _ in range(count):
        frame_bytes_result = reader.read_bytes()
        if isinstance(frame_bytes_result, Rejected):
            return frame_bytes_result
        frame_result = decode_policy_request_frame(
            frame_bytes_result.value
        )
        if isinstance(frame_result, Rejected):
            return frame_result
        frames.append(frame_result.value)
    finish_result = reader.finish()
    if isinstance(finish_result, Rejected):
        return finish_result
    return Ok(value=PolicyRequestBatchFrame(frames=tuple(frames)))


def encode_policy_response_frame(
    frame: PolicyResponseFrame,
) -> bytes:
    """Encode one response frame as binary bytes."""
    writer = _BinaryWriter()
    if isinstance(frame, RejectedPolicyResponseFrame):
        writer.write_i64(_RESPONSE_REJECTED)
        writer.write_text(frame.reason)
        return writer.bytes()
    writer.write_i64(_RESPONSE_COMPLETED)
    _write_int_tuple(writer, frame.trace_token_ids)
    writer.write_i64(frame.decision_handle.model_rank_index)
    writer.write_i64(frame.decision_handle.policy_version)
    writer.write_i64(frame.decision_handle.slot_index)
    writer.write_i64(frame.decision_handle.slot_generation)
    writer.write_i64(frame.choice_count)
    return writer.bytes()


def decode_policy_response_frame(
    data: bytes,
) -> _result.Ok[PolicyResponseFrame] | _result.Rejected:
    """Decode one binary response frame."""
    reader = _BinaryReader(data=data)
    tag_result = reader.read_i64()
    if isinstance(tag_result, Rejected):
        return tag_result
    tag = tag_result.value
    if tag == _RESPONSE_REJECTED:
        reason_result = reader.read_text()
        if isinstance(reason_result, Rejected):
            return reason_result
        finish_result = reader.finish()
        if isinstance(finish_result, Rejected):
            return finish_result
        return Ok(
            value=RejectedPolicyResponseFrame(reason_result.value)
        )
    if tag != _RESPONSE_COMPLETED:
        return Rejected(reason="invalid policy response frame tag")
    trace_result = _read_int_tuple(reader)
    if isinstance(trace_result, Rejected):
        return trace_result
    rank_result = reader.read_i64()
    if isinstance(rank_result, Rejected):
        return rank_result
    version_result = reader.read_i64()
    if isinstance(version_result, Rejected):
        return version_result
    slot_result = reader.read_i64()
    if isinstance(slot_result, Rejected):
        return slot_result
    generation_result = reader.read_i64()
    if isinstance(generation_result, Rejected):
        return generation_result
    choice_count_result = reader.read_i64()
    if isinstance(choice_count_result, Rejected):
        return choice_count_result
    finish_result = reader.finish()
    if isinstance(finish_result, Rejected):
        return finish_result
    return Ok(
        value=CompletedPolicyResponseFrame(
            trace_token_ids=trace_result.value,
            decision_handle=DecisionHandle(
                model_rank_index=rank_result.value,
                policy_version=version_result.value,
                slot_index=slot_result.value,
                slot_generation=generation_result.value,
            ),
            choice_count=choice_count_result.value,
        )
    )


def _byte_parts() -> list[bytes]:
    return []


@dataclass(slots=True)
class _BinaryWriter:
    _parts: list[bytes] = field(default_factory=_byte_parts)

    def bytes(self) -> bytes:
        return b"".join(self._parts)

    def write_i64(self, value: int) -> None:
        self._parts.append(value.to_bytes(8, "little", signed=True))

    def write_bool(self, value: bool) -> None:
        self._parts.append(b"\x01" if value else b"\x00")

    def write_f32(self, value: float) -> None:
        self._parts.append(_FLOAT32.pack(value))

    def write_text(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.write_i64(len(encoded))
        self._parts.append(encoded)

    def write_bytes(self, value: bytes) -> None:
        self.write_i64(len(value))
        self._parts.append(value)


@dataclass(slots=True)
class _BinaryReader:
    data: bytes
    offset: int = 0

    def finish(self) -> Ok[None] | Rejected:
        if self.offset == len(self.data):
            return Ok(value=None)
        return Rejected(reason="policy frame has trailing bytes")

    def read_i64(self) -> Ok[int] | Rejected:
        if not self._has(8):
            return Rejected(reason="policy frame is truncated")
        value = int.from_bytes(
            self.data[self.offset : self.offset + 8],
            "little",
            signed=True,
        )
        self.offset += 8
        return Ok(value=value)

    def read_bool(self) -> Ok[bool] | Rejected:
        if not self._has(1):
            return Rejected(reason="policy frame is truncated")
        value = self.data[self.offset]
        self.offset += 1
        if value == 0:
            return Ok(value=False)
        if value == 1:
            return Ok(value=True)
        return Rejected(reason="policy frame contains invalid bool")

    def read_f32(self) -> Ok[float] | Rejected:
        if not self._has(_FLOAT32.size):
            return Rejected(reason="policy frame is truncated")
        values = cast(
            tuple[float],
            _FLOAT32.unpack_from(self.data, self.offset),
        )
        self.offset += _FLOAT32.size
        return Ok(value=float(values[0]))

    def read_text(self) -> Ok[str] | Rejected:
        length_result = self.read_i64()
        if isinstance(length_result, Rejected):
            return length_result
        length = length_result.value
        if length < 0:
            return Rejected(
                reason="policy frame text length is invalid"
            )
        if not self._has(length):
            return Rejected(reason="policy frame is truncated")
        try:
            value = self.data[
                self.offset : self.offset + length
            ].decode("utf-8")
        except UnicodeDecodeError:
            return Rejected(reason="policy frame text is invalid UTF-8")
        self.offset += length
        return Ok(value=value)

    def read_bytes(self) -> Ok[bytes] | Rejected:
        length_result = self.read_i64()
        if isinstance(length_result, Rejected):
            return length_result
        length = length_result.value
        if length < 0:
            return Rejected(
                reason="policy frame bytes length is invalid"
            )
        if not self._has(length):
            return Rejected(reason="policy frame is truncated")
        value = self.data[self.offset : self.offset + length]
        self.offset += length
        return Ok(value=value)

    def _has(self, size: int) -> bool:
        assert size >= 0
        return self.offset + size <= len(self.data)


def _write_decision_key(
    writer: _BinaryWriter, key: PolicyDecisionKey
) -> None:
    writer.write_i64(key.base_seed)
    writer.write_i64(key.policy_version)
    writer.write_i64(key.episode_id)
    writer.write_i64(key.player_index)
    writer.write_i64(key.decision_index)


def _read_decision_key(
    reader: _BinaryReader,
) -> Ok[PolicyDecisionKey] | Rejected:
    base_seed_result = reader.read_i64()
    if isinstance(base_seed_result, Rejected):
        return base_seed_result
    policy_version_result = reader.read_i64()
    if isinstance(policy_version_result, Rejected):
        return policy_version_result
    episode_id_result = reader.read_i64()
    if isinstance(episode_id_result, Rejected):
        return episode_id_result
    player_index_result = reader.read_i64()
    if isinstance(player_index_result, Rejected):
        return player_index_result
    decision_index_result = reader.read_i64()
    if isinstance(decision_index_result, Rejected):
        return decision_index_result
    return Ok(
        value=PolicyDecisionKey(
            base_seed=base_seed_result.value,
            policy_version=policy_version_result.value,
            episode_id=episode_id_result.value,
            player_index=player_index_result.value,
            decision_index=decision_index_result.value,
        )
    )


def _write_action_plan_frame(
    writer: _BinaryWriter, frame: ActionPlanFrame
) -> None:
    writer.write_i64(frame.kind_code)
    _write_int_tuple(writer, frame.available_counts)
    _write_int_tuple(writer, frame.effective_suits)
    _write_bool_tuple(writer, frame.same_suit_mask)
    _write_bool_tuple(writer, frame.off_suit_mask)
    _write_bool_tuple(writer, frame.pair_face_mask)
    writer.write_i64(frame.min_select)
    writer.write_i64(frame.max_select)
    writer.write_i64(frame.exact_select)
    writer.write_i64(frame.required_same_suit_count)
    writer.write_i64(frame.pair_floor)
    writer.write_bool(frame.has_tractor)
    _write_int_rows(writer, frame.trace_tokens)
    _write_bool_rows(writer, frame.pair_plan_masks)


def _read_action_plan_frame(
    reader: _BinaryReader,
) -> Ok[ActionPlanFrame] | Rejected:
    kind_result = reader.read_i64()
    if isinstance(kind_result, Rejected):
        return kind_result
    available_result = _read_int_tuple(reader)
    if isinstance(available_result, Rejected):
        return available_result
    suits_result = _read_int_tuple(reader)
    if isinstance(suits_result, Rejected):
        return suits_result
    same_result = _read_bool_tuple(reader)
    if isinstance(same_result, Rejected):
        return same_result
    off_result = _read_bool_tuple(reader)
    if isinstance(off_result, Rejected):
        return off_result
    pair_face_result = _read_bool_tuple(reader)
    if isinstance(pair_face_result, Rejected):
        return pair_face_result
    min_result = reader.read_i64()
    if isinstance(min_result, Rejected):
        return min_result
    max_result = reader.read_i64()
    if isinstance(max_result, Rejected):
        return max_result
    exact_result = reader.read_i64()
    if isinstance(exact_result, Rejected):
        return exact_result
    same_count_result = reader.read_i64()
    if isinstance(same_count_result, Rejected):
        return same_count_result
    pair_floor_result = reader.read_i64()
    if isinstance(pair_floor_result, Rejected):
        return pair_floor_result
    has_tractor_result = reader.read_bool()
    if isinstance(has_tractor_result, Rejected):
        return has_tractor_result
    trace_result = _read_int_rows(reader)
    if isinstance(trace_result, Rejected):
        return trace_result
    pair_plan_result = _read_bool_rows(reader)
    if isinstance(pair_plan_result, Rejected):
        return pair_plan_result
    return Ok(
        value=ActionPlanFrame(
            kind_code=kind_result.value,
            available_counts=available_result.value,
            effective_suits=suits_result.value,
            same_suit_mask=same_result.value,
            off_suit_mask=off_result.value,
            pair_face_mask=pair_face_result.value,
            min_select=min_result.value,
            max_select=max_result.value,
            exact_select=exact_result.value,
            required_same_suit_count=same_count_result.value,
            pair_floor=pair_floor_result.value,
            has_tractor=has_tractor_result.value,
            trace_tokens=trace_result.value,
            pair_plan_masks=pair_plan_result.value,
        )
    )


def _write_int_tuple(
    writer: _BinaryWriter, values: tuple[int, ...]
) -> None:
    writer.write_i64(len(values))
    for value in values:
        writer.write_i64(value)


def _read_int_tuple(
    reader: _BinaryReader,
) -> Ok[tuple[int, ...]] | Rejected:
    length_result = reader.read_i64()
    if isinstance(length_result, Rejected):
        return length_result
    length = length_result.value
    if length < 0:
        return Rejected(reason="policy frame int length is invalid")
    values: list[int] = []
    for _ in range(length):
        value_result = reader.read_i64()
        if isinstance(value_result, Rejected):
            return value_result
        values.append(value_result.value)
    return Ok(value=tuple(values))


def _write_bool_tuple(
    writer: _BinaryWriter, values: tuple[bool, ...]
) -> None:
    writer.write_i64(len(values))
    for value in values:
        writer.write_bool(value)


def _read_bool_tuple(
    reader: _BinaryReader,
) -> Ok[tuple[bool, ...]] | Rejected:
    length_result = reader.read_i64()
    if isinstance(length_result, Rejected):
        return length_result
    length = length_result.value
    if length < 0:
        return Rejected(reason="policy frame bool length is invalid")
    values: list[bool] = []
    for _ in range(length):
        value_result = reader.read_bool()
        if isinstance(value_result, Rejected):
            return value_result
        values.append(value_result.value)
    return Ok(value=tuple(values))


def _write_int_rows(
    writer: _BinaryWriter, rows: tuple[tuple[int, ...], ...]
) -> None:
    writer.write_i64(len(rows))
    for row in rows:
        _write_int_tuple(writer, row)


def _read_int_rows(
    reader: _BinaryReader,
) -> Ok[tuple[tuple[int, ...], ...]] | Rejected:
    length_result = reader.read_i64()
    if isinstance(length_result, Rejected):
        return length_result
    length = length_result.value
    if length < 0:
        return Rejected(reason="policy frame row count is invalid")
    rows: list[tuple[int, ...]] = []
    for _ in range(length):
        row_result = _read_int_tuple(reader)
        if isinstance(row_result, Rejected):
            return row_result
        rows.append(row_result.value)
    return Ok(value=tuple(rows))


def _write_bool_rows(
    writer: _BinaryWriter, rows: tuple[tuple[bool, ...], ...]
) -> None:
    writer.write_i64(len(rows))
    for row in rows:
        _write_bool_tuple(writer, row)


def _read_bool_rows(
    reader: _BinaryReader,
) -> Ok[tuple[tuple[bool, ...], ...]] | Rejected:
    length_result = reader.read_i64()
    if isinstance(length_result, Rejected):
        return length_result
    length = length_result.value
    if length < 0:
        return Rejected(reason="policy frame row count is invalid")
    rows: list[tuple[bool, ...]] = []
    for _ in range(length):
        row_result = _read_bool_tuple(reader)
        if isinstance(row_result, Rejected):
            return row_result
        rows.append(row_result.value)
    return Ok(value=tuple(rows))


def _write_float_rows(
    writer: _BinaryWriter, rows: tuple[tuple[float, ...], ...]
) -> None:
    writer.write_i64(len(rows))
    for row in rows:
        writer.write_i64(len(row))
        for value in row:
            writer.write_f32(value)


def _read_float_rows(
    reader: _BinaryReader,
) -> Ok[tuple[tuple[float, ...], ...]] | Rejected:
    length_result = reader.read_i64()
    if isinstance(length_result, Rejected):
        return length_result
    length = length_result.value
    if length < 0:
        return Rejected(reason="policy frame row count is invalid")
    rows: list[tuple[float, ...]] = []
    for _ in range(length):
        row_length_result = reader.read_i64()
        if isinstance(row_length_result, Rejected):
            return row_length_result
        row_length = row_length_result.value
        if row_length < 0:
            return Rejected(
                reason="policy frame float length is invalid"
            )
        row: list[float] = []
        for _ in range(row_length):
            value_result = reader.read_f32()
            if isinstance(value_result, Rejected):
                return value_result
            row.append(value_result.value)
        rows.append(tuple(row))
    return Ok(value=tuple(rows))
