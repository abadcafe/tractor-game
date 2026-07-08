"""Model-rank policy request staging."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch import Tensor

from server import result as _result
from server.result import Ok, Rejected
from server.training.policy_inference_wire import (
    DevicePolicyRequestBatch,
    DevicePolicyRequestBuffer,
    PolicyRequestMetadata,
    PolicyRequestRoute,
    PolicyRequestWireBatch,
    allocate_device_policy_request_buffer,
    decode_policy_request_metadata,
    max_policy_request_wire_bytes,
    unpack_policy_request_batch_into,
)
from server.training.runtime.model_rank.inference_transport import (
    ConnectionPolicyRequestReceiver,
)


@dataclass(frozen=True, slots=True)
class StagedPolicyRequestBatch:
    """One staged policy request batch ready for model inference."""

    routes: tuple[PolicyRequestRoute, ...]
    device_batch: DevicePolicyRequestBatch
    wire_byte_count: int
    recv_seconds: float
    h2d_seconds: float
    device_decode_seconds: float

    def __post_init__(self) -> None:
        assert self.routes
        assert len(self.routes) == len(
            self.device_batch.policy_versions
        )
        assert self.wire_byte_count > 0
        assert self.recv_seconds >= 0.0
        assert self.h2d_seconds >= 0.0
        assert self.device_decode_seconds >= 0.0

    def batch_size(self) -> int:
        """Return request count."""
        return len(self.routes)


@dataclass(slots=True)
class PolicyRequestStager:
    """Stage caller-selected request wires into reusable host slots."""

    batch_size: int
    max_observation_tokens: int
    device: torch.device
    _max_wire_bytes: int = field(init=False)
    _host_slots: Tensor = field(init=False)
    _device_slots: tuple[Tensor, Tensor] | None = field(init=False)
    _device_request_buffers: tuple[
        DevicePolicyRequestBuffer, DevicePolicyRequestBuffer
    ] = field(init=False)
    _device_slot_index: int = field(init=False)
    _device_buffer_index: int = field(init=False)
    _metadata: list[PolicyRequestMetadata] = field(init=False)
    _recv_start: float = field(init=False)

    def __post_init__(self) -> None:
        assert self.batch_size > 0
        assert self.max_observation_tokens > 0
        self._max_wire_bytes = max_policy_request_wire_bytes(
            max_observation_tokens=self.max_observation_tokens
        )
        self._host_slots = _allocate_host_slots(
            batch_size=self.batch_size,
            max_wire_bytes=self._max_wire_bytes,
            device=self.device,
        )
        self._device_slots = _allocate_device_slots(
            batch_size=self.batch_size,
            max_wire_bytes=self._max_wire_bytes,
            device=self.device,
        )
        self._device_request_buffers = _allocate_device_request_buffers(
            batch_size=self.batch_size,
            max_observation_tokens=self.max_observation_tokens,
            device=self.device,
        )
        self._device_slot_index = 0
        self._device_buffer_index = 0
        self._metadata = []
        self._recv_start = 0.0

    def begin_batch(self) -> None:
        """Start receiving one data-plane batch."""
        assert not self._metadata
        self._recv_start = time.perf_counter()

    def can_receive(self) -> bool:
        """Return whether this batch has an unused host slot."""
        return len(self._metadata) < self.batch_size

    def receive_from(
        self, receiver: ConnectionPolicyRequestReceiver
    ) -> _result.Ok[None] | _result.Rejected:
        """Receive one ready request into the next host slot."""
        assert self.can_receive()
        row_index = len(self._metadata)
        row_view = _host_slot_view(self._host_slots[row_index])
        byte_count_result = receiver.receive_bytes_into(row_view)
        if isinstance(byte_count_result, Rejected):
            return byte_count_result
        byte_count = byte_count_result.value
        if byte_count > self._max_wire_bytes:
            return Rejected(reason="policy request wire exceeds slot")
        metadata_result = decode_policy_request_metadata(
            row_view[:byte_count]
        )
        if isinstance(metadata_result, Rejected):
            return metadata_result
        if (
            metadata_result.value.token_count
            > self.max_observation_tokens
        ):
            return Rejected(
                reason="policy request observation exceeds token budget"
            )
        self._metadata.append(metadata_result.value)
        return Ok(value=None)

    def finish_batch(
        self,
    ) -> _result.Ok[StagedPolicyRequestBatch] | _result.Rejected:
        """Stage the received host slots into device tensors."""
        assert self._metadata
        metadata = tuple(self._metadata)
        self._metadata.clear()
        recv_seconds = time.perf_counter() - self._recv_start
        staged_result = _stage_host_slots(
            host_slots=self._host_slots,
            device_slot=self._next_device_slot(),
            output_buffer=self._next_device_request_buffer(),
            metadata=metadata,
            device=self.device,
            recv_seconds=recv_seconds,
        )
        if isinstance(staged_result, Rejected):
            return staged_result
        return Ok(value=staged_result.value)

    def discard_batch(self) -> None:
        """Drop partially received metadata after a staging error."""
        self._metadata.clear()

    def _next_device_slot(self) -> Tensor | None:
        slots = self._device_slots
        if slots is None:
            return None
        slot = slots[self._device_slot_index]
        self._device_slot_index = 1 - self._device_slot_index
        return slot

    def _next_device_request_buffer(self) -> DevicePolicyRequestBuffer:
        buffer = self._device_request_buffers[self._device_buffer_index]
        self._device_buffer_index = 1 - self._device_buffer_index
        return buffer


def stage_policy_request_wires(
    *,
    requests: PolicyRequestWireBatch,
    max_observation_tokens: int,
    device: torch.device,
) -> _result.Ok[StagedPolicyRequestBatch] | _result.Rejected:
    """Stage already-built request wires through the same boundary."""
    max_wire_bytes = max_policy_request_wire_bytes(
        max_observation_tokens=max_observation_tokens
    )
    host_slots = _allocate_host_slots(
        batch_size=requests.batch_size(),
        max_wire_bytes=max_wire_bytes,
        device=device,
    )
    device_slots = _allocate_device_slots(
        batch_size=requests.batch_size(),
        max_wire_bytes=max_wire_bytes,
        device=device,
    )
    output_buffer = allocate_device_policy_request_buffer(
        batch_size=requests.batch_size(),
        max_observation_tokens=max_observation_tokens,
        device=device,
    )
    metadata: list[PolicyRequestMetadata] = []
    for index, request in enumerate(requests.requests):
        if request.byte_count() > max_wire_bytes:
            return Rejected(reason="policy request wire exceeds slot")
        source = torch.frombuffer(
            bytearray(request.data), dtype=torch.uint8
        )
        host_slots[index, : request.byte_count()].copy_(source)
        metadata_result = decode_policy_request_metadata(request.data)
        if isinstance(metadata_result, Rejected):
            return metadata_result
        if metadata_result.value.token_count > max_observation_tokens:
            return Rejected(
                reason="policy request observation exceeds token budget"
            )
        metadata.append(metadata_result.value)
    return _stage_host_slots(
        host_slots=host_slots,
        device_slot=None if device_slots is None else device_slots[0],
        output_buffer=output_buffer,
        metadata=tuple(metadata),
        device=device,
        recv_seconds=0.0,
    )


def _stage_host_slots(
    *,
    host_slots: Tensor,
    device_slot: Tensor | None,
    output_buffer: DevicePolicyRequestBuffer,
    metadata: tuple[PolicyRequestMetadata, ...],
    device: torch.device,
    recv_seconds: float,
) -> _result.Ok[StagedPolicyRequestBatch] | _result.Rejected:
    row_count = len(metadata)
    assert row_count > 0
    max_received_bytes = max(item.byte_count for item in metadata)
    h2d_start = time.perf_counter()
    device_bytes = _copy_host_to_device(
        host_slots=host_slots[:row_count, :max_received_bytes],
        device_slot=(
            None
            if device_slot is None
            else device_slot[:row_count, :max_received_bytes]
        ),
        device=device,
    )
    h2d_seconds = time.perf_counter() - h2d_start
    decode_start = time.perf_counter()
    device_batch_result = unpack_policy_request_batch_into(
        device_bytes=device_bytes,
        metadata=metadata,
        output=output_buffer,
    )
    device_decode_seconds = time.perf_counter() - decode_start
    if isinstance(device_batch_result, Rejected):
        return device_batch_result
    return Ok(
        value=StagedPolicyRequestBatch(
            routes=tuple(item.route for item in metadata),
            device_batch=device_batch_result.value,
            wire_byte_count=sum(item.byte_count for item in metadata),
            recv_seconds=recv_seconds,
            h2d_seconds=h2d_seconds,
            device_decode_seconds=device_decode_seconds,
        )
    )


def _allocate_host_slots(
    *,
    batch_size: int,
    max_wire_bytes: int,
    device: torch.device,
) -> Tensor:
    assert batch_size > 0
    assert max_wire_bytes > 0
    if device.type == "cuda":
        return torch.empty(
            (batch_size, max_wire_bytes),
            dtype=torch.uint8,
            device=torch.device("cpu"),
            pin_memory=True,
        )
    return torch.empty(
        (batch_size, max_wire_bytes),
        dtype=torch.uint8,
        device=torch.device("cpu"),
    )


def _allocate_device_slots(
    *,
    batch_size: int,
    max_wire_bytes: int,
    device: torch.device,
) -> tuple[Tensor, Tensor] | None:
    assert batch_size > 0
    assert max_wire_bytes > 0
    if device.type == "cpu":
        return None
    return (
        torch.empty(
            (batch_size, max_wire_bytes),
            dtype=torch.uint8,
            device=device,
        ),
        torch.empty(
            (batch_size, max_wire_bytes),
            dtype=torch.uint8,
            device=device,
        ),
    )


def _allocate_device_request_buffers(
    *,
    batch_size: int,
    max_observation_tokens: int,
    device: torch.device,
) -> tuple[DevicePolicyRequestBuffer, DevicePolicyRequestBuffer]:
    assert batch_size > 0
    assert max_observation_tokens > 0
    return (
        allocate_device_policy_request_buffer(
            batch_size=batch_size,
            max_observation_tokens=max_observation_tokens,
            device=device,
        ),
        allocate_device_policy_request_buffer(
            batch_size=batch_size,
            max_observation_tokens=max_observation_tokens,
            device=device,
        ),
    )


def _copy_host_to_device(
    *,
    host_slots: Tensor,
    device_slot: Tensor | None,
    device: torch.device,
) -> Tensor:
    if device.type == "cpu":
        return host_slots
    if device.type != "cuda":
        assert device_slot is not None
        device_slot.copy_(host_slots)
        return device_slot
    assert device_slot is not None
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        device_slot.copy_(host_slots, non_blocking=True)
    event = torch.cuda.Event()
    event.record(stream)
    torch.cuda.current_stream(device).wait_event(event)
    return device_slot


def _host_slot_view(slot: Tensor) -> memoryview:
    assert slot.device.type == "cpu"
    return memoryview(slot.numpy())
