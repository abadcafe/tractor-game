"""Model-rank policy request ingress and final batch assembly."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch import Tensor

from server import result as _result
from server.result import Ok, Rejected
from server.training.policy_inference_batch import (
    DevicePolicyRequestBatch,
    PolicyRequestRoute,
)
from server.training.policy_inference_batch.device import (
    copy_policy_request_host_frame_to_device,
    materialize_policy_request_frame,
)
from server.training.policy_inference_batch.frame import (
    decode_policy_request_frame_metadata,
)
from server.training.policy_inference_batch.schema import (
    max_policy_request_batch_frame_bytes,
)
from server.training.policy_inference_batch.types import (
    PolicyRequestFrameMetadata,
)
from server.training.runtime.model_rank.inference_transport import (
    AsyncPolicyPeer,
)
from server.training.semantic_action_plan import DeviceActionPlanBatch
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tensorize import ObservationTensorBatch


@dataclass(frozen=True, slots=True)
class ModelRankInferenceBatch:
    """One ingress batch ready for model inference."""

    routes: tuple[PolicyRequestRoute, ...]
    device_batch: DevicePolicyRequestBatch
    generation_step_counts: tuple[int, ...]
    wire_byte_count: int
    recv_seconds: float
    h2d_seconds: float
    device_decode_seconds: float
    frame_count: int
    shape_bucket_count: int = 1
    shape_padding_tokens_saved: int = 0

    def __post_init__(self) -> None:
        assert self.routes
        assert len(self.routes) == len(
            self.device_batch.policy_versions
        )
        assert len(self.generation_step_counts) == len(self.routes)
        assert self.wire_byte_count > 0
        assert self.recv_seconds >= 0.0
        assert self.h2d_seconds >= 0.0
        assert self.device_decode_seconds >= 0.0
        assert self.frame_count > 0
        assert self.shape_bucket_count > 0
        assert self.shape_padding_tokens_saved >= 0

    def batch_size(self) -> int:
        """Return request count."""
        return len(self.routes)


@dataclass(frozen=True, slots=True)
class _ReceivedPolicyRequestFrame:
    slot: _PinnedHostFrameSlot
    metadata: PolicyRequestFrameMetadata
    routes: tuple[PolicyRequestRoute, ...]
    device_batch: DevicePolicyRequestBatch
    generation_step_counts: tuple[int, ...]
    wire_byte_count: int
    recv_seconds: float
    device_decode_seconds: float
    frame_count: int

    def row_count(self) -> int:
        return len(self.routes)

    def padded_generation_steps(self) -> int:
        return self.device_batch.padded_generation_steps


def _host_frame_slot_list() -> list[_PinnedHostFrameSlot]:
    return []


def _route_list() -> list[PolicyRequestRoute]:
    return []


def _policy_version_list() -> list[int]:
    return []


def _generation_step_count_list() -> list[int]:
    return []


@dataclass(slots=True)
class _PinnedHostFrameSlot:
    tensor: Tensor
    pending_event: torch.cuda.Event | None = None

    def wait_ready(self) -> None:
        event = self.pending_event
        if event is None:
            return
        event.synchronize()
        self.pending_event = None


@dataclass(slots=True)
class PolicyRequestIngress:
    """Receive request frames and assemble one final device batch."""

    batch_size: int
    max_observation_tokens: int
    device: torch.device
    _max_frame_bytes: int = field(init=False)
    _host_slots: list[_PinnedHostFrameSlot] = field(
        default_factory=_host_frame_slot_list
    )
    _next_slot_index: int = field(init=False)
    _builder: _FinalPolicyRequestBatchBuilder | None = field(init=False)
    _workspace: _FinalPolicyRequestBatchBuilder | None = field(
        init=False
    )
    _pending_frame: _ReceivedPolicyRequestFrame | None = field(
        init=False
    )
    _single_frame: _ReceivedPolicyRequestFrame | None = field(
        init=False
    )
    _recv_start: float = field(init=False)

    def __post_init__(self) -> None:
        assert self.batch_size > 0
        assert self.max_observation_tokens > 0
        self._max_frame_bytes = max_policy_request_batch_frame_bytes(
            batch_capacity=self.batch_size,
            max_observation_tokens=self.max_observation_tokens,
            padded_generation_steps=SEMANTIC_CODEC.max_argument_tokens,
        )
        self._next_slot_index = 0
        self._builder = None
        self._workspace = None
        self._pending_frame = None
        self._single_frame = None
        self._recv_start = 0.0

    def begin_batch(self) -> None:
        """Start receiving one data-plane batch."""
        assert self._builder is None
        assert self._single_frame is None
        self._next_slot_index = 0
        self._recv_start = time.perf_counter()

    def can_receive(self) -> bool:
        """Return whether this batch has room for at least one row."""
        if self._active_row_count() >= self.batch_size:
            return False
        return self._pending_frame is None

    def has_pending_rows(self) -> bool:
        """Return whether a previously received frame still has rows."""
        return self._pending_frame is not None

    async def receive_from(
        self, peer: AsyncPolicyPeer
    ) -> _result.Ok[None] | _result.Rejected:
        """Receive one ready request frame and append active rows."""
        assert self.can_receive()
        if self.has_pending_rows():
            return self.drain_pending_rows()
        slot = self._acquire_slot()
        frame_view = _host_frame_view(slot.tensor)
        byte_count_result = await peer.receive_request_into(frame_view)
        if isinstance(byte_count_result, Rejected):
            return byte_count_result
        byte_count = byte_count_result.value
        if byte_count > self._max_frame_bytes:
            return Rejected(reason="policy request frame exceeds slot")
        received = frame_view[:byte_count]
        metadata_result = decode_policy_request_frame_metadata(received)
        if isinstance(metadata_result, Rejected):
            return metadata_result
        metadata = metadata_result.value
        validation_result = self._validate_frame_metadata(metadata)
        if isinstance(validation_result, Rejected):
            return validation_result
        frame_result = self._stage_host_frame(
            slot=slot,
            byte_count=byte_count,
            metadata=metadata,
            recv_seconds=time.perf_counter() - self._recv_start,
            wire_byte_count=byte_count,
        )
        if isinstance(frame_result, Rejected):
            return frame_result
        append_result = self._append_or_defer(frame_result.value)
        if isinstance(append_result, Rejected):
            return append_result
        return Ok(value=None)

    def drain_pending_rows(self) -> _result.Ok[None] | _result.Rejected:
        """Append pending frame rows while this batch has room."""
        pending = self._pending_frame
        if pending is None:
            return Ok(value=None)
        if not self._can_append(pending):
            return Ok(value=None)
        append_result = self._append_frame(pending)
        if isinstance(append_result, Rejected):
            return append_result
        self._pending_frame = None
        return Ok(value=None)

    def finish_batch(
        self,
    ) -> _result.Ok[ModelRankInferenceBatch] | _result.Rejected:
        """Return the assembled final inference batch."""
        builder = self._builder
        single_frame = self._single_frame
        assert builder is not None or single_frame is not None
        self._builder = None
        self._single_frame = None
        if builder is not None:
            return Ok(value=builder.finish())
        assert single_frame is not None
        return self._finish_single_frame(single_frame)

    def _stage_host_frame(
        self,
        *,
        slot: _PinnedHostFrameSlot,
        byte_count: int,
        metadata: PolicyRequestFrameMetadata,
        recv_seconds: float,
        wire_byte_count: int,
    ) -> _result.Ok[_ReceivedPolicyRequestFrame] | _result.Rejected:
        decode_start = time.perf_counter()
        host_frame = slot.tensor[:byte_count]
        device_batch_result = materialize_policy_request_frame(
            device_frame=host_frame,
            metadata=metadata,
        )
        device_decode_seconds = time.perf_counter() - decode_start
        if isinstance(device_batch_result, Rejected):
            return device_batch_result
        return Ok(
            value=_ReceivedPolicyRequestFrame(
                slot=slot,
                metadata=metadata,
                routes=metadata.routes,
                device_batch=device_batch_result.value,
                generation_step_counts=metadata.generation_step_counts,
                wire_byte_count=wire_byte_count,
                recv_seconds=recv_seconds,
                device_decode_seconds=device_decode_seconds,
                frame_count=1,
            )
        )

    def discard_batch(self) -> None:
        """Drop partial aggregate rows after a staging error."""
        self._builder = None

    def _validate_frame_metadata(
        self, metadata: PolicyRequestFrameMetadata
    ) -> _result.Ok[None] | _result.Rejected:
        if metadata.batch_capacity > self.batch_size:
            return Rejected(reason="policy request frame exceeds slot")
        if (
            metadata.max_observation_tokens
            != self.max_observation_tokens
        ):
            return Rejected(
                reason="policy request observation layout mismatch"
            )
        return Ok(value=None)

    def _append_or_defer(
        self, frame: _ReceivedPolicyRequestFrame
    ) -> Ok[None] | Rejected:
        if self._can_append(frame):
            return self._append_frame(frame)
        self._pending_frame = frame
        return Ok(value=None)

    def _append_frame(
        self, frame: _ReceivedPolicyRequestFrame
    ) -> Ok[None] | Rejected:
        single_frame = self._single_frame
        builder = self._builder
        if builder is None and single_frame is None:
            self._single_frame = frame
            return Ok(value=None)
        if builder is None:
            assert single_frame is not None
            builder = self._workspace_for(single_frame.device_batch)
            builder.reset()
            self._builder = builder
            first_h2d_seconds = builder.append(single_frame)
            _record_slot_copy_event(
                slot=single_frame.slot, device=self.device
            )
            builder.h2d_seconds += first_h2d_seconds
            self._single_frame = None
        h2d_seconds = builder.append(frame)
        _record_slot_copy_event(slot=frame.slot, device=self.device)
        builder.h2d_seconds += h2d_seconds
        return Ok(value=None)

    def _finish_single_frame(
        self, frame: _ReceivedPolicyRequestFrame
    ) -> _result.Ok[ModelRankInferenceBatch] | _result.Rejected:
        h2d_start = time.perf_counter()
        if self.device.type == "cpu":
            device_batch = frame.device_batch
            h2d_seconds = 0.0
        else:
            host_frame = frame.slot.tensor[: frame.wire_byte_count]
            device_frame = copy_policy_request_host_frame_to_device(
                host_frame=host_frame,
                device_slot=None,
                device=self.device,
            )
            h2d_seconds = time.perf_counter() - h2d_start
            decode_start = time.perf_counter()
            batch_result = materialize_policy_request_frame(
                device_frame=device_frame,
                metadata=frame.metadata,
            )
            if isinstance(batch_result, Rejected):
                return batch_result
            device_decode_seconds = (
                frame.device_decode_seconds
                + time.perf_counter()
                - decode_start
            )
            return Ok(
                value=ModelRankInferenceBatch(
                    routes=frame.routes,
                    device_batch=batch_result.value,
                    generation_step_counts=(
                        frame.generation_step_counts
                    ),
                    wire_byte_count=frame.wire_byte_count,
                    recv_seconds=frame.recv_seconds,
                    h2d_seconds=h2d_seconds,
                    device_decode_seconds=device_decode_seconds,
                    frame_count=frame.frame_count,
                )
            )
        return Ok(
            value=ModelRankInferenceBatch(
                routes=frame.routes,
                device_batch=device_batch,
                generation_step_counts=frame.generation_step_counts,
                wire_byte_count=frame.wire_byte_count,
                recv_seconds=frame.recv_seconds,
                h2d_seconds=h2d_seconds,
                device_decode_seconds=frame.device_decode_seconds,
                frame_count=frame.frame_count,
            )
        )

    def _workspace_for(
        self, template: DevicePolicyRequestBatch
    ) -> _FinalPolicyRequestBatchBuilder:
        workspace = self._workspace
        if workspace is None or not workspace.accepts_template(
            template=template,
            batch_size=self.batch_size,
            device=self.device,
        ):
            workspace = _FinalPolicyRequestBatchBuilder.create(
                batch_size=self.batch_size,
                template=template,
                device=self.device,
            )
            self._workspace = workspace
        return workspace

    def _can_append(self, frame: _ReceivedPolicyRequestFrame) -> bool:
        if (
            self._active_row_count() + frame.row_count()
            > self.batch_size
        ):
            return False
        return True

    def _active_row_count(self) -> int:
        single_frame = self._single_frame
        single_count = (
            0 if single_frame is None else single_frame.row_count()
        )
        builder = self._builder
        if builder is None:
            return single_count
        return single_count + builder.row_count

    def _acquire_slot(self) -> _PinnedHostFrameSlot:
        index = self._next_slot_index
        self._next_slot_index += 1
        if index == len(self._host_slots):
            self._host_slots.append(
                _PinnedHostFrameSlot(
                    tensor=_allocate_host_frame(
                        byte_count=self._max_frame_bytes,
                        device=self.device,
                    )
                )
            )
        slot = self._host_slots[index]
        slot.wait_ready()
        return slot


def _allocate_host_frame(
    *, byte_count: int, device: torch.device
) -> Tensor:
    assert byte_count > 0
    return torch.empty(
        (byte_count,),
        dtype=torch.uint8,
        device=torch.device("cpu"),
        pin_memory=device.type == "cuda",
    )


def _host_frame_view(frame: Tensor) -> memoryview:
    assert frame.device.type == "cpu"
    return memoryview(frame.numpy())


@dataclass(slots=True)
class _FinalPolicyRequestBatchBuilder:
    observation_batch: ObservationTensorBatch
    action_plan_batch: DeviceActionPlanBatch
    sampling_thresholds: Tensor
    generation_step_counts: Tensor
    max_batch_size: int
    routes: list[PolicyRequestRoute] = field(
        default_factory=_route_list
    )
    policy_versions: list[int] = field(
        default_factory=_policy_version_list
    )
    row_count: int = 0
    padded_generation_steps: int = 0
    generation_step_count_values: list[int] = field(
        default_factory=_generation_step_count_list
    )
    wire_byte_count: int = 0
    recv_seconds: float = 0.0
    h2d_seconds: float = 0.0
    device_decode_seconds: float = 0.0
    frame_count: int = 0

    @classmethod
    def create(
        cls,
        *,
        batch_size: int,
        template: DevicePolicyRequestBatch,
        device: torch.device,
    ) -> _FinalPolicyRequestBatchBuilder:
        observation = template.observation_batch
        return cls(
            observation_batch=ObservationTensorBatch(
                component_ids=_empty_like_rows(
                    observation.component_ids,
                    row_count=batch_size,
                    device=device,
                ),
                numeric_values=_empty_like_rows(
                    observation.numeric_values,
                    row_count=batch_size,
                    device=device,
                ),
                numeric_masks=_empty_like_rows(
                    observation.numeric_masks,
                    row_count=batch_size,
                    device=device,
                ),
            ),
            action_plan_batch=_empty_action_plan_batch(
                template.action_plan_batch,
                row_count=batch_size,
                padded_generation_steps=(
                    SEMANTIC_CODEC.max_argument_tokens
                ),
                device=device,
            ),
            sampling_thresholds=torch.zeros(
                (batch_size, SEMANTIC_CODEC.max_argument_tokens),
                dtype=template.sampling_thresholds.dtype,
                device=device,
            ),
            generation_step_counts=torch.empty(
                (batch_size,),
                dtype=template.generation_step_counts.dtype,
                device=device,
            ),
            max_batch_size=batch_size,
        )

    def accepts_template(
        self,
        *,
        template: DevicePolicyRequestBatch,
        batch_size: int,
        device: torch.device,
    ) -> bool:
        """Return whether this workspace can stage another batch."""
        return (
            self.max_batch_size == batch_size
            and self.observation_batch.component_ids.device == device
            and self.observation_batch.component_ids.dtype
            == template.observation_batch.component_ids.dtype
            and self.observation_batch.numeric_values.dtype
            == template.observation_batch.numeric_values.dtype
            and self.sampling_thresholds.dtype
            == template.sampling_thresholds.dtype
            and self.generation_step_counts.dtype
            == template.generation_step_counts.dtype
        )

    def reset(self) -> None:
        """Clear staged row metadata while reusing tensor storage."""
        self.routes.clear()
        self.policy_versions.clear()
        self.row_count = 0
        self.padded_generation_steps = 0
        self.generation_step_count_values.clear()
        self.wire_byte_count = 0
        self.recv_seconds = 0.0
        self.h2d_seconds = 0.0
        self.device_decode_seconds = 0.0
        self.frame_count = 0

    def append(self, frame: _ReceivedPolicyRequestFrame) -> float:
        count = frame.row_count()
        assert count > 0
        start = self.row_count
        end = start + count
        assert end <= self.max_batch_size
        h2d_start = time.perf_counter()
        _copy_observation_rows(
            destination=self.observation_batch,
            source=frame.device_batch.observation_batch,
            start=start,
        )
        _copy_action_plan_rows(
            destination=self.action_plan_batch,
            source=frame.device_batch.action_plan_batch,
            start=start,
        )
        self.sampling_thresholds[start:end, :].zero_()
        self.sampling_thresholds[
            start:end, : frame.padded_generation_steps()
        ].copy_(
            frame.device_batch.sampling_thresholds,
            non_blocking=_non_blocking_copy(
                destination=self.sampling_thresholds,
                source=frame.device_batch.sampling_thresholds,
            ),
        )
        self.generation_step_counts[start:end].copy_(
            frame.device_batch.generation_step_counts,
            non_blocking=_non_blocking_copy(
                destination=self.generation_step_counts,
                source=frame.device_batch.generation_step_counts,
            ),
        )
        self.routes.extend(frame.routes)
        self.policy_versions.extend(frame.device_batch.policy_versions)
        self.generation_step_count_values.extend(
            frame.generation_step_counts
        )
        self.row_count = end
        self.padded_generation_steps = max(
            self.padded_generation_steps,
            frame.padded_generation_steps(),
        )
        self.wire_byte_count += frame.wire_byte_count
        self.recv_seconds += frame.recv_seconds
        self.device_decode_seconds += frame.device_decode_seconds
        self.frame_count += frame.frame_count
        return time.perf_counter() - h2d_start

    def finish(self) -> ModelRankInferenceBatch:
        assert self.row_count > 0
        assert self.padded_generation_steps > 0
        row_slice = slice(0, self.row_count)
        return ModelRankInferenceBatch(
            routes=tuple(self.routes),
            device_batch=DevicePolicyRequestBatch(
                observation_batch=ObservationTensorBatch(
                    component_ids=(
                        self.observation_batch.component_ids[row_slice]
                    ),
                    numeric_values=(
                        self.observation_batch.numeric_values[row_slice]
                    ),
                    numeric_masks=(
                        self.observation_batch.numeric_masks[row_slice]
                    ),
                ),
                action_plan_batch=_slice_action_plan_batch(
                    self.action_plan_batch,
                    row_count=self.row_count,
                    padded_generation_steps=self.padded_generation_steps,
                ),
                sampling_thresholds=(
                    self.sampling_thresholds[
                        row_slice, : self.padded_generation_steps
                    ]
                ),
                generation_step_counts=(
                    self.generation_step_counts[row_slice]
                ),
                policy_versions=tuple(self.policy_versions),
                padded_generation_steps=self.padded_generation_steps,
            ),
            generation_step_counts=tuple(
                self.generation_step_count_values
            ),
            wire_byte_count=self.wire_byte_count,
            recv_seconds=self.recv_seconds,
            h2d_seconds=self.h2d_seconds,
            device_decode_seconds=self.device_decode_seconds,
            frame_count=self.frame_count,
        )


def _slice_action_plan_batch(
    batch: DeviceActionPlanBatch,
    *,
    row_count: int,
    padded_generation_steps: int,
) -> DeviceActionPlanBatch:
    row_slice = slice(0, row_count)
    return DeviceActionPlanBatch(
        kind_codes=batch.kind_codes[row_slice],
        available_counts=batch.available_counts[row_slice],
        effective_suits=batch.effective_suits[row_slice],
        same_suit_mask=batch.same_suit_mask[row_slice],
        off_suit_mask=batch.off_suit_mask[row_slice],
        pair_face_mask=batch.pair_face_mask[row_slice],
        min_select=batch.min_select[row_slice],
        max_select=batch.max_select[row_slice],
        exact_select=batch.exact_select[row_slice],
        required_same_suit_count=(
            batch.required_same_suit_count[row_slice]
        ),
        pair_floor=batch.pair_floor[row_slice],
        has_tractor=batch.has_tractor[row_slice],
        trace_tokens=(
            batch.trace_tokens[row_slice, :, :padded_generation_steps]
        ),
        trace_token_mask=(
            batch.trace_token_mask[
                row_slice, :, :padded_generation_steps
            ]
        ),
        trace_lengths=batch.trace_lengths[row_slice],
        trace_row_mask=batch.trace_row_mask[row_slice],
        pair_plan_masks=batch.pair_plan_masks[row_slice],
        pair_plan_row_mask=batch.pair_plan_row_mask[row_slice],
    )


def _empty_action_plan_batch(
    template: DeviceActionPlanBatch,
    *,
    row_count: int,
    padded_generation_steps: int,
    device: torch.device,
) -> DeviceActionPlanBatch:
    return DeviceActionPlanBatch(
        kind_codes=_empty_like_rows(
            template.kind_codes, row_count=row_count, device=device
        ),
        available_counts=_empty_like_rows(
            template.available_counts,
            row_count=row_count,
            device=device,
        ),
        effective_suits=_empty_like_rows(
            template.effective_suits, row_count=row_count, device=device
        ),
        same_suit_mask=_empty_like_rows(
            template.same_suit_mask, row_count=row_count, device=device
        ),
        off_suit_mask=_empty_like_rows(
            template.off_suit_mask, row_count=row_count, device=device
        ),
        pair_face_mask=_empty_like_rows(
            template.pair_face_mask, row_count=row_count, device=device
        ),
        min_select=_empty_like_rows(
            template.min_select, row_count=row_count, device=device
        ),
        max_select=_empty_like_rows(
            template.max_select, row_count=row_count, device=device
        ),
        exact_select=_empty_like_rows(
            template.exact_select, row_count=row_count, device=device
        ),
        required_same_suit_count=_empty_like_rows(
            template.required_same_suit_count,
            row_count=row_count,
            device=device,
        ),
        pair_floor=_empty_like_rows(
            template.pair_floor, row_count=row_count, device=device
        ),
        has_tractor=_empty_like_rows(
            template.has_tractor, row_count=row_count, device=device
        ),
        trace_tokens=torch.zeros(
            (*template.trace_tokens.shape[:1],),
            dtype=template.trace_tokens.dtype,
            device=device,
        ).new_zeros(
            (
                row_count,
                int(template.trace_tokens.shape[1]),
                padded_generation_steps,
            )
        ),
        trace_token_mask=torch.zeros(
            (
                row_count,
                int(template.trace_token_mask.shape[1]),
                padded_generation_steps,
            ),
            dtype=template.trace_token_mask.dtype,
            device=device,
        ),
        trace_lengths=_empty_like_rows(
            template.trace_lengths, row_count=row_count, device=device
        ),
        trace_row_mask=_empty_like_rows(
            template.trace_row_mask, row_count=row_count, device=device
        ),
        pair_plan_masks=_empty_like_rows(
            template.pair_plan_masks, row_count=row_count, device=device
        ),
        pair_plan_row_mask=_empty_like_rows(
            template.pair_plan_row_mask,
            row_count=row_count,
            device=device,
        ),
    )


def _empty_like_rows(
    value: Tensor, *, row_count: int, device: torch.device
) -> Tensor:
    return torch.empty(
        (row_count, *value.shape[1:]),
        dtype=value.dtype,
        device=device,
    )


def _copy_observation_rows(
    *,
    destination: ObservationTensorBatch,
    source: ObservationTensorBatch,
    start: int,
) -> None:
    count = int(source.component_ids.shape[0])
    destination.component_ids[start : start + count].copy_(
        source.component_ids,
        non_blocking=_non_blocking_copy(
            destination=destination.component_ids,
            source=source.component_ids,
        ),
    )
    destination.numeric_values[start : start + count].copy_(
        source.numeric_values,
        non_blocking=_non_blocking_copy(
            destination=destination.numeric_values,
            source=source.numeric_values,
        ),
    )
    destination.numeric_masks[start : start + count].copy_(
        source.numeric_masks,
        non_blocking=_non_blocking_copy(
            destination=destination.numeric_masks,
            source=source.numeric_masks,
        ),
    )


def _copy_action_plan_rows(
    *,
    destination: DeviceActionPlanBatch,
    source: DeviceActionPlanBatch,
    start: int,
) -> None:
    count = source.batch_size()
    _copy_fixed(destination.kind_codes, source.kind_codes, start=start)
    _copy_fixed(
        destination.available_counts,
        source.available_counts,
        start=start,
    )
    _copy_fixed(
        destination.effective_suits, source.effective_suits, start=start
    )
    _copy_fixed(
        destination.same_suit_mask, source.same_suit_mask, start=start
    )
    _copy_fixed(
        destination.off_suit_mask, source.off_suit_mask, start=start
    )
    _copy_fixed(
        destination.pair_face_mask, source.pair_face_mask, start=start
    )
    _copy_fixed(destination.min_select, source.min_select, start=start)
    _copy_fixed(destination.max_select, source.max_select, start=start)
    _copy_fixed(
        destination.exact_select, source.exact_select, start=start
    )
    _copy_fixed(
        destination.required_same_suit_count,
        source.required_same_suit_count,
        start=start,
    )
    _copy_fixed(destination.pair_floor, source.pair_floor, start=start)
    _copy_fixed(
        destination.has_tractor, source.has_tractor, start=start
    )
    generation_width = int(source.trace_tokens.shape[-1])
    destination.trace_tokens[start : start + count].zero_()
    destination.trace_token_mask[start : start + count].fill_(False)
    destination.trace_tokens[
        start : start + count, :, :generation_width
    ].copy_(
        source.trace_tokens,
        non_blocking=_non_blocking_copy(
            destination=destination.trace_tokens,
            source=source.trace_tokens,
        ),
    )
    destination.trace_token_mask[
        start : start + count, :, :generation_width
    ].copy_(
        source.trace_token_mask,
        non_blocking=_non_blocking_copy(
            destination=destination.trace_token_mask,
            source=source.trace_token_mask,
        ),
    )
    _copy_fixed(
        destination.trace_lengths, source.trace_lengths, start=start
    )
    _copy_fixed(
        destination.trace_row_mask, source.trace_row_mask, start=start
    )
    _copy_fixed(
        destination.pair_plan_masks, source.pair_plan_masks, start=start
    )
    _copy_fixed(
        destination.pair_plan_row_mask,
        source.pair_plan_row_mask,
        start=start,
    )


def _copy_fixed(
    destination: Tensor, source: Tensor, *, start: int
) -> None:
    count = int(source.shape[0])
    destination[start : start + count].copy_(
        source,
        non_blocking=_non_blocking_copy(
            destination=destination, source=source
        ),
    )


def _record_slot_copy_event(
    *, slot: _PinnedHostFrameSlot, device: torch.device
) -> None:
    if device.type != "cuda":
        return
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(device))
    slot.pending_event = event


def _non_blocking_copy(*, destination: Tensor, source: Tensor) -> bool:
    return (
        destination.device.type == "cuda"
        and source.device.type == "cpu"
    )
