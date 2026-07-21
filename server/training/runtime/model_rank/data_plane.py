"""Command-scoped model-rank inference data plane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
from server.training.policy_inference_batch import (
    DevicePolicyRequestBatch,
    PolicyRequestRoute,
)
from server.training.runtime.async_ipc import (
    AsyncChildControlEndpoint,
    AsyncFrameEndpoint,
    wait_readable_frames,
)
from server.training.runtime.model_rank.inference_transport import (
    AsyncPolicyPeer,
)
from server.training.runtime.model_rank.messages import (
    ModelRankCommand,
    ModelRankResponse,
)
from server.training.runtime.model_rank.shape_planner import (
    InferenceShapePlan,
    plan_inference_shape_batches,
)
from server.training.runtime.model_rank.staging import (
    ModelRankInferenceBatch,
    PolicyRequestIngress,
)
from server.training.semantic_action_plan import DeviceActionPlanBatch
from server.training.tensorize import ObservationTensorBatch


class ModelRankBatchHandler(Protocol):
    """Process one inference batch inside a data-plane transfer."""

    async def __call__(
        self, batch: ModelRankInferenceBatch
    ) -> _result.Ok[None] | _result.Rejected: ...


class ModelRankRejectHandler(Protocol):
    """Reject one request batch inside a data-plane transfer."""

    async def __call__(
        self, *, routes: tuple[PolicyRequestRoute, ...], reason: str
    ) -> _result.Ok[None] | _result.Rejected: ...


@dataclass(slots=True)
class ModelRankDataPlane:
    """Drain inference requests until the next coordinator command."""

    control: AsyncChildControlEndpoint[
        ModelRankCommand, ModelRankResponse
    ]
    request_peers: tuple[AsyncPolicyPeer, ...]
    ingress: PolicyRequestIngress

    async def run_until_command(
        self,
        *,
        policy_version: int,
        process_batch: ModelRankBatchHandler,
        reject_batch: ModelRankRejectHandler,
    ) -> _result.Ok[ModelRankCommand] | _result.Rejected:
        """Process this policy version until a command is readable."""
        assert policy_version >= 0
        while True:
            if self.ingress.has_pending_rows():
                request_result = await self._handle_ready_requests(
                    initial_ready=(),
                    policy_version=policy_version,
                    process_batch=process_batch,
                    reject_batch=reject_batch,
                )
                if isinstance(request_result, Rejected):
                    return request_result
                continue
            ready_requests = self._ready_request_peers(
                timeout_seconds=0.0
            )
            if isinstance(ready_requests, Rejected):
                return ready_requests
            if ready_requests:
                request_result = await self._handle_ready_requests(
                    initial_ready=ready_requests,
                    policy_version=policy_version,
                    process_batch=process_batch,
                    reject_batch=reject_batch,
                )
                if isinstance(request_result, Rejected):
                    return request_result
                continue
            if await self.control.command_ready(0.0):
                return await self.control.recv_command()
            ready = await self._wait_for_input()
            if isinstance(ready, Rejected):
                return ready
            ready_requests = self._request_peers_from_ready(ready.value)
            if ready_requests:
                request_result = await self._handle_ready_requests(
                    initial_ready=ready_requests,
                    policy_version=policy_version,
                    process_batch=process_batch,
                    reject_batch=reject_batch,
                )
                if isinstance(request_result, Rejected):
                    return request_result
                continue
            return await self.control.recv_command()

    async def _handle_ready_requests(
        self,
        *,
        initial_ready: tuple[AsyncPolicyPeer, ...],
        policy_version: int,
        process_batch: ModelRankBatchHandler,
        reject_batch: ModelRankRejectHandler,
    ) -> _result.Ok[None] | _result.Rejected:
        batch_result = await self._receive_ready_batch(
            initial_ready=initial_ready
        )
        if isinstance(batch_result, Rejected):
            return batch_result
        batch = batch_result.value
        rows = _partition_policy_version_rows(
            batch.device_batch.policy_versions,
            policy_version=policy_version,
        )
        if rows.mismatched:
            reject_result = await reject_batch(
                routes=_select_routes(batch.routes, rows.mismatched),
                reason="policy request version does not match command",
            )
            if isinstance(reject_result, Rejected):
                return reject_result
        if not rows.matched:
            return Ok(value=None)
        return await _process_shape_buckets(
            batch=batch,
            rows=rows.matched,
            process_batch=process_batch,
        )

    async def _receive_ready_batch(
        self,
        *,
        initial_ready: tuple[AsyncPolicyPeer, ...],
    ) -> _result.Ok[ModelRankInferenceBatch] | _result.Rejected:
        self.ingress.begin_batch()
        pending_result = self.ingress.drain_pending_rows()
        if isinstance(pending_result, Rejected):
            self.ingress.discard_batch()
            return pending_result
        ready_receivers = initial_ready
        while ready_receivers and self.ingress.can_receive():
            for peer in ready_receivers:
                if not self.ingress.can_receive():
                    break
                receive_result = await self.ingress.receive_from(peer)
                if isinstance(receive_result, Rejected):
                    self.ingress.discard_batch()
                    return receive_result
            if self.ingress.can_receive():
                ready_result = self._ready_request_peers(
                    timeout_seconds=0.0
                )
                if isinstance(ready_result, Rejected):
                    self.ingress.discard_batch()
                    return ready_result
                ready_receivers = ready_result
        return self.ingress.finish_batch()

    def _ready_request_peers(
        self, *, timeout_seconds: float
    ) -> tuple[AsyncPolicyPeer, ...] | _result.Rejected:
        if not self.request_peers:
            return ()
        ready_result = _ready_endpoints_now(
            peers=self.request_peers,
            timeout_seconds=timeout_seconds,
        )
        if isinstance(ready_result, Rejected):
            return ready_result
        return self._request_peers_from_ready(ready_result.value)

    async def _wait_for_input(
        self,
    ) -> _result.Ok[tuple[AsyncFrameEndpoint, ...]] | _result.Rejected:
        endpoints = tuple(
            peer.endpoint for peer in self.request_peers
        ) + (self.control.frame_endpoint,)
        return await wait_readable_frames(
            endpoints=endpoints,
            timeout_seconds=None,
        )

    def _request_peers_from_ready(
        self, ready: tuple[AsyncFrameEndpoint, ...]
    ) -> tuple[AsyncPolicyPeer, ...]:
        return tuple(
            peer
            for peer in self.request_peers
            if _endpoint_in_ready(peer.endpoint, ready)
        )


def _endpoint_in_ready(
    endpoint: AsyncFrameEndpoint, ready: tuple[AsyncFrameEndpoint, ...]
) -> bool:
    return any(endpoint is item for item in ready)


def _ready_endpoints_now(
    *,
    peers: tuple[AsyncPolicyPeer, ...],
    timeout_seconds: float,
) -> _result.Ok[tuple[AsyncFrameEndpoint, ...]] | _result.Rejected:
    assert timeout_seconds == 0.0
    return Ok(
        value=tuple(
            peer.endpoint
            for peer in peers
            if peer.endpoint.is_readable()
        )
    )


@dataclass(frozen=True, slots=True)
class _PolicyVersionRows:
    matched: tuple[int, ...]
    mismatched: tuple[int, ...]


def _partition_policy_version_rows(
    versions: tuple[int, ...], *, policy_version: int
) -> _PolicyVersionRows:
    matched: list[int] = []
    mismatched: list[int] = []
    for index, version in enumerate(versions):
        if version == policy_version:
            matched.append(index)
            continue
        mismatched.append(index)
    return _PolicyVersionRows(
        matched=tuple(matched),
        mismatched=tuple(mismatched),
    )


def _select_routes(
    routes: tuple[PolicyRequestRoute, ...], rows: tuple[int, ...]
) -> tuple[PolicyRequestRoute, ...]:
    assert rows
    return tuple(routes[index] for index in rows)


def _select_staged_rows(
    batch: ModelRankInferenceBatch, rows: tuple[int, ...]
) -> ModelRankInferenceBatch:
    assert rows
    if _rows_cover_batch(batch=batch, rows=rows):
        return batch
    contiguous = _contiguous_row_slice(rows)
    if contiguous is not None:
        return _slice_staged_rows(batch=batch, row_slice=contiguous)
    row_index = _row_index_tensor(
        rows, like=batch.device_batch.sampling_thresholds
    )
    padded_generation_steps = _selected_padded_generation_steps(
        batch.generation_step_counts, rows=rows
    )
    return ModelRankInferenceBatch(
        routes=_select_routes(batch.routes, rows),
        device_batch=_select_device_request_rows(
            batch.device_batch,
            rows=rows,
            row_index=row_index,
            padded_generation_steps=padded_generation_steps,
        ),
        generation_step_counts=tuple(
            batch.generation_step_counts[index] for index in rows
        ),
        wire_byte_count=batch.wire_byte_count,
        recv_seconds=batch.recv_seconds,
        h2d_seconds=batch.h2d_seconds,
        device_decode_seconds=batch.device_decode_seconds,
        frame_count=batch.frame_count,
    )


def _slice_staged_rows(
    *, batch: ModelRankInferenceBatch, row_slice: slice
) -> ModelRankInferenceBatch:
    rows = tuple(
        range(
            _slice_start(row_slice),
            _slice_stop(row_slice),
        )
    )
    padded_generation_steps = _selected_padded_generation_steps(
        batch.generation_step_counts, rows=rows
    )
    return ModelRankInferenceBatch(
        routes=batch.routes[row_slice],
        device_batch=_slice_device_request_rows(
            batch.device_batch,
            row_slice=row_slice,
            padded_generation_steps=padded_generation_steps,
        ),
        generation_step_counts=batch.generation_step_counts[row_slice],
        wire_byte_count=batch.wire_byte_count,
        recv_seconds=batch.recv_seconds,
        h2d_seconds=batch.h2d_seconds,
        device_decode_seconds=batch.device_decode_seconds,
        frame_count=batch.frame_count,
    )


def _rows_cover_batch(
    *, batch: ModelRankInferenceBatch, rows: tuple[int, ...]
) -> bool:
    return rows == tuple(range(batch.batch_size()))


def _contiguous_row_slice(rows: tuple[int, ...]) -> slice | None:
    assert rows
    start = rows[0]
    previous = start
    for row in rows[1:]:
        if row != previous + 1:
            return None
        previous = row
    return slice(start, previous + 1)


def _slice_start(row_slice: slice) -> int:
    assert isinstance(row_slice.start, int)
    return row_slice.start


def _slice_stop(row_slice: slice) -> int:
    assert isinstance(row_slice.stop, int)
    return row_slice.stop


def _select_device_request_rows(
    batch: DevicePolicyRequestBatch,
    *,
    rows: tuple[int, ...],
    row_index: Tensor,
    padded_generation_steps: int,
) -> DevicePolicyRequestBatch:
    return DevicePolicyRequestBatch(
        observation_batch=ObservationTensorBatch(
            category_ids=_select_tensor_rows(
                batch.observation_batch.category_ids, row_index
            ),
            scalar_values=_select_tensor_rows(
                batch.observation_batch.scalar_values, row_index
            ),
            card_rule_values=_select_tensor_rows(
                batch.observation_batch.card_rule_values, row_index
            ),
            coordinate_values=_select_tensor_rows(
                batch.observation_batch.coordinate_values, row_index
            ),
            coordinate_masks=_select_tensor_rows(
                batch.observation_batch.coordinate_masks, row_index
            ),
            candidate_category_ids=_select_tensor_rows(
                batch.observation_batch.candidate_category_ids,
                row_index,
            ),
            candidate_counts=_select_tensor_rows(
                batch.observation_batch.candidate_counts, row_index
            ),
            candidate_card_rule_values=_select_tensor_rows(
                batch.observation_batch.candidate_card_rule_values,
                row_index,
            ),
            query_indices=_select_tensor_rows(
                batch.observation_batch.query_indices, row_index
            ),
        ),
        action_plan_batch=_select_action_plan_rows(
            batch.action_plan_batch,
            row_index=row_index,
        ),
        sampling_thresholds=_select_tensor_rows(
            batch.sampling_thresholds, row_index
        )[:, :padded_generation_steps],
        generation_step_counts=_select_tensor_rows(
            batch.generation_step_counts, row_index
        ),
        policy_versions=tuple(
            batch.policy_versions[index] for index in rows
        ),
        padded_generation_steps=padded_generation_steps,
    )


def _slice_device_request_rows(
    batch: DevicePolicyRequestBatch,
    *,
    row_slice: slice,
    padded_generation_steps: int,
) -> DevicePolicyRequestBatch:
    return DevicePolicyRequestBatch(
        observation_batch=ObservationTensorBatch(
            category_ids=batch.observation_batch.category_ids[
                row_slice
            ],
            scalar_values=batch.observation_batch.scalar_values[
                row_slice
            ],
            card_rule_values=batch.observation_batch.card_rule_values[
                row_slice
            ],
            coordinate_values=batch.observation_batch.coordinate_values[
                row_slice
            ],
            coordinate_masks=batch.observation_batch.coordinate_masks[
                row_slice
            ],
            candidate_category_ids=batch.observation_batch.candidate_category_ids[
                row_slice
            ],
            candidate_counts=batch.observation_batch.candidate_counts[
                row_slice
            ],
            candidate_card_rule_values=batch.observation_batch.candidate_card_rule_values[
                row_slice
            ],
            query_indices=batch.observation_batch.query_indices[
                row_slice
            ],
        ),
        action_plan_batch=_slice_action_plan_rows(
            batch.action_plan_batch,
            row_slice=row_slice,
            padded_generation_steps=padded_generation_steps,
        ),
        sampling_thresholds=batch.sampling_thresholds[
            row_slice, :padded_generation_steps
        ],
        generation_step_counts=batch.generation_step_counts[row_slice],
        policy_versions=batch.policy_versions[row_slice],
        padded_generation_steps=padded_generation_steps,
    )


async def _process_shape_buckets(
    *,
    batch: ModelRankInferenceBatch,
    rows: tuple[int, ...],
    process_batch: ModelRankBatchHandler,
) -> Ok[None] | Rejected:
    plan = plan_inference_shape_batches(
        batch.generation_step_counts, rows=rows
    )
    for bucket in plan.buckets:
        process_result = await process_batch(
            _annotate_shape_plan(
                _select_staged_rows(batch, bucket), plan
            )
        )
        if isinstance(process_result, Rejected):
            return process_result
    return Ok(value=None)


def _annotate_shape_plan(
    batch: ModelRankInferenceBatch, plan: InferenceShapePlan
) -> ModelRankInferenceBatch:
    return ModelRankInferenceBatch(
        routes=batch.routes,
        device_batch=batch.device_batch,
        generation_step_counts=batch.generation_step_counts,
        wire_byte_count=batch.wire_byte_count,
        recv_seconds=batch.recv_seconds,
        h2d_seconds=batch.h2d_seconds,
        device_decode_seconds=batch.device_decode_seconds,
        frame_count=batch.frame_count,
        shape_bucket_count=plan.bucket_count(),
        shape_padding_tokens_saved=plan.saved_padding_tokens(),
    )


def _selected_padded_generation_steps(
    counts: tuple[int, ...], *, rows: tuple[int, ...]
) -> int:
    return max(counts[row] for row in rows)


def _select_action_plan_rows(
    batch: DeviceActionPlanBatch, *, row_index: Tensor
) -> DeviceActionPlanBatch:
    return DeviceActionPlanBatch(
        kind_codes=_select_tensor_rows(batch.kind_codes, row_index),
        available_counts=_select_tensor_rows(
            batch.available_counts, row_index
        ),
        effective_suits=_select_tensor_rows(
            batch.effective_suits, row_index
        ),
        same_suit_mask=_select_tensor_rows(
            batch.same_suit_mask, row_index
        ),
        off_suit_mask=_select_tensor_rows(
            batch.off_suit_mask, row_index
        ),
        pair_face_mask=_select_tensor_rows(
            batch.pair_face_mask, row_index
        ),
        min_select=_select_tensor_rows(batch.min_select, row_index),
        max_select=_select_tensor_rows(batch.max_select, row_index),
        exact_select=_select_tensor_rows(batch.exact_select, row_index),
        required_same_suit_count=_select_tensor_rows(
            batch.required_same_suit_count, row_index
        ),
        pair_floor=_select_tensor_rows(batch.pair_floor, row_index),
        has_tractor=_select_tensor_rows(batch.has_tractor, row_index),
        trace_choice_ids=_select_tensor_rows(
            batch.trace_choice_ids, row_index
        ),
        trace_choice_mask=_select_tensor_rows(
            batch.trace_choice_mask, row_index
        ),
        trace_lengths=_select_tensor_rows(
            batch.trace_lengths, row_index
        ),
        trace_row_mask=_select_tensor_rows(
            batch.trace_row_mask, row_index
        ),
        pair_plan_masks=_select_tensor_rows(
            batch.pair_plan_masks, row_index
        ),
        pair_plan_row_mask=_select_tensor_rows(
            batch.pair_plan_row_mask, row_index
        ),
    )


def _slice_action_plan_rows(
    batch: DeviceActionPlanBatch,
    *,
    row_slice: slice,
    padded_generation_steps: int,
) -> DeviceActionPlanBatch:
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
        trace_choice_ids=batch.trace_choice_ids[
            row_slice, :, :padded_generation_steps
        ],
        trace_choice_mask=batch.trace_choice_mask[
            row_slice, :, :padded_generation_steps
        ],
        trace_lengths=batch.trace_lengths[row_slice],
        trace_row_mask=batch.trace_row_mask[row_slice],
        pair_plan_masks=batch.pair_plan_masks[row_slice],
        pair_plan_row_mask=batch.pair_plan_row_mask[row_slice],
    )


def _select_tensor_rows(tensor: Tensor, row_index: Tensor) -> Tensor:
    return tensor.index_select(0, row_index)


def _row_index_tensor(rows: tuple[int, ...], *, like: Tensor) -> Tensor:
    assert rows
    return torch.tensor(rows, dtype=torch.long, device=like.device)
