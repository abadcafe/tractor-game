"""Model-rank-owned append-only sample slab for PPO replay."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server import result as _result
from server.result import Ok, Rejected
from server.training.policy_sampling.records import (
    CompactTraceTokenIds,
    DecisionHandle,
    ModelRankPolicyDecision,
    RankReturnTargets,
    SampledPolicyBatch,
)
from server.training.ppo.minibatch import TensorizedPPOMinibatch
from server.training.ppo.replay_tensors import PPOReplayTensorBatch
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tensorize import ObservationTensorBatch

_INITIAL_CAPACITY = 256

type ModelRankDecisionResult = (
    _result.Ok[ModelRankPolicyDecision] | _result.Rejected
)


@dataclass(frozen=True, slots=True)
class _ResponseTraceCpuView:
    selected_token_ids: Tensor

    def __post_init__(self) -> None:
        assert self.selected_token_ids.ndim == 2
        assert int(self.selected_token_ids.shape[0]) > 0


@dataclass(frozen=True, slots=True)
class ArenaPPOBatchSource:
    """PPO minibatch source backed by one model-rank sample slab."""

    arena: ModelRankSampleArena
    policy_version: int
    model_rank_index: int
    row_indices: Tensor
    step_counts: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    return_values: Tensor
    raw_advantages: Tensor
    max_step_count: int

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.model_rank_index >= 0
        assert self.max_step_count > 0
        sample_count = int(self.row_indices.shape[0])
        assert sample_count > 0
        assert self.row_indices.ndim == 1
        assert self.step_counts.shape == self.row_indices.shape
        assert (
            self.old_log_probabilities.shape == self.row_indices.shape
        )
        assert self.old_values.shape == self.row_indices.shape
        assert self.return_values.shape == self.row_indices.shape
        assert self.raw_advantages.shape == self.row_indices.shape

    def sample_count(self) -> int:
        """Return trainable sample count."""
        return int(self.row_indices.shape[0])

    def select_minibatch(
        self,
        *,
        indices: Tensor,
        advantages: Tensor,
        global_count: Tensor,
    ) -> TensorizedPPOMinibatch:
        """Return one PPO minibatch directly from slab rows."""
        return self.arena.select_ppo_minibatch(
            source=self,
            indices=indices,
            advantages=advantages,
            global_count=global_count,
        )


@dataclass(slots=True)
class ModelRankSampleArena:
    """Append sampled policy rows and expose committed PPO batches."""

    model_rank_index: int
    device: torch.device
    _capacity: int = 0
    _row_count: int = 0
    _policy_version: int | None = None
    _row_policy_versions: Tensor | None = None
    _row_step_counts: Tensor | None = None
    _component_ids: Tensor | None = None
    _numeric_values: Tensor | None = None
    _numeric_masks: Tensor | None = None
    _selected_token_ids: Tensor | None = None
    _choice_token_ids: Tensor | None = None
    _choice_masks: Tensor | None = None
    _selected_choice_offsets: Tensor | None = None
    _old_log_probabilities: Tensor | None = None
    _old_values: Tensor | None = None

    def __post_init__(self) -> None:
        assert self.model_rank_index >= 0

    def clear(self) -> None:
        """Make every slab row reusable after state sync or update."""
        self._row_count = 0
        self._policy_version = None

    def store_sampled_batch(
        self, *, batch: SampledPolicyBatch
    ) -> tuple[ModelRankDecisionResult, ...]:
        """Append sampled rows and return response-ready decisions."""
        assert batch.old_values.device == self.device
        sample_count = len(batch.policy_versions)
        assert sample_count > 0
        version_result = _single_policy_version(batch.policy_versions)
        if isinstance(version_result, Rejected):
            return _rejected_decisions(
                reason=version_result.reason, count=sample_count
            )
        policy_version = version_result.value
        if (
            self._policy_version is not None
            and self._policy_version != policy_version
        ):
            return _rejected_decisions(
                reason="sample slab policy version mismatch",
                count=sample_count,
            )
        if self._policy_version is None:
            self._policy_version = policy_version
        self._ensure_capacity_for_batch(
            batch=batch, sample_count=sample_count
        )
        self._validate_batch_shape(batch)
        start = self._row_count
        end = start + sample_count
        self._write_sampled_batch(start=start, end=end, batch=batch)
        self._row_count = end
        step_counts = _int_tensor_tuple(batch.step_counts)
        choice_counts = _int_tensor_tuple(batch.choice_counts)
        row_indices = tuple(range(start, end))
        return _stored_decisions(
            model_rank_index=self.model_rank_index,
            policy_versions=batch.policy_versions,
            step_counts=step_counts,
            choice_counts=choice_counts,
            traces=_response_trace_cpu_view(
                batch=batch,
                step_counts=step_counts,
            ),
            row_indices=row_indices,
        )

    def ppo_batch_source(
        self, *, returns: RankReturnTargets
    ) -> _result.Ok[ArenaPPOBatchSource] | _result.Rejected:
        """Resolve committed row handles into an arena-backed source."""
        if returns.is_empty():
            return Rejected(reason="return commit has no decisions")
        if returns.model_rank_index != self.model_rank_index:
            return Rejected(
                reason="return batch targets the wrong model rank"
            )
        row_indices = returns.row_indices.to(
            dtype=torch.long, device=self.device
        )
        step_counts = returns.step_counts.to(
            dtype=torch.long, device=self.device
        )
        validation_result = self._validate_return_rows(
            policy_version=returns.policy_version,
            rows=row_indices,
            step_counts=step_counts,
        )
        if isinstance(validation_result, Rejected):
            return validation_result
        old_values = self._old_values_tensor().index_select(
            dim=0, index=row_indices
        )
        return_values = returns.return_values.to(
            dtype=torch.float32, device=self.device
        )
        old_log_probabilities = (
            self._old_log_probabilities_tensor().index_select(
                dim=0, index=row_indices
            )
        )
        return _result.Ok(
            value=ArenaPPOBatchSource(
                arena=self,
                policy_version=returns.policy_version,
                model_rank_index=self.model_rank_index,
                row_indices=row_indices,
                step_counts=step_counts,
                old_log_probabilities=old_log_probabilities,
                old_values=old_values,
                return_values=return_values,
                raw_advantages=return_values - old_values,
                max_step_count=returns.max_step_count,
            )
        )

    def discard_return_batch(
        self, *, returns: RankReturnTargets
    ) -> None:
        """Ignore row-level discard for append-only slab storage."""
        if returns.model_rank_index != self.model_rank_index:
            return

    def discard_uncommitted_policy_version(
        self, *, policy_version: int
    ) -> None:
        """Release all rows for a completed policy version."""
        assert policy_version >= 0
        if self._policy_version == policy_version:
            self.clear()

    def select_ppo_minibatch(
        self,
        *,
        source: ArenaPPOBatchSource,
        indices: Tensor,
        advantages: Tensor,
        global_count: Tensor,
    ) -> TensorizedPPOMinibatch:
        """Return one PPO minibatch directly from slab rows."""
        assert indices.ndim == 1
        local_count = int(indices.shape[0])
        assert local_count > 0
        selected_rows = source.row_indices.index_select(0, indices)
        selected_step_counts = source.step_counts.index_select(
            0, indices
        )
        replay = PPOReplayTensorBatch(
            sample_count=local_count,
            max_step_count=source.max_step_count,
            selected_token_ids_padded=(
                self._selected_token_ids_tensor().index_select(
                    dim=0, index=selected_rows
                )[:, : source.max_step_count]
            ),
            choice_token_ids=(
                self._choice_token_ids_tensor().index_select(
                    dim=0, index=selected_rows
                )[:, : source.max_step_count, :]
            ),
            choice_masks=(
                self._choice_masks_tensor().index_select(
                    dim=0, index=selected_rows
                )[:, : source.max_step_count, :]
            ),
            selected_choice_offsets=(
                self._selected_choice_offsets_tensor().index_select(
                    dim=0, index=selected_rows
                )[:, : source.max_step_count]
            ),
            step_mask=_step_mask(
                step_counts=selected_step_counts,
                max_step_count=source.max_step_count,
            ),
            step_counts=selected_step_counts,
        )
        return TensorizedPPOMinibatch(
            observation_batch=ObservationTensorBatch(
                component_ids=self._component_ids_tensor().index_select(
                    dim=0, index=selected_rows
                ),
                numeric_values=self._numeric_values_tensor().index_select(
                    dim=0, index=selected_rows
                ),
                numeric_masks=self._numeric_masks_tensor().index_select(
                    dim=0, index=selected_rows
                ),
            ),
            replay=replay,
            sample_indices=indices,
            old_log_probabilities=source.old_log_probabilities.index_select(
                dim=0, index=indices
            ),
            old_values=source.old_values.index_select(
                dim=0, index=indices
            ),
            advantages=advantages.index_select(dim=0, index=indices),
            return_values=source.return_values.index_select(
                dim=0, index=indices
            ),
            local_count=local_count,
            global_count=global_count,
        )

    def _ensure_capacity_for_batch(
        self, *, batch: SampledPolicyBatch, sample_count: int
    ) -> None:
        assert sample_count > 0
        needed = self._row_count + sample_count
        if self._capacity == 0:
            self._initialize_tensors(
                batch=batch, capacity=max(_INITIAL_CAPACITY, needed)
            )
        self._ensure_observation_token_capacity(batch)
        self._ensure_choice_capacity(batch)
        while self._capacity < needed:
            self._grow_rows()

    def _initialize_tensors(
        self, *, batch: SampledPolicyBatch, capacity: int
    ) -> None:
        assert capacity > 0
        self._capacity = capacity
        observation = batch.observation_batch
        token_count = int(observation.component_ids.shape[1])
        component_count = int(observation.component_ids.shape[2])
        numeric_count = int(observation.numeric_values.shape[2])
        choice_width = int(batch.choice_token_ids.shape[2])
        self._row_policy_versions = torch.zeros(
            (capacity,), dtype=torch.long, device=self.device
        )
        self._row_step_counts = torch.zeros(
            (capacity,), dtype=torch.long, device=self.device
        )
        self._component_ids = torch.empty(
            (capacity, token_count, component_count),
            dtype=observation.component_ids.dtype,
            device=self.device,
        )
        self._numeric_values = torch.empty(
            (capacity, token_count, numeric_count),
            dtype=observation.numeric_values.dtype,
            device=self.device,
        )
        self._numeric_masks = torch.empty(
            (capacity, token_count, numeric_count),
            dtype=observation.numeric_masks.dtype,
            device=self.device,
        )
        self._selected_token_ids = torch.zeros(
            (capacity, SEMANTIC_CODEC.max_argument_tokens),
            dtype=torch.long,
            device=self.device,
        )
        self._choice_token_ids = torch.zeros(
            (
                capacity,
                SEMANTIC_CODEC.max_argument_tokens,
                choice_width,
            ),
            dtype=torch.int16,
            device=self.device,
        )
        self._choice_masks = torch.zeros(
            self._choice_token_ids.shape,
            dtype=torch.bool,
            device=self.device,
        )
        self._selected_choice_offsets = torch.zeros(
            (capacity, SEMANTIC_CODEC.max_argument_tokens),
            dtype=torch.long,
            device=self.device,
        )
        self._old_log_probabilities = torch.zeros(
            (capacity,), dtype=torch.float32, device=self.device
        )
        self._old_values = torch.zeros(
            (capacity,), dtype=torch.float32, device=self.device
        )

    def _grow_rows(self) -> None:
        old_capacity = self._capacity
        new_capacity = old_capacity * 2
        self._row_policy_versions = _grow_rows(
            self._row_policy_versions_tensor(),
            new_capacity=new_capacity,
        )
        self._row_step_counts = _grow_rows(
            self._row_step_counts_tensor(), new_capacity=new_capacity
        )
        self._component_ids = _grow_rows(
            self._component_ids_tensor(), new_capacity=new_capacity
        )
        self._numeric_values = _grow_rows(
            self._numeric_values_tensor(), new_capacity=new_capacity
        )
        self._numeric_masks = _grow_rows(
            self._numeric_masks_tensor(), new_capacity=new_capacity
        )
        self._selected_token_ids = _grow_rows(
            self._selected_token_ids_tensor(),
            new_capacity=new_capacity,
        )
        self._choice_token_ids = _grow_rows(
            self._choice_token_ids_tensor(), new_capacity=new_capacity
        )
        self._choice_masks = _grow_rows(
            self._choice_masks_tensor(), new_capacity=new_capacity
        )
        self._selected_choice_offsets = _grow_rows(
            self._selected_choice_offsets_tensor(),
            new_capacity=new_capacity,
        )
        self._old_log_probabilities = _grow_rows(
            self._old_log_probabilities_tensor(),
            new_capacity=new_capacity,
        )
        self._old_values = _grow_rows(
            self._old_values_tensor(), new_capacity=new_capacity
        )
        self._capacity = new_capacity

    def _ensure_observation_token_capacity(
        self, batch: SampledPolicyBatch
    ) -> None:
        token_count = int(
            batch.observation_batch.component_ids.shape[1]
        )
        current_token_count = int(self._component_ids_tensor().shape[1])
        if token_count <= current_token_count:
            return
        self._component_ids = _grow_observation_tokens(
            self._component_ids_tensor(), token_count=token_count
        )
        self._numeric_values = _grow_observation_tokens(
            self._numeric_values_tensor(), token_count=token_count
        )
        self._numeric_masks = _grow_observation_tokens(
            self._numeric_masks_tensor(), token_count=token_count
        )

    def _ensure_choice_capacity(
        self, batch: SampledPolicyBatch
    ) -> None:
        choice_width = int(batch.choice_token_ids.shape[2])
        current_width = int(self._choice_token_ids_tensor().shape[2])
        if choice_width <= current_width:
            return
        self._choice_token_ids = _grow_choice_width(
            self._choice_token_ids_tensor(), choice_width=choice_width
        )
        self._choice_masks = _grow_choice_width(
            self._choice_masks_tensor(), choice_width=choice_width
        )

    def _validate_batch_shape(self, batch: SampledPolicyBatch) -> None:
        max_generation_steps = int(
            batch.selected_token_ids_padded.shape[1]
        )
        assert max_generation_steps > 0
        assert (
            max_generation_steps <= SEMANTIC_CODEC.max_argument_tokens
        )
        assert batch.choice_token_ids.ndim == 3
        assert int(batch.choice_token_ids.shape[1]) == (
            max_generation_steps
        )
        assert int(batch.choice_token_ids.shape[2]) <= int(
            self._choice_token_ids_tensor().shape[2]
        )
        assert batch.choice_masks.shape == batch.choice_token_ids.shape
        assert (
            int(batch.selected_choice_offsets.shape[1])
            == max_generation_steps
        )
        assert (
            batch.observation_batch.component_ids.shape[2:]
            == (self._component_ids_tensor().shape[2:])
        )
        assert (
            batch.observation_batch.numeric_values.shape[2:]
            == (self._numeric_values_tensor().shape[2:])
        )
        assert (
            batch.observation_batch.numeric_masks.shape[2:]
            == (self._numeric_masks_tensor().shape[2:])
        )

    def _write_sampled_batch(
        self, *, start: int, end: int, batch: SampledPolicyBatch
    ) -> None:
        assert 0 <= start < end <= self._capacity
        row_slice = slice(start, end)
        sample_count = end - start
        assert sample_count == len(batch.policy_versions)
        max_generation_steps = int(
            batch.selected_token_ids_padded.shape[1]
        )
        choice_width = int(batch.choice_token_ids.shape[2])
        self._row_policy_versions_tensor()[row_slice] = torch.tensor(
            batch.policy_versions, dtype=torch.long, device=self.device
        )
        self._row_step_counts_tensor()[row_slice].copy_(
            batch.step_counts
        )
        _write_observation_rows(
            destination=self._component_ids_tensor(),
            row_slice=row_slice,
            source=batch.observation_batch.component_ids,
        )
        _write_observation_rows(
            destination=self._numeric_values_tensor(),
            row_slice=row_slice,
            source=batch.observation_batch.numeric_values,
        )
        _write_observation_rows(
            destination=self._numeric_masks_tensor(),
            row_slice=row_slice,
            source=batch.observation_batch.numeric_masks,
        )
        self._selected_token_ids_tensor()[row_slice].zero_()
        self._selected_token_ids_tensor()[
            row_slice, :max_generation_steps
        ].copy_(batch.selected_token_ids_padded)
        self._choice_token_ids_tensor()[row_slice].zero_()
        self._choice_masks_tensor()[row_slice].fill_(False)
        self._choice_token_ids_tensor()[
            row_slice, :max_generation_steps, :choice_width
        ].copy_(batch.choice_token_ids)
        self._choice_masks_tensor()[
            row_slice, :max_generation_steps, :choice_width
        ].copy_(batch.choice_masks)
        self._selected_choice_offsets_tensor()[row_slice].zero_()
        self._selected_choice_offsets_tensor()[
            row_slice, :max_generation_steps
        ].copy_(batch.selected_choice_offsets)
        self._old_log_probabilities_tensor()[row_slice].copy_(
            batch.old_log_probabilities
        )
        self._old_values_tensor()[row_slice].copy_(batch.old_values)

    def _validate_return_rows(
        self,
        *,
        policy_version: int,
        rows: Tensor,
        step_counts: Tensor,
    ) -> _result.Ok[None] | _result.Rejected:
        assert rows.ndim == 1
        assert step_counts.shape == rows.shape
        if int(rows.shape[0]) == 0:
            return Rejected(reason="return commit has no decisions")
        if self._row_count == 0:
            return Rejected(
                reason="return commit references missing replay"
            )
        in_range = (rows >= 0) & (rows < self._row_count)
        safe_rows = rows.clamp(min=0, max=max(self._row_count - 1, 0))
        if not _bool_tensor_value(in_range.all()):
            return Rejected(
                reason="return commit references missing replay"
            )
        versions = self._row_policy_versions_tensor().index_select(
            dim=0, index=safe_rows
        )
        expected = torch.full_like(versions, policy_version)
        if not _bool_tensor_value((versions == expected).all()):
            return Rejected(
                reason="replay record policy version mismatch"
            )
        stored_step_counts = (
            self._row_step_counts_tensor().index_select(
                dim=0, index=safe_rows
            )
        )
        if not _bool_tensor_value(
            (stored_step_counts == step_counts).all()
        ):
            return Rejected(
                reason="return commit replay step count mismatch"
            )
        return Ok(value=None)

    def _row_policy_versions_tensor(self) -> Tensor:
        assert self._row_policy_versions is not None
        return self._row_policy_versions

    def _row_step_counts_tensor(self) -> Tensor:
        assert self._row_step_counts is not None
        return self._row_step_counts

    def _component_ids_tensor(self) -> Tensor:
        assert self._component_ids is not None
        return self._component_ids

    def _numeric_values_tensor(self) -> Tensor:
        assert self._numeric_values is not None
        return self._numeric_values

    def _numeric_masks_tensor(self) -> Tensor:
        assert self._numeric_masks is not None
        return self._numeric_masks

    def _selected_token_ids_tensor(self) -> Tensor:
        assert self._selected_token_ids is not None
        return self._selected_token_ids

    def _choice_token_ids_tensor(self) -> Tensor:
        assert self._choice_token_ids is not None
        return self._choice_token_ids

    def _choice_masks_tensor(self) -> Tensor:
        assert self._choice_masks is not None
        return self._choice_masks

    def _selected_choice_offsets_tensor(self) -> Tensor:
        assert self._selected_choice_offsets is not None
        return self._selected_choice_offsets

    def _old_log_probabilities_tensor(self) -> Tensor:
        assert self._old_log_probabilities is not None
        return self._old_log_probabilities

    def _old_values_tensor(self) -> Tensor:
        assert self._old_values is not None
        return self._old_values


def _single_policy_version(
    policy_versions: tuple[int, ...],
) -> Ok[int] | Rejected:
    assert policy_versions
    policy_version = policy_versions[0]
    if any(version != policy_version for version in policy_versions):
        return Rejected(reason="sample batch mixes policy versions")
    return Ok(value=policy_version)


def _rejected_decisions(
    *, reason: str, count: int
) -> tuple[ModelRankDecisionResult, ...]:
    assert reason
    assert count > 0
    return tuple(Rejected(reason=reason) for _ in range(count))


def _grow_rows(values: Tensor, *, new_capacity: int) -> Tensor:
    assert new_capacity > int(values.shape[0])
    result = torch.zeros(
        (new_capacity, *tuple(int(size) for size in values.shape[1:])),
        dtype=values.dtype,
        device=values.device,
    )
    result[: int(values.shape[0])].copy_(values)
    return result


def _write_observation_rows(
    *, destination: Tensor, row_slice: slice, source: Tensor
) -> None:
    assert destination.ndim == 3
    assert source.ndim == 3
    assert int(source.shape[1]) <= int(destination.shape[1])
    assert int(source.shape[2]) == int(destination.shape[2])
    destination[row_slice].zero_()
    destination[row_slice, : int(source.shape[1]), :].copy_(source)


def _grow_observation_tokens(
    values: Tensor, *, token_count: int
) -> Tensor:
    assert token_count > int(values.shape[1])
    result = torch.zeros(
        (
            int(values.shape[0]),
            token_count,
            int(values.shape[2]),
        ),
        dtype=values.dtype,
        device=values.device,
    )
    result[:, : int(values.shape[1]), :].copy_(values)
    return result


def _grow_choice_width(values: Tensor, *, choice_width: int) -> Tensor:
    assert choice_width > int(values.shape[2])
    result = torch.zeros(
        (
            int(values.shape[0]),
            int(values.shape[1]),
            choice_width,
        ),
        dtype=values.dtype,
        device=values.device,
    )
    result[:, :, : int(values.shape[2])].copy_(values)
    return result


def _step_mask(*, step_counts: Tensor, max_step_count: int) -> Tensor:
    positions = torch.arange(
        max_step_count, dtype=torch.long, device=step_counts.device
    ).unsqueeze(0)
    return positions < step_counts.unsqueeze(1)


def _bool_tensor_value(value: Tensor) -> bool:
    assert value.shape == ()
    return bool(value.detach().cpu().item())


def _int_tensor_tuple(values: Tensor) -> tuple[int, ...]:
    cpu_values = values.detach().cpu()
    return tuple(
        int(cpu_values[index].item())
        for index in range(int(cpu_values.shape[0]))
    )


def _stored_decisions(
    *,
    model_rank_index: int,
    policy_versions: tuple[int, ...],
    step_counts: tuple[int, ...],
    choice_counts: tuple[int, ...],
    traces: _ResponseTraceCpuView,
    row_indices: tuple[int, ...],
) -> tuple[ModelRankDecisionResult, ...]:
    assert len(policy_versions) == len(row_indices)
    assert len(step_counts) == len(row_indices)
    assert len(choice_counts) == len(row_indices)
    return tuple(
        _stored_decision_for_position(
            model_rank_index=model_rank_index,
            policy_versions=policy_versions,
            step_counts=step_counts,
            choice_counts=choice_counts,
            traces=traces,
            position=position,
            row_index=row_indices[position],
        )
        for position in range(len(policy_versions))
    )


def _stored_decision_for_position(
    *,
    model_rank_index: int,
    policy_versions: tuple[int, ...],
    step_counts: tuple[int, ...],
    choice_counts: tuple[int, ...],
    traces: _ResponseTraceCpuView,
    position: int,
    row_index: int,
) -> ModelRankDecisionResult:
    step_count = step_counts[position]
    return Ok(
        value=ModelRankPolicyDecision(
            trace_token_ids=_compact_trace_token_ids(
                traces=traces,
                position=position,
                step_count=step_count,
            ),
            decision_handle=DecisionHandle(
                model_rank_index=model_rank_index,
                policy_version=policy_versions[position],
                row_index=row_index,
            ),
            choice_count=choice_counts[position],
        )
    )


def _response_trace_cpu_view(
    *,
    batch: SampledPolicyBatch,
    step_counts: tuple[int, ...],
) -> _ResponseTraceCpuView:
    max_step_count = max(step_counts)
    assert max_step_count > 0
    selected_tokens = (
        batch.selected_token_ids_padded[:, :max_step_count]
        .detach()
        .cpu()
    )
    return _ResponseTraceCpuView(selected_token_ids=selected_tokens)


def _compact_trace_token_ids(
    *, traces: _ResponseTraceCpuView, position: int, step_count: int
) -> CompactTraceTokenIds:
    assert step_count > 0
    return CompactTraceTokenIds.from_cpu_tensor(
        tokens=traces.selected_token_ids[position],
        count=step_count,
    )
