"""Model-rank-owned append-only sample slab for PPO replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, cast

import torch
from torch import Tensor

from server import result as _result
from server.result import Ok, Rejected
from server.training.policy_sampling.records import (
    CompactPolicyDecisionBatch,
    CompactTraceTokenBatch,
    RankReturnTargets,
)
from server.training.ppo.minibatch import TensorizedPPOMinibatch
from server.training.ppo.replay_tensors import PPOReplayTensorBatch
from server.training.semantic_action_plan import (
    SemanticActionSampleBatch,
)
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tensorize import ObservationTensorBatch

_INITIAL_CAPACITY = 256

type ModelRankDecisionBatchResult = (
    _result.Ok[CompactPolicyDecisionBatch] | _result.Rejected
)


class _IntTensorListable(Protocol):
    def tolist(self) -> list[int]: ...


def _arena_minibatch_workspace() -> "_ArenaMinibatchWorkspace":
    return _ArenaMinibatchWorkspace()


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
    workspace: "_ArenaMinibatchWorkspace" = field(
        default_factory=_arena_minibatch_workspace
    )

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
class _ArenaMinibatchWorkspace:
    _component_ids: Tensor | None = None
    _numeric_values: Tensor | None = None
    _numeric_masks: Tensor | None = None
    _selected_token_ids: Tensor | None = None
    _choice_token_ids: Tensor | None = None
    _choice_masks: Tensor | None = None
    _selected_choice_offsets: Tensor | None = None
    _active_sample_indices: Tensor | None = None
    _active_step_indices: Tensor | None = None
    _source_step_indices: Tensor | None = None
    _step_counts: Tensor | None = None
    _old_log_probabilities: Tensor | None = None
    _old_values: Tensor | None = None
    _advantages: Tensor | None = None
    _return_values: Tensor | None = None

    def materialize(
        self,
        *,
        component_ids_source: Tensor,
        numeric_values_source: Tensor,
        numeric_masks_source: Tensor,
        selected_token_ids_source: Tensor,
        choice_token_ids_source: Tensor,
        choice_masks_source: Tensor,
        selected_choice_offsets_source: Tensor,
        device: torch.device,
        selected_rows: Tensor,
        selected_step_counts: Tensor,
        selected_step_offsets: Tensor,
        source: ArenaPPOBatchSource,
        indices: Tensor,
        advantages: Tensor,
        global_count: Tensor,
    ) -> TensorizedPPOMinibatch:
        """Copy one shuffled minibatch into reusable tensor storage."""
        local_count = int(indices.shape[0])
        assert local_count > 0
        self._ensure(
            component_ids_source=component_ids_source,
            numeric_values_source=numeric_values_source,
            numeric_masks_source=numeric_masks_source,
            choice_token_ids_source=choice_token_ids_source,
            device=device,
            local_count=local_count,
            max_step_count=source.max_step_count,
        )
        self._write_active_step_indices(
            selected_step_counts=selected_step_counts,
            selected_step_offsets=selected_step_offsets,
            local_count=local_count,
            max_step_count=source.max_step_count,
        )
        active_step_count = int(
            self._source_step_indices_tensor().shape[0]
        )
        self._ensure_active_replay(
            choice_token_ids_source=choice_token_ids_source,
            device=device,
            active_step_count=active_step_count,
        )
        component_ids = self._component_ids_tensor()[:local_count]
        numeric_values = self._numeric_values_tensor()[:local_count]
        numeric_masks = self._numeric_masks_tensor()[:local_count]
        selected_token_ids = self._selected_token_ids_tensor()[
            :local_count, : source.max_step_count
        ]
        choice_width = int(choice_token_ids_source.shape[1])
        choice_token_ids = self._choice_token_ids_tensor()[
            :active_step_count, :choice_width
        ]
        choice_masks = self._choice_masks_tensor()[
            :active_step_count, :choice_width
        ]
        selected_choice_offsets = (
            self._selected_choice_offsets_tensor()[:active_step_count]
        )
        active_sample_indices = self._active_sample_indices_tensor()[
            :active_step_count
        ]
        active_step_indices = self._active_step_indices_tensor()[
            :active_step_count
        ]
        source_step_indices = self._source_step_indices_tensor()[
            :active_step_count
        ]
        step_counts = self._step_counts_tensor()[:local_count]
        old_log_probabilities = self._old_log_probabilities_tensor()[
            :local_count
        ]
        old_values = self._old_values_tensor()[:local_count]
        advantage_values = self._advantages_tensor()[:local_count]
        return_values = self._return_values_tensor()[:local_count]

        torch.index_select(
            component_ids_source,
            dim=0,
            index=selected_rows,
            out=component_ids,
        )
        torch.index_select(
            numeric_values_source,
            dim=0,
            index=selected_rows,
            out=numeric_values,
        )
        torch.index_select(
            numeric_masks_source,
            dim=0,
            index=selected_rows,
            out=numeric_masks,
        )
        torch.index_select(
            selected_token_ids_source[:, : source.max_step_count],
            dim=0,
            index=selected_rows,
            out=selected_token_ids,
        )
        torch.index_select(
            choice_token_ids_source,
            dim=0,
            index=source_step_indices,
            out=choice_token_ids,
        )
        torch.index_select(
            choice_masks_source,
            dim=0,
            index=source_step_indices,
            out=choice_masks,
        )
        torch.index_select(
            selected_choice_offsets_source,
            dim=0,
            index=source_step_indices,
            out=selected_choice_offsets,
        )
        step_counts.copy_(selected_step_counts)
        torch.index_select(
            source.old_log_probabilities,
            dim=0,
            index=indices,
            out=old_log_probabilities,
        )
        torch.index_select(
            source.old_values,
            dim=0,
            index=indices,
            out=old_values,
        )
        torch.index_select(
            advantages,
            dim=0,
            index=indices,
            out=advantage_values,
        )
        torch.index_select(
            source.return_values,
            dim=0,
            index=indices,
            out=return_values,
        )
        return TensorizedPPOMinibatch(
            observation_batch=ObservationTensorBatch(
                component_ids=component_ids,
                numeric_values=numeric_values,
                numeric_masks=numeric_masks,
            ),
            replay=PPOReplayTensorBatch(
                sample_count=local_count,
                max_step_count=source.max_step_count,
                active_step_count=active_step_count,
                selected_token_ids_padded=selected_token_ids,
                active_sample_indices=active_sample_indices,
                active_step_indices=active_step_indices,
                choice_token_ids=choice_token_ids,
                choice_masks=choice_masks,
                selected_choice_offsets=selected_choice_offsets,
                step_counts=step_counts,
            ),
            sample_indices=indices,
            old_log_probabilities=old_log_probabilities,
            old_values=old_values,
            advantages=advantage_values,
            return_values=return_values,
            local_count=local_count,
            global_count=global_count,
        )

    def _ensure(
        self,
        *,
        component_ids_source: Tensor,
        numeric_values_source: Tensor,
        numeric_masks_source: Tensor,
        choice_token_ids_source: Tensor,
        device: torch.device,
        local_count: int,
        max_step_count: int,
    ) -> None:
        assert local_count > 0
        assert max_step_count > 0
        component_shape = (
            local_count,
            int(component_ids_source.shape[1]),
            int(component_ids_source.shape[2]),
        )
        numeric_shape = (
            local_count,
            int(numeric_values_source.shape[1]),
            int(numeric_values_source.shape[2]),
        )
        token_shape = (local_count, max_step_count)
        self._component_ids = _ensure_tensor(
            self._component_ids,
            shape=component_shape,
            dtype=component_ids_source.dtype,
            device=device,
        )
        self._numeric_values = _ensure_tensor(
            self._numeric_values,
            shape=numeric_shape,
            dtype=numeric_values_source.dtype,
            device=device,
        )
        self._numeric_masks = _ensure_tensor(
            self._numeric_masks,
            shape=numeric_shape,
            dtype=numeric_masks_source.dtype,
            device=device,
        )
        self._selected_token_ids = _ensure_tensor(
            self._selected_token_ids,
            shape=token_shape,
            dtype=torch.long,
            device=device,
        )
        self._step_counts = _ensure_tensor(
            self._step_counts,
            shape=(local_count,),
            dtype=torch.long,
            device=device,
        )
        self._old_log_probabilities = _ensure_tensor(
            self._old_log_probabilities,
            shape=(local_count,),
            dtype=torch.float32,
            device=device,
        )
        self._old_values = _ensure_tensor(
            self._old_values,
            shape=(local_count,),
            dtype=torch.float32,
            device=device,
        )
        self._advantages = _ensure_tensor(
            self._advantages,
            shape=(local_count,),
            dtype=torch.float32,
            device=device,
        )
        self._return_values = _ensure_tensor(
            self._return_values,
            shape=(local_count,),
            dtype=torch.float32,
            device=device,
        )

    def _ensure_active_replay(
        self,
        *,
        choice_token_ids_source: Tensor,
        device: torch.device,
        active_step_count: int,
    ) -> None:
        assert active_step_count >= 0
        choice_width = int(choice_token_ids_source.shape[1])
        choice_shape = (
            active_step_count,
            choice_width,
        )
        self._choice_token_ids = _ensure_min_capacity_tensor(
            self._choice_token_ids,
            shape=choice_shape,
            dtype=torch.int16,
            device=device,
        )
        self._choice_masks = _ensure_min_capacity_tensor(
            self._choice_masks,
            shape=choice_shape,
            dtype=torch.bool,
            device=device,
        )
        self._selected_choice_offsets = _ensure_min_capacity_tensor(
            self._selected_choice_offsets,
            shape=(active_step_count,),
            dtype=torch.long,
            device=device,
        )

    def _write_active_step_indices(
        self,
        *,
        selected_step_counts: Tensor,
        selected_step_offsets: Tensor,
        local_count: int,
        max_step_count: int,
    ) -> None:
        positions = torch.arange(
            max_step_count,
            dtype=torch.long,
            device=selected_step_counts.device,
        ).unsqueeze(0)
        step_positions = positions.expand(local_count, -1)
        active_mask = step_positions < selected_step_counts.unsqueeze(1)
        active_positions = torch.nonzero(
            active_mask.reshape(-1), as_tuple=False
        ).flatten()
        self._active_sample_indices = torch.div(
            active_positions,
            max_step_count,
            rounding_mode="floor",
        )
        self._active_step_indices = active_positions.remainder(
            max_step_count
        )
        self._source_step_indices = (
            selected_step_offsets.index_select(
                0, self._active_sample_indices
            )
            + self._active_step_indices
        )

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

    def _active_sample_indices_tensor(self) -> Tensor:
        assert self._active_sample_indices is not None
        return self._active_sample_indices

    def _active_step_indices_tensor(self) -> Tensor:
        assert self._active_step_indices is not None
        return self._active_step_indices

    def _source_step_indices_tensor(self) -> Tensor:
        assert self._source_step_indices is not None
        return self._source_step_indices

    def _step_counts_tensor(self) -> Tensor:
        assert self._step_counts is not None
        return self._step_counts

    def _old_log_probabilities_tensor(self) -> Tensor:
        assert self._old_log_probabilities is not None
        return self._old_log_probabilities

    def _old_values_tensor(self) -> Tensor:
        assert self._old_values is not None
        return self._old_values

    def _advantages_tensor(self) -> Tensor:
        assert self._advantages is not None
        return self._advantages

    def _return_values_tensor(self) -> Tensor:
        assert self._return_values is not None
        return self._return_values


@dataclass(slots=True)
class ModelRankSampleArena:
    """Append sampled policy rows and expose committed PPO batches."""

    model_rank_index: int
    device: torch.device
    _capacity: int = 0
    _step_capacity: int = 0
    _row_count: int = 0
    _step_count: int = 0
    _policy_version: int | None = None
    _row_policy_versions: Tensor | None = None
    _row_step_counts: Tensor | None = None
    _row_step_offsets: Tensor | None = None
    _component_ids: Tensor | None = None
    _numeric_values: Tensor | None = None
    _numeric_masks: Tensor | None = None
    _selected_token_ids: Tensor | None = None
    _flat_choice_token_ids: Tensor | None = None
    _flat_choice_masks: Tensor | None = None
    _flat_selected_choice_offsets: Tensor | None = None
    _old_log_probabilities: Tensor | None = None
    _old_values: Tensor | None = None

    def __post_init__(self) -> None:
        assert self.model_rank_index >= 0

    def clear(self) -> None:
        """Make every slab row reusable after state sync or update."""
        self._row_count = 0
        self._step_count = 0
        self._policy_version = None

    def store_sampled_result(
        self,
        *,
        policy_versions: tuple[int, ...],
        observation_batch: ObservationTensorBatch,
        semantic_sample: SemanticActionSampleBatch,
        old_values: Tensor,
    ) -> ModelRankDecisionBatchResult:
        """Append sampled tensors and return decisions."""
        assert old_values.device == self.device
        sample_count = len(policy_versions)
        assert sample_count > 0
        version_result = _single_policy_version(policy_versions)
        if isinstance(version_result, Rejected):
            return version_result
        policy_version = version_result.value
        if (
            self._policy_version is not None
            and self._policy_version != policy_version
        ):
            return Rejected(
                reason="sample slab policy version mismatch"
            )
        if self._policy_version is None:
            self._policy_version = policy_version
        self._ensure_capacity_for_sample(
            observation_batch=observation_batch,
            semantic_sample=semantic_sample,
            sample_count=sample_count,
        )
        self._validate_sample_shape(
            observation_batch=observation_batch,
            semantic_sample=semantic_sample,
            old_values=old_values,
        )
        start = self._row_count
        end = start + sample_count
        self._write_sampled_result(
            start=start,
            end=end,
            policy_versions=policy_versions,
            observation_batch=observation_batch,
            semantic_sample=semantic_sample,
            old_values=old_values,
        )
        self._row_count = end
        step_counts = _int_tensor_tuple(semantic_sample.step_counts)
        choice_counts = _int_tensor_tuple(semantic_sample.choice_counts)
        row_indices = tuple(range(start, end))
        return Ok(
            value=CompactPolicyDecisionBatch(
                model_rank_index=self.model_rank_index,
                policy_versions=policy_versions,
                row_indices=row_indices,
                choice_counts=choice_counts,
                trace_token_batch=CompactTraceTokenBatch.from_cpu_tensor(
                    tokens=_response_trace_cpu_tensor(
                        selected_token_ids=(
                            semantic_sample.selected_token_ids_padded
                        ),
                        step_counts=step_counts,
                    ),
                    trace_counts=step_counts,
                ),
            )
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
        selected_step_offsets = (
            self._row_step_offsets_tensor().index_select(
                0, selected_rows
            )
        )
        return source.workspace.materialize(
            component_ids_source=self._component_ids_tensor(),
            numeric_values_source=self._numeric_values_tensor(),
            numeric_masks_source=self._numeric_masks_tensor(),
            selected_token_ids_source=self._selected_token_ids_tensor(),
            choice_token_ids_source=self._flat_choice_token_ids_tensor(),
            choice_masks_source=self._flat_choice_masks_tensor(),
            selected_choice_offsets_source=(
                self._flat_selected_choice_offsets_tensor()
            ),
            device=self.device,
            selected_rows=selected_rows,
            selected_step_counts=selected_step_counts,
            selected_step_offsets=selected_step_offsets,
            source=source,
            indices=indices,
            advantages=advantages,
            global_count=global_count,
        )

    def _ensure_capacity_for_sample(
        self,
        *,
        observation_batch: ObservationTensorBatch,
        semantic_sample: SemanticActionSampleBatch,
        sample_count: int,
    ) -> None:
        assert sample_count > 0
        needed = self._row_count + sample_count
        step_needed = self._step_count + int(
            semantic_sample.choice_token_ids.shape[0]
        )
        if self._capacity == 0:
            self._initialize_tensors_from_sample(
                observation_batch=observation_batch,
                semantic_sample=semantic_sample,
                capacity=max(_INITIAL_CAPACITY, needed),
                step_capacity=max(_INITIAL_CAPACITY, step_needed, 1),
            )
        self._ensure_observation_token_capacity_for_sample(
            observation_batch
        )
        self._ensure_choice_capacity_for_sample(semantic_sample)
        while self._capacity < needed:
            self._grow_rows()
        while self._step_capacity < step_needed:
            self._grow_steps()

    def _initialize_tensors_from_sample(
        self,
        *,
        observation_batch: ObservationTensorBatch,
        semantic_sample: SemanticActionSampleBatch,
        capacity: int,
        step_capacity: int,
    ) -> None:
        assert capacity > 0
        assert step_capacity > 0
        self._capacity = capacity
        self._step_capacity = step_capacity
        observation = observation_batch
        token_count = int(observation.component_ids.shape[1])
        component_count = int(observation.component_ids.shape[2])
        numeric_count = int(observation.numeric_values.shape[2])
        choice_width = int(semantic_sample.choice_token_ids.shape[1])
        self._row_policy_versions = torch.zeros(
            (capacity,), dtype=torch.long, device=self.device
        )
        self._row_step_counts = torch.zeros(
            (capacity,), dtype=torch.long, device=self.device
        )
        self._row_step_offsets = torch.zeros(
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
        self._flat_choice_token_ids = torch.zeros(
            (step_capacity, choice_width),
            dtype=torch.int16,
            device=self.device,
        )
        self._flat_choice_masks = torch.zeros(
            self._flat_choice_token_ids.shape,
            dtype=torch.bool,
            device=self.device,
        )
        self._flat_selected_choice_offsets = torch.zeros(
            (step_capacity,),
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
        self._row_step_offsets = _grow_rows(
            self._row_step_offsets_tensor(), new_capacity=new_capacity
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
        self._old_log_probabilities = _grow_rows(
            self._old_log_probabilities_tensor(),
            new_capacity=new_capacity,
        )
        self._old_values = _grow_rows(
            self._old_values_tensor(), new_capacity=new_capacity
        )
        self._capacity = new_capacity

    def _grow_steps(self) -> None:
        old_capacity = self._step_capacity
        new_capacity = old_capacity * 2
        self._flat_choice_token_ids = _grow_rows(
            self._flat_choice_token_ids_tensor(),
            new_capacity=new_capacity,
        )
        self._flat_choice_masks = _grow_rows(
            self._flat_choice_masks_tensor(), new_capacity=new_capacity
        )
        self._flat_selected_choice_offsets = _grow_rows(
            self._flat_selected_choice_offsets_tensor(),
            new_capacity=new_capacity,
        )
        self._step_capacity = new_capacity

    def _ensure_observation_token_capacity_for_sample(
        self, observation_batch: ObservationTensorBatch
    ) -> None:
        token_count = int(observation_batch.component_ids.shape[1])
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

    def _ensure_choice_capacity_for_sample(
        self, semantic_sample: SemanticActionSampleBatch
    ) -> None:
        choice_width = int(semantic_sample.choice_token_ids.shape[1])
        current_width = int(
            self._flat_choice_token_ids_tensor().shape[1]
        )
        if choice_width <= current_width:
            return
        self._flat_choice_token_ids = _grow_choice_width(
            self._flat_choice_token_ids_tensor(),
            choice_width=choice_width,
        )
        self._flat_choice_masks = _grow_choice_width(
            self._flat_choice_masks_tensor(), choice_width=choice_width
        )

    def _validate_sample_shape(
        self,
        *,
        observation_batch: ObservationTensorBatch,
        semantic_sample: SemanticActionSampleBatch,
        old_values: Tensor,
    ) -> None:
        max_generation_steps = int(
            semantic_sample.selected_token_ids_padded.shape[1]
        )
        assert max_generation_steps > 0
        assert (
            max_generation_steps <= SEMANTIC_CODEC.max_argument_tokens
        )
        assert semantic_sample.active_sample_indices.ndim == 1
        assert semantic_sample.active_step_indices.shape == (
            semantic_sample.active_sample_indices.shape
        )
        assert semantic_sample.choice_token_ids.ndim == 2
        assert int(semantic_sample.choice_token_ids.shape[1]) <= int(
            self._flat_choice_token_ids_tensor().shape[1]
        )
        assert (
            semantic_sample.choice_masks.shape
            == semantic_sample.choice_token_ids.shape
        )
        assert (
            semantic_sample.selected_choice_offsets.shape
            == semantic_sample.active_sample_indices.shape
        )
        assert (
            observation_batch.component_ids.shape[2:]
            == (self._component_ids_tensor().shape[2:])
        )
        assert (
            observation_batch.numeric_values.shape[2:]
            == (self._numeric_values_tensor().shape[2:])
        )
        assert (
            observation_batch.numeric_masks.shape[2:]
            == (self._numeric_masks_tensor().shape[2:])
        )
        assert (
            old_values.shape == semantic_sample.log_probabilities.shape
        )

    def _write_sampled_result(
        self,
        *,
        start: int,
        end: int,
        policy_versions: tuple[int, ...],
        observation_batch: ObservationTensorBatch,
        semantic_sample: SemanticActionSampleBatch,
        old_values: Tensor,
    ) -> None:
        assert 0 <= start < end <= self._capacity
        row_slice = slice(start, end)
        sample_count = end - start
        assert sample_count == len(policy_versions)
        max_generation_steps = int(
            semantic_sample.selected_token_ids_padded.shape[1]
        )
        active_step_count = int(
            semantic_sample.choice_token_ids.shape[0]
        )
        step_start = self._step_count
        step_end = step_start + active_step_count
        self._row_policy_versions_tensor()[row_slice] = torch.tensor(
            policy_versions, dtype=torch.long, device=self.device
        )
        self._row_step_counts_tensor()[row_slice].copy_(
            semantic_sample.step_counts
        )
        self._row_step_offsets_tensor()[row_slice].copy_(
            _row_step_offsets(
                step_start=step_start,
                step_counts=semantic_sample.step_counts,
            )
        )
        _write_observation_rows(
            destination=self._component_ids_tensor(),
            row_slice=row_slice,
            source=observation_batch.component_ids,
        )
        _write_observation_rows(
            destination=self._numeric_values_tensor(),
            row_slice=row_slice,
            source=observation_batch.numeric_values,
        )
        _write_observation_rows(
            destination=self._numeric_masks_tensor(),
            row_slice=row_slice,
            source=observation_batch.numeric_masks,
        )
        self._selected_token_ids_tensor()[row_slice].zero_()
        self._selected_token_ids_tensor()[
            row_slice, :max_generation_steps
        ].copy_(semantic_sample.selected_token_ids_padded)
        self._flat_choice_token_ids_tensor()[
            step_start:step_end, :
        ].copy_(semantic_sample.choice_token_ids)
        self._flat_choice_masks_tensor()[step_start:step_end, :].copy_(
            semantic_sample.choice_masks
        )
        self._flat_selected_choice_offsets_tensor()[
            step_start:step_end
        ].copy_(semantic_sample.selected_choice_offsets)
        self._old_log_probabilities_tensor()[row_slice].copy_(
            semantic_sample.log_probabilities
        )
        self._old_values_tensor()[row_slice].copy_(old_values)
        self._step_count = step_end

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

    def _row_step_offsets_tensor(self) -> Tensor:
        assert self._row_step_offsets is not None
        return self._row_step_offsets

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

    def _flat_choice_token_ids_tensor(self) -> Tensor:
        assert self._flat_choice_token_ids is not None
        return self._flat_choice_token_ids

    def _flat_choice_masks_tensor(self) -> Tensor:
        assert self._flat_choice_masks is not None
        return self._flat_choice_masks

    def _flat_selected_choice_offsets_tensor(self) -> Tensor:
        assert self._flat_selected_choice_offsets is not None
        return self._flat_selected_choice_offsets

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
    assert values.ndim == 2
    assert choice_width > int(values.shape[1])
    result = torch.zeros(
        (
            int(values.shape[0]),
            choice_width,
        ),
        dtype=values.dtype,
        device=values.device,
    )
    result[:, : int(values.shape[1])].copy_(values)
    return result


def _ensure_tensor(
    value: Tensor | None,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if (
        value is not None
        and value.shape == shape
        and value.dtype == dtype
        and value.device == device
    ):
        return value
    return torch.empty(shape, dtype=dtype, device=device)


def _ensure_min_capacity_tensor(
    value: Tensor | None,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if (
        value is not None
        and value.dtype == dtype
        and value.device == device
        and len(value.shape) == len(shape)
        and all(
            int(value.shape[index]) >= dimension
            for index, dimension in enumerate(shape)
        )
    ):
        return value
    return torch.empty(shape, dtype=dtype, device=device)


def _bool_tensor_value(value: Tensor) -> bool:
    assert value.shape == ()
    return bool(value.detach().cpu().item())


def _int_tensor_tuple(values: Tensor) -> tuple[int, ...]:
    assert values.ndim == 1
    cpu_values = cast(_IntTensorListable, values.detach().cpu())
    return tuple(cpu_values.tolist())


def _row_step_offsets(
    *, step_start: int, step_counts: Tensor
) -> Tensor:
    assert step_start >= 0
    assert step_counts.ndim == 1
    return torch.cumsum(step_counts, dim=0) - step_counts + step_start


def _response_trace_cpu_tensor(
    *,
    selected_token_ids: Tensor,
    step_counts: tuple[int, ...],
) -> Tensor:
    max_step_count = max(step_counts)
    assert max_step_count > 0
    return selected_token_ids[:, :max_step_count].detach().cpu()
