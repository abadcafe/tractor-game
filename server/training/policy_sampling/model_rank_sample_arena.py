"""Model-rank-owned append-only sample arena for PPO replay."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected
from server.training.observation_structure import STRUCTURE_AXIS_COUNT
from server.training.policy_sampling.records import (
    CompactActionChoiceBatch,
    CompactPolicyDecisionBatch,
    RankReturnTargets,
)
from server.training.ppo.minibatch import TensorizedPPOMinibatch
from server.training.ppo.replay_tensors import PPOReplayTensorBatch
from server.training.semantic_action_plan import ActionSampleBatch
from server.training.semantic_actions.choices import (
    ACTION_CHOICE_COUNT,
    CARD_CHOICE_COUNT,
    MAX_ACTION_STEPS,
)
from server.training.tensorize import ObservationTensorBatch
from server.training.tokenization.encoding_schema import CATEGORY_COUNT

_INITIAL_CAPACITY = 256

type ModelRankDecisionBatchResult = (
    _result.Ok[CompactPolicyDecisionBatch] | _result.Rejected
)


@dataclass(frozen=True, slots=True)
class ArenaPPOBatchSource:
    """Committed PPO rows backed by one model-rank arena."""

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

    def sample_count(self) -> int:
        return int(self.row_indices.shape[0])

    def select_minibatch(
        self,
        *,
        indices: Tensor,
        advantages: Tensor,
        global_count: Tensor,
    ) -> TensorizedPPOMinibatch:
        return self.arena.select_ppo_minibatch(
            source=self,
            indices=indices,
            advantages=advantages,
            global_count=global_count,
        )


@dataclass(slots=True)
class ModelRankSampleArena:
    """Append policy rows and expose exact fixed-vocabulary replay."""

    model_rank_index: int
    device: torch.device
    _capacity: int = 0
    _step_capacity: int = 0
    _token_capacity: int = 0
    _row_count: int = 0
    _step_count: int = 0
    _policy_version: int | None = None
    _row_policy_versions: Tensor | None = None
    _row_step_counts: Tensor | None = None
    _row_step_offsets: Tensor | None = None
    _category_ids: Tensor | None = None
    _scalar_values: Tensor | None = None
    _card_rule_values: Tensor | None = None
    _encoded_structure_coordinates: Tensor | None = None
    _candidate_category_ids: Tensor | None = None
    _candidate_counts: Tensor | None = None
    _candidate_card_rule_values: Tensor | None = None
    _query_indices: Tensor | None = None
    _choice_ids: Tensor | None = None
    _flat_legal_choice_masks: Tensor | None = None
    _old_log_probabilities: Tensor | None = None
    _old_values: Tensor | None = None

    def __post_init__(self) -> None:
        assert self.model_rank_index >= 0

    def clear(self) -> None:
        self._row_count = 0
        self._step_count = 0
        self._policy_version = None

    def store_sampled_result(
        self,
        *,
        policy_versions: tuple[int, ...],
        observation_batch: ObservationTensorBatch,
        action_sample: ActionSampleBatch,
        old_values: Tensor,
    ) -> ModelRankDecisionBatchResult:
        sample_count = len(policy_versions)
        assert sample_count > 0
        version_result = _single_policy_version(policy_versions)
        if isinstance(version_result, Rejected):
            return version_result
        policy_version = version_result.value
        if self._policy_version not in (None, policy_version):
            return Rejected(
                reason="sample arena policy version mismatch"
            )
        self._policy_version = policy_version
        self._ensure_capacity(
            observation=observation_batch,
            row_needed=self._row_count + sample_count,
            step_needed=self._step_count
            + int(action_sample.legal_choice_masks.shape[0]),
        )
        self._validate_sample(
            observation=observation_batch,
            sample=action_sample,
            old_values=old_values,
            sample_count=sample_count,
        )
        start = self._row_count
        end = start + sample_count
        step_start = self._step_count
        step_end = step_start + int(
            action_sample.legal_choice_masks.shape[0]
        )
        rows = slice(start, end)
        self._row_policy_versions_tensor()[rows] = torch.tensor(
            policy_versions, dtype=torch.long, device=self.device
        )
        self._row_step_counts_tensor()[rows].copy_(
            action_sample.step_counts
        )
        self._row_step_offsets_tensor()[rows].copy_(
            torch.cumsum(action_sample.step_counts, 0)
            - action_sample.step_counts
            + step_start
        )
        self._write_observation(rows=rows, source=observation_batch)
        self._choice_ids_tensor()[rows].zero_()
        width = int(action_sample.choice_ids_padded.shape[1])
        self._choice_ids_tensor()[rows, :width].copy_(
            action_sample.choice_ids_padded
        )
        self._flat_legal_choice_masks_tensor()[
            step_start:step_end
        ].copy_(action_sample.legal_choice_masks)
        self._old_log_probabilities_tensor()[rows].copy_(
            action_sample.log_probabilities
        )
        self._old_values_tensor()[rows].copy_(old_values)
        self._row_count = end
        self._step_count = step_end
        step_counts = _int_tuple(action_sample.step_counts)
        return Ok(
            value=CompactPolicyDecisionBatch(
                model_rank_index=self.model_rank_index,
                policy_versions=policy_versions,
                row_indices=tuple(range(start, end)),
                choice_counts=_int_tuple(action_sample.choice_counts),
                action_choice_batch=CompactActionChoiceBatch.from_cpu_tensor(
                    choice_ids=action_sample.choice_ids_padded.detach().cpu(),
                    choice_counts=step_counts,
                ),
            )
        )

    def ppo_batch_source(
        self, *, returns: RankReturnTargets
    ) -> _result.Ok[ArenaPPOBatchSource] | _result.Rejected:
        if returns.is_empty():
            return Rejected(reason="return commit has no decisions")
        if returns.model_rank_index != self.model_rank_index:
            return Rejected(
                reason="return batch targets the wrong model rank"
            )
        rows = returns.row_indices.to(
            dtype=torch.long, device=self.device
        )
        steps = returns.step_counts.to(
            dtype=torch.long, device=self.device
        )
        valid = self._validate_return_rows(
            policy_version=returns.policy_version,
            rows=rows,
            step_counts=steps,
        )
        if isinstance(valid, Rejected):
            return valid
        old_values = self._old_values_tensor().index_select(0, rows)
        return_values = returns.return_values.to(
            dtype=torch.float32, device=self.device
        )
        old_log_probabilities = (
            self._old_log_probabilities_tensor().index_select(0, rows)
        )
        return Ok(
            value=ArenaPPOBatchSource(
                arena=self,
                policy_version=returns.policy_version,
                model_rank_index=self.model_rank_index,
                row_indices=rows,
                step_counts=steps,
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
        if returns.model_rank_index != self.model_rank_index:
            return

    def discard_uncommitted_policy_version(
        self, *, policy_version: int
    ) -> None:
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
        assert indices.ndim == 1
        local_count = int(indices.shape[0])
        assert local_count > 0
        selected_rows = source.row_indices.index_select(0, indices)
        step_counts = source.step_counts.index_select(0, indices)
        max_steps = source.max_step_count
        positions = (
            torch.arange(
                max_steps, dtype=torch.long, device=self.device
            )
            .unsqueeze(0)
            .expand(local_count, -1)
        )
        active_mask = positions < step_counts.unsqueeze(1)
        active_sample_indices = (
            torch.arange(
                local_count, dtype=torch.long, device=self.device
            )
            .unsqueeze(1)
            .expand(-1, max_steps)[active_mask]
        )
        active_step_indices = positions[active_mask]
        source_step_indices = (
            self._row_step_offsets_tensor()
            .index_select(0, selected_rows)
            .unsqueeze(1)
            .expand(-1, max_steps)[active_mask]
            + active_step_indices
        )
        observation = self._select_observation(selected_rows)
        replay = PPOReplayTensorBatch(
            sample_count=local_count,
            max_step_count=max_steps,
            active_step_count=int(active_step_indices.shape[0]),
            choice_ids_padded=self._choice_ids_tensor().index_select(
                0, selected_rows
            )[:, :max_steps],
            active_sample_indices=active_sample_indices,
            active_step_indices=active_step_indices,
            legal_choice_masks=self._flat_legal_choice_masks_tensor().index_select(
                0, source_step_indices
            ),
            step_counts=step_counts,
        )
        return TensorizedPPOMinibatch(
            observation_batch=observation,
            replay=replay,
            sample_indices=indices,
            old_log_probabilities=source.old_log_probabilities.index_select(
                0, indices
            ),
            old_values=source.old_values.index_select(0, indices),
            advantages=advantages.index_select(0, indices),
            return_values=source.return_values.index_select(0, indices),
            local_count=local_count,
            global_count=global_count,
        )

    def _ensure_capacity(
        self,
        *,
        observation: ObservationTensorBatch,
        row_needed: int,
        step_needed: int,
    ) -> None:
        token_needed = int(observation.category_ids.shape[1])
        if self._capacity == 0:
            self._initialize(
                row_capacity=max(_INITIAL_CAPACITY, row_needed),
                step_capacity=max(_INITIAL_CAPACITY, step_needed, 1),
                token_capacity=token_needed,
            )
        while self._capacity < row_needed:
            self._grow_rows(self._capacity * 2)
        while self._step_capacity < step_needed:
            self._grow_steps(self._step_capacity * 2)
        if self._token_capacity < token_needed:
            self._grow_tokens(token_needed)

    def _initialize(
        self,
        *,
        row_capacity: int,
        step_capacity: int,
        token_capacity: int,
    ) -> None:
        self._capacity = row_capacity
        self._step_capacity = step_capacity
        self._token_capacity = token_capacity
        self._row_policy_versions = _zeros(
            (row_capacity,), torch.long, self.device
        )
        self._row_step_counts = _zeros(
            (row_capacity,), torch.long, self.device
        )
        self._row_step_offsets = _zeros(
            (row_capacity,), torch.long, self.device
        )
        self._category_ids = _zeros(
            (row_capacity, token_capacity, CATEGORY_COUNT),
            torch.long,
            self.device,
        )
        self._scalar_values = _zeros(
            (row_capacity, token_capacity), torch.float32, self.device
        )
        self._card_rule_values = _zeros(
            (row_capacity, token_capacity, 2),
            torch.float32,
            self.device,
        )
        self._encoded_structure_coordinates = _zeros(
            (row_capacity, token_capacity, STRUCTURE_AXIS_COUNT),
            torch.long,
            self.device,
        )
        self._candidate_category_ids = _zeros(
            (row_capacity, CARD_CHOICE_COUNT, 3),
            torch.long,
            self.device,
        )
        self._candidate_counts = _zeros(
            (row_capacity, CARD_CHOICE_COUNT),
            torch.float32,
            self.device,
        )
        self._candidate_card_rule_values = _zeros(
            (row_capacity, CARD_CHOICE_COUNT, 2),
            torch.float32,
            self.device,
        )
        self._query_indices = _zeros(
            (row_capacity,), torch.long, self.device
        )
        self._choice_ids = _zeros(
            (row_capacity, MAX_ACTION_STEPS), torch.long, self.device
        )
        self._flat_legal_choice_masks = _zeros(
            (step_capacity, ACTION_CHOICE_COUNT),
            torch.bool,
            self.device,
        )
        self._old_log_probabilities = _zeros(
            (row_capacity,), torch.float32, self.device
        )
        self._old_values = _zeros(
            (row_capacity,), torch.float32, self.device
        )

    def _write_observation(
        self, *, rows: slice, source: ObservationTensorBatch
    ) -> None:
        token_count = int(source.category_ids.shape[1])
        sequence_pairs = (
            (self._category_ids_tensor(), source.category_ids),
            (self._scalar_values_tensor(), source.scalar_values),
            (self._card_rule_values_tensor(), source.card_rule_values),
            (
                self._encoded_structure_coordinates_tensor(),
                source.encoded_structure_coordinates,
            ),
        )
        for destination, values in sequence_pairs:
            destination[rows].zero_()
            destination[rows, :token_count].copy_(values)
        self._candidate_category_ids_tensor()[rows].copy_(
            source.candidate_category_ids
        )
        self._candidate_counts_tensor()[rows].copy_(
            source.candidate_counts
        )
        self._candidate_card_rule_values_tensor()[rows].copy_(
            source.candidate_card_rule_values
        )
        self._query_indices_tensor()[rows].copy_(source.query_indices)

    def _select_observation(
        self, rows: Tensor
    ) -> ObservationTensorBatch:
        return ObservationTensorBatch(
            category_ids=self._category_ids_tensor().index_select(
                0, rows
            ),
            scalar_values=self._scalar_values_tensor().index_select(
                0, rows
            ),
            card_rule_values=self._card_rule_values_tensor().index_select(
                0, rows
            ),
            encoded_structure_coordinates=(
                self._encoded_structure_coordinates_tensor().index_select(
                    0, rows
                )
            ),
            candidate_category_ids=self._candidate_category_ids_tensor().index_select(
                0, rows
            ),
            candidate_counts=self._candidate_counts_tensor().index_select(
                0, rows
            ),
            candidate_card_rule_values=self._candidate_card_rule_values_tensor().index_select(
                0, rows
            ),
            query_indices=self._query_indices_tensor().index_select(
                0, rows
            ),
        )

    def _validate_sample(
        self,
        *,
        observation: ObservationTensorBatch,
        sample: ActionSampleBatch,
        old_values: Tensor,
        sample_count: int,
    ) -> None:
        assert int(observation.category_ids.shape[0]) == sample_count
        assert int(sample.choice_ids_padded.shape[0]) == sample_count
        assert sample.legal_choice_masks.shape[1] == ACTION_CHOICE_COUNT
        assert old_values.shape == (sample_count,)

    def _validate_return_rows(
        self,
        *,
        policy_version: int,
        rows: Tensor,
        step_counts: Tensor,
    ) -> _result.Ok[None] | _result.Rejected:
        if int(rows.shape[0]) == 0 or self._row_count == 0:
            return Rejected(
                reason="return commit references missing replay"
            )
        in_range = (rows >= 0) & (rows < self._row_count)
        if not _tensor_bool(in_range.all()):
            return Rejected(
                reason="return commit references missing replay"
            )
        versions = self._row_policy_versions_tensor().index_select(
            0, rows
        )
        if not _tensor_bool((versions == policy_version).all()):
            return Rejected(
                reason="replay record policy version mismatch"
            )
        stored_steps = self._row_step_counts_tensor().index_select(
            0, rows
        )
        if not _tensor_bool((stored_steps == step_counts).all()):
            return Rejected(
                reason="return commit replay step count mismatch"
            )
        return Ok(value=None)

    def _grow_rows(self, capacity: int) -> None:
        self._row_policy_versions = _grow_first_dimension(
            self._row_policy_versions_tensor(), capacity
        )
        self._row_step_counts = _grow_first_dimension(
            self._row_step_counts_tensor(), capacity
        )
        self._row_step_offsets = _grow_first_dimension(
            self._row_step_offsets_tensor(), capacity
        )
        self._category_ids = _grow_first_dimension(
            self._category_ids_tensor(), capacity
        )
        self._scalar_values = _grow_first_dimension(
            self._scalar_values_tensor(), capacity
        )
        self._card_rule_values = _grow_first_dimension(
            self._card_rule_values_tensor(), capacity
        )
        self._encoded_structure_coordinates = _grow_first_dimension(
            self._encoded_structure_coordinates_tensor(), capacity
        )
        self._candidate_category_ids = _grow_first_dimension(
            self._candidate_category_ids_tensor(), capacity
        )
        self._candidate_counts = _grow_first_dimension(
            self._candidate_counts_tensor(), capacity
        )
        self._candidate_card_rule_values = _grow_first_dimension(
            self._candidate_card_rule_values_tensor(), capacity
        )
        self._query_indices = _grow_first_dimension(
            self._query_indices_tensor(), capacity
        )
        self._choice_ids = _grow_first_dimension(
            self._choice_ids_tensor(), capacity
        )
        self._old_log_probabilities = _grow_first_dimension(
            self._old_log_probabilities_tensor(), capacity
        )
        self._old_values = _grow_first_dimension(
            self._old_values_tensor(), capacity
        )
        self._capacity = capacity

    def _grow_steps(self, capacity: int) -> None:
        self._flat_legal_choice_masks = _grow_first_dimension(
            self._flat_legal_choice_masks_tensor(), capacity
        )
        self._step_capacity = capacity

    def _grow_tokens(self, capacity: int) -> None:
        self._category_ids = _grow_second_dimension(
            self._category_ids_tensor(), capacity
        )
        self._scalar_values = _grow_second_dimension(
            self._scalar_values_tensor(), capacity
        )
        self._card_rule_values = _grow_second_dimension(
            self._card_rule_values_tensor(), capacity
        )
        self._encoded_structure_coordinates = _grow_second_dimension(
            self._encoded_structure_coordinates_tensor(), capacity
        )
        self._token_capacity = capacity

    def _row_policy_versions_tensor(self) -> Tensor:
        return _present(self._row_policy_versions)

    def _row_step_counts_tensor(self) -> Tensor:
        return _present(self._row_step_counts)

    def _row_step_offsets_tensor(self) -> Tensor:
        return _present(self._row_step_offsets)

    def _category_ids_tensor(self) -> Tensor:
        return _present(self._category_ids)

    def _scalar_values_tensor(self) -> Tensor:
        return _present(self._scalar_values)

    def _card_rule_values_tensor(self) -> Tensor:
        return _present(self._card_rule_values)

    def _encoded_structure_coordinates_tensor(self) -> Tensor:
        return _present(self._encoded_structure_coordinates)

    def _candidate_category_ids_tensor(self) -> Tensor:
        return _present(self._candidate_category_ids)

    def _candidate_counts_tensor(self) -> Tensor:
        return _present(self._candidate_counts)

    def _candidate_card_rule_values_tensor(self) -> Tensor:
        return _present(self._candidate_card_rule_values)

    def _query_indices_tensor(self) -> Tensor:
        return _present(self._query_indices)

    def _choice_ids_tensor(self) -> Tensor:
        return _present(self._choice_ids)

    def _flat_legal_choice_masks_tensor(self) -> Tensor:
        return _present(self._flat_legal_choice_masks)

    def _old_log_probabilities_tensor(self) -> Tensor:
        return _present(self._old_log_probabilities)

    def _old_values_tensor(self) -> Tensor:
        return _present(self._old_values)


def _single_policy_version(
    versions: tuple[int, ...],
) -> Ok[int] | Rejected:
    assert versions
    if any(value != versions[0] for value in versions):
        return Rejected(reason="sample batch mixes policy versions")
    return Ok(value=versions[0])


def _zeros(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> Tensor:
    return torch.zeros(shape, dtype=dtype, device=device)


def _grow_first_dimension(values: Tensor, capacity: int) -> Tensor:
    result = torch.zeros(
        (capacity, *tuple(int(size) for size in values.shape[1:])),
        dtype=values.dtype,
        device=values.device,
    )
    result[: int(values.shape[0])].copy_(values)
    return result


def _grow_second_dimension(values: Tensor, capacity: int) -> Tensor:
    shape = tuple(int(size) for size in values.shape)
    result = torch.zeros(
        (shape[0], capacity, *shape[2:]),
        dtype=values.dtype,
        device=values.device,
    )
    result[:, : shape[1]].copy_(values)
    return result


def _present(value: Tensor | None) -> Tensor:
    assert value is not None
    return value


def _tensor_bool(value: Tensor) -> bool:
    assert value.shape == ()
    return bool(value.detach().cpu().item())


def _int_tuple(values: Tensor) -> tuple[int, ...]:
    assert values.ndim == 1
    return tuple(
        int(values[index].detach().cpu().item())
        for index in range(int(values.shape[0]))
    )


__all__ = ("ArenaPPOBatchSource", "ModelRankSampleArena")
