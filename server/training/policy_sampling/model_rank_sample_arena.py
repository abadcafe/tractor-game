"""Model-rank-owned device sample arena for PPO-ready decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from server import result as _result
from server.result import Ok, Rejected
from server.training.policy_sampling.records import (
    DecisionHandle,
    ModelRankPolicyDecision,
    RankReturnBatch,
    SampledPolicyBatch,
)
from server.training.ppo.replay_tensors import (
    PPOReplayTensorBatch,
    ReadyPPOBatch,
)
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tensorize import ObservationTensorBatch

_INITIAL_CAPACITY = 256

type ModelRankDecisionResult = (
    _result.Ok[ModelRankPolicyDecision] | _result.Rejected
)


def _int_list() -> list[int]:
    return []


def _bool_list() -> list[bool]:
    return []


@dataclass(frozen=True, slots=True)
class _SampledBatchCpuView:
    policy_versions: tuple[int, ...]
    status_codes: tuple[int, ...]
    step_counts: tuple[int, ...]
    choice_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ResponseTraceCpuView:
    valid_positions: tuple[int, ...]
    selected_token_ids: Tensor

    def __post_init__(self) -> None:
        assert self.valid_positions
        assert self.selected_token_ids.ndim == 2
        assert int(self.selected_token_ids.shape[0]) == len(
            self.valid_positions
        )


@dataclass(slots=True)
class ModelRankSampleArena:
    """Own sampled replay tensors and materialized return targets."""

    model_rank_index: int
    device: torch.device
    _capacity: int = 0
    _free_slots: list[int] = field(default_factory=_int_list)
    _slot_generations: list[int] = field(default_factory=_int_list)
    _slot_active: list[bool] = field(default_factory=_bool_list)
    _component_ids: Tensor | None = None
    _numeric_values: Tensor | None = None
    _numeric_masks: Tensor | None = None
    _selected_token_ids: Tensor | None = None
    _legal_choice_ids: Tensor | None = None
    _legal_choice_masks: Tensor | None = None
    _selected_choice_offsets: Tensor | None = None
    _step_counts: Tensor | None = None
    _old_log_probabilities: Tensor | None = None
    _old_values: Tensor | None = None
    _policy_versions: Tensor | None = None

    def __post_init__(self) -> None:
        assert self.model_rank_index >= 0

    def clear(self) -> None:
        """Mark all slots reusable after state sync or shutdown."""
        self._free_slots = list(reversed(range(self._capacity)))
        self._slot_active = [False for _ in range(self._capacity)]

    def store_sampled_batch(
        self, *, batch: SampledPolicyBatch
    ) -> tuple[ModelRankDecisionResult, ...]:
        """Store sampled rows and return response-ready decisions."""
        assert batch.old_values.device == self.device
        cpu = _sampled_batch_cpu_view(batch)
        valid_positions = _valid_sample_positions(cpu)
        if not valid_positions:
            return _rejected_decisions(cpu)
        self._ensure_capacity_for_batch(
            batch=batch, sample_count=len(valid_positions)
        )
        self._validate_batch_shape(batch)
        slots = self._claim_slots(len(valid_positions))
        generations = self._activate_slots(slots)
        valid_tensor = torch.tensor(
            valid_positions, dtype=torch.long, device=self.device
        )
        slot_tensor = torch.tensor(
            slots, dtype=torch.long, device=self.device
        )
        self._write_sampled_batch(
            slots=slot_tensor,
            selected_rows=valid_tensor,
            policy_versions=tuple(
                batch.policy_versions[index]
                for index in valid_positions
            ),
            batch=batch,
        )
        return _stored_decisions(
            model_rank_index=self.model_rank_index,
            cpu=cpu,
            traces=_response_trace_cpu_view(
                batch=batch,
                valid_positions=valid_positions,
                step_counts=cpu.step_counts,
            ),
            valid_positions=valid_positions,
            slots=slots,
            generations=generations,
        )

    def materialize_return_batch(
        self, *, returns: RankReturnBatch
    ) -> _result.Ok[ReadyPPOBatch] | _result.Rejected:
        """Resolve committed handles into flat device PPO samples."""
        if returns.is_empty():
            return Rejected(reason="return commit has no decisions")
        if returns.model_rank_index != self.model_rank_index:
            return Rejected(
                reason="return batch targets the wrong model rank"
            )
        slot_tensor = returns.slot_indices.to(
            dtype=torch.long, device=self.device
        )
        generation_tensor = returns.slot_generations.to(
            dtype=torch.long, device=self.device
        )
        validation_result = self._validate_return_slots(
            policy_version=returns.policy_version,
            slot_indices=slot_tensor,
            slot_generations=generation_tensor,
        )
        if isinstance(validation_result, Rejected):
            return validation_result
        step_counts = self._step_counts_tensor().index_select(
            dim=0, index=slot_tensor
        )
        max_step_count = _positive_tensor_max(step_counts)
        old_values = self._old_values_tensor().index_select(
            dim=0, index=slot_tensor
        )
        legal_choice_masks = (
            self._legal_choice_masks_tensor().index_select(
                dim=0, index=slot_tensor
            )[:, :max_step_count, :]
        )
        max_choice_count = _positive_choice_width(legal_choice_masks)
        return_values = returns.return_values.to(
            dtype=torch.float32, device=self.device
        )
        return _result.Ok(
            value=ReadyPPOBatch(
                policy_version=returns.policy_version,
                observation_batch=ObservationTensorBatch(
                    component_ids=self._component_ids_tensor().index_select(
                        dim=0, index=slot_tensor
                    ),
                    numeric_values=self._numeric_values_tensor().index_select(
                        dim=0, index=slot_tensor
                    ),
                    numeric_masks=self._numeric_masks_tensor().index_select(
                        dim=0, index=slot_tensor
                    ),
                ),
                replay=PPOReplayTensorBatch(
                    sample_count=int(slot_tensor.shape[0]),
                    step_count=_positive_tensor_sum(step_counts),
                    max_step_count=max_step_count,
                    selected_token_ids_padded=(
                        self._selected_token_ids_tensor().index_select(
                            dim=0, index=slot_tensor
                        )[:, :max_step_count]
                    ),
                    legal_choice_ids_padded=(
                        self._legal_choice_ids_tensor().index_select(
                            dim=0, index=slot_tensor
                        )[:, :max_step_count, :max_choice_count]
                    ),
                    legal_choice_masks_padded=(
                        legal_choice_masks[:, :, :max_choice_count]
                    ),
                    selected_choice_offsets_padded=(
                        self._selected_choice_offsets_tensor().index_select(
                            dim=0, index=slot_tensor
                        )[:, :max_step_count]
                    ),
                    step_mask=_step_mask(
                        step_counts=step_counts,
                        max_step_count=max_step_count,
                    ),
                    step_counts=step_counts,
                ),
                old_log_probabilities=(
                    self._old_log_probabilities_tensor().index_select(
                        dim=0, index=slot_tensor
                    )
                ),
                old_values=old_values,
                return_values=return_values,
                raw_advantages=return_values - old_values,
            )
        )

    def discard_return_batch(self, *, returns: RankReturnBatch) -> None:
        """Release committed slots after a successful PPO update."""
        if returns.model_rank_index != self.model_rank_index:
            return
        cpu_slots = returns.slot_indices.detach().cpu()
        cpu_generations = returns.slot_generations.detach().cpu()
        for index in range(int(cpu_slots.shape[0])):
            slot = int(cpu_slots[index].item())
            generation = int(cpu_generations[index].item())
            if not self._slot_is_current(
                slot_index=slot, slot_generation=generation
            ):
                continue
            self._slot_active[slot] = False
            self._free_slots.append(slot)

    def discard_uncommitted_policy_version(
        self, *, policy_version: int
    ) -> None:
        """Release active slots from a completed rollout version."""
        assert policy_version >= 0
        if self._capacity == 0:
            return
        versions = self._policy_versions_tensor()
        for slot in range(self._capacity):
            if not self._slot_active[slot]:
                continue
            slot_version = int(versions[slot].detach().cpu().item())
            if slot_version != policy_version:
                continue
            self._slot_active[slot] = False
            self._free_slots.append(slot)

    def _ensure_capacity_for_batch(
        self, *, batch: SampledPolicyBatch, sample_count: int
    ) -> None:
        assert sample_count > 0
        if self._capacity == 0:
            self._initialize_tensors(batch=batch)
        self._ensure_observation_token_capacity(batch)
        self._ensure_choice_capacity(batch)
        while len(self._free_slots) < sample_count:
            self._grow()

    def _initialize_tensors(self, *, batch: SampledPolicyBatch) -> None:
        self._capacity = _INITIAL_CAPACITY
        observation = batch.observation_batch
        token_count = int(observation.component_ids.shape[1])
        component_count = int(observation.component_ids.shape[2])
        numeric_count = int(observation.numeric_values.shape[2])
        self._component_ids = torch.empty(
            (self._capacity, token_count, component_count),
            dtype=observation.component_ids.dtype,
            device=self.device,
        )
        self._numeric_values = torch.empty(
            (self._capacity, token_count, numeric_count),
            dtype=observation.numeric_values.dtype,
            device=self.device,
        )
        self._numeric_masks = torch.empty(
            (self._capacity, token_count, numeric_count),
            dtype=observation.numeric_masks.dtype,
            device=self.device,
        )
        self._selected_token_ids = torch.zeros(
            (self._capacity, SEMANTIC_CODEC.max_argument_tokens),
            dtype=torch.long,
            device=self.device,
        )
        self._legal_choice_ids = torch.zeros(
            (
                self._capacity,
                SEMANTIC_CODEC.max_argument_tokens,
                int(batch.legal_choice_ids_padded.shape[2]),
            ),
            dtype=torch.int16,
            device=self.device,
        )
        self._legal_choice_masks = torch.zeros(
            self._legal_choice_ids.shape,
            dtype=torch.bool,
            device=self.device,
        )
        self._selected_choice_offsets = torch.zeros(
            (self._capacity, SEMANTIC_CODEC.max_argument_tokens),
            dtype=torch.long,
            device=self.device,
        )
        self._step_counts = torch.zeros(
            (self._capacity,), dtype=torch.long, device=self.device
        )
        self._old_log_probabilities = torch.zeros(
            (self._capacity,), dtype=torch.float32, device=self.device
        )
        self._old_values = torch.zeros(
            (self._capacity,), dtype=torch.float32, device=self.device
        )
        self._policy_versions = torch.zeros(
            (self._capacity,), dtype=torch.long, device=self.device
        )
        self._slot_generations = [0 for _ in range(self._capacity)]
        self._slot_active = [False for _ in range(self._capacity)]
        self._free_slots = list(reversed(range(self._capacity)))

    def _grow(self) -> None:
        old_capacity = self._capacity
        new_capacity = old_capacity * 2
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
            self._selected_token_ids_tensor(), new_capacity=new_capacity
        )
        self._legal_choice_ids = _grow_rows(
            self._legal_choice_ids_tensor(), new_capacity=new_capacity
        )
        self._legal_choice_masks = _grow_rows(
            self._legal_choice_masks_tensor(), new_capacity=new_capacity
        )
        self._selected_choice_offsets = _grow_rows(
            self._selected_choice_offsets_tensor(),
            new_capacity=new_capacity,
        )
        self._step_counts = _grow_rows(
            self._step_counts_tensor(), new_capacity=new_capacity
        )
        self._old_log_probabilities = _grow_rows(
            self._old_log_probabilities_tensor(),
            new_capacity=new_capacity,
        )
        self._old_values = _grow_rows(
            self._old_values_tensor(), new_capacity=new_capacity
        )
        self._policy_versions = _grow_rows(
            self._policy_versions_tensor(), new_capacity=new_capacity
        )
        self._capacity = new_capacity
        self._slot_generations.extend(
            0 for _ in range(new_capacity - old_capacity)
        )
        self._slot_active.extend(
            False for _ in range(new_capacity - old_capacity)
        )
        self._free_slots.extend(
            reversed(range(old_capacity, new_capacity))
        )

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
        choice_count = int(batch.legal_choice_ids_padded.shape[2])
        current_choice_count = int(
            self._legal_choice_ids_tensor().shape[2]
        )
        if choice_count <= current_choice_count:
            return
        self._legal_choice_ids = _grow_choices(
            self._legal_choice_ids_tensor(), choice_count=choice_count
        )
        self._legal_choice_masks = _grow_choices(
            self._legal_choice_masks_tensor(), choice_count=choice_count
        )

    def _validate_batch_shape(self, batch: SampledPolicyBatch) -> None:
        assert batch.selected_token_ids_padded.shape[1] == (
            SEMANTIC_CODEC.max_argument_tokens
        )
        assert batch.legal_choice_ids_padded.shape[1] == (
            SEMANTIC_CODEC.max_argument_tokens
        )
        assert batch.legal_choice_masks_padded.shape == (
            batch.legal_choice_ids_padded.shape
        )
        assert batch.selected_choice_offsets_padded.shape[1] == (
            SEMANTIC_CODEC.max_argument_tokens
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

    def _claim_slots(self, count: int) -> tuple[int, ...]:
        assert count > 0
        assert len(self._free_slots) >= count
        return tuple(self._free_slots.pop() for _ in range(count))

    def _activate_slots(
        self, slots: tuple[int, ...]
    ) -> tuple[int, ...]:
        generations: list[int] = []
        for slot in slots:
            generation = self._slot_generations[slot] + 1
            self._slot_generations[slot] = generation
            self._slot_active[slot] = True
            generations.append(generation)
        return tuple(generations)

    def _write_sampled_batch(
        self,
        *,
        slots: Tensor,
        selected_rows: Tensor,
        policy_versions: tuple[int, ...],
        batch: SampledPolicyBatch,
    ) -> None:
        assert policy_versions
        selected_observation = batch.observation_batch
        self._component_ids_tensor().index_copy_(
            0,
            slots,
            selected_observation.component_ids.index_select(
                dim=0, index=selected_rows
            ),
        )
        self._numeric_values_tensor().index_copy_(
            0,
            slots,
            selected_observation.numeric_values.index_select(
                dim=0, index=selected_rows
            ),
        )
        self._numeric_masks_tensor().index_copy_(
            0,
            slots,
            selected_observation.numeric_masks.index_select(
                dim=0, index=selected_rows
            ),
        )
        self._selected_token_ids_tensor().index_copy_(
            0,
            slots,
            batch.selected_token_ids_padded.index_select(
                dim=0, index=selected_rows
            ),
        )
        choice_width = int(batch.legal_choice_ids_padded.shape[2])
        self._legal_choice_ids_tensor().index_fill_(0, slots, 0)
        self._legal_choice_masks_tensor().index_fill_(0, slots, False)
        self._legal_choice_ids_tensor()[
            :, :, :choice_width
        ].index_copy_(
            0,
            slots,
            batch.legal_choice_ids_padded.index_select(
                dim=0, index=selected_rows
            ),
        )
        self._legal_choice_masks_tensor()[
            :, :, :choice_width
        ].index_copy_(
            0,
            slots,
            batch.legal_choice_masks_padded.index_select(
                dim=0, index=selected_rows
            ),
        )
        self._selected_choice_offsets_tensor().index_copy_(
            0,
            slots,
            batch.selected_choice_offsets_padded.index_select(
                dim=0, index=selected_rows
            ),
        )
        self._step_counts_tensor().index_copy_(
            0, slots, batch.step_counts.index_select(0, selected_rows)
        )
        self._old_log_probabilities_tensor().index_copy_(
            0,
            slots,
            batch.old_log_probabilities.index_select(0, selected_rows),
        )
        self._old_values_tensor().index_copy_(
            0, slots, batch.old_values.index_select(0, selected_rows)
        )
        policy_version_tensor = torch.tensor(
            policy_versions,
            dtype=torch.long,
            device=self.device,
        )
        self._policy_versions_tensor().index_copy_(
            0, slots, policy_version_tensor
        )

    def _slot_is_current(
        self, *, slot_index: int, slot_generation: int
    ) -> bool:
        if slot_index >= self._capacity:
            return False
        if not self._slot_active[slot_index]:
            return False
        return self._slot_generations[slot_index] == slot_generation

    def _validate_return_slots(
        self,
        *,
        policy_version: int,
        slot_indices: Tensor,
        slot_generations: Tensor,
    ) -> _result.Ok[None] | _result.Rejected:
        if int(slot_indices.shape[0]) == 0:
            return Rejected(reason="return commit has no decisions")
        cpu_slots = slot_indices.detach().cpu()
        cpu_generations = slot_generations.detach().cpu()
        for index in range(int(cpu_slots.shape[0])):
            slot = int(cpu_slots[index].item())
            generation = int(cpu_generations[index].item())
            if not self._slot_is_current(
                slot_index=slot, slot_generation=generation
            ):
                return Rejected(
                    reason="return commit references missing replay"
                )
        versions = self._policy_versions_tensor().index_select(
            dim=0, index=slot_indices
        )
        expected = torch.full_like(versions, policy_version)
        if bool((versions != expected).any().detach().cpu().item()):
            return Rejected(
                reason="replay record policy version mismatch"
            )
        return Ok(value=None)

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

    def _legal_choice_ids_tensor(self) -> Tensor:
        assert self._legal_choice_ids is not None
        return self._legal_choice_ids

    def _legal_choice_masks_tensor(self) -> Tensor:
        assert self._legal_choice_masks is not None
        return self._legal_choice_masks

    def _selected_choice_offsets_tensor(self) -> Tensor:
        assert self._selected_choice_offsets is not None
        return self._selected_choice_offsets

    def _step_counts_tensor(self) -> Tensor:
        assert self._step_counts is not None
        return self._step_counts

    def _old_log_probabilities_tensor(self) -> Tensor:
        assert self._old_log_probabilities is not None
        return self._old_log_probabilities

    def _old_values_tensor(self) -> Tensor:
        assert self._old_values is not None
        return self._old_values

    def _policy_versions_tensor(self) -> Tensor:
        assert self._policy_versions is not None
        return self._policy_versions


def _grow_rows(values: Tensor, *, new_capacity: int) -> Tensor:
    assert new_capacity > int(values.shape[0])
    result = torch.zeros(
        (new_capacity, *tuple(int(size) for size in values.shape[1:])),
        dtype=values.dtype,
        device=values.device,
    )
    result[: int(values.shape[0])].copy_(values)
    return result


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


def _grow_choices(values: Tensor, *, choice_count: int) -> Tensor:
    assert choice_count > int(values.shape[2])
    result = torch.zeros(
        (
            int(values.shape[0]),
            int(values.shape[1]),
            choice_count,
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


def _positive_tensor_max(values: Tensor) -> int:
    result = int(values.detach().cpu().max().item())
    assert result > 0
    return result


def _positive_tensor_sum(values: Tensor) -> int:
    result = int(values.detach().cpu().sum().item())
    assert result > 0
    return result


def _positive_choice_width(choice_masks: Tensor) -> int:
    choice_counts = choice_masks.to(dtype=torch.long).sum(dim=2)
    result = int(choice_counts.detach().cpu().max().item())
    assert result > 0
    return result


def _sampled_batch_cpu_view(
    batch: SampledPolicyBatch,
) -> _SampledBatchCpuView:
    return _SampledBatchCpuView(
        policy_versions=batch.policy_versions,
        status_codes=_int_tensor_tuple(batch.status_codes),
        step_counts=_int_tensor_tuple(batch.step_counts),
        choice_counts=_int_tensor_tuple(batch.choice_counts),
    )


def _int_tensor_tuple(values: Tensor) -> tuple[int, ...]:
    cpu_values = values.detach().cpu()
    return tuple(
        int(cpu_values[index].item())
        for index in range(int(cpu_values.shape[0]))
    )


def _valid_sample_positions(
    cpu: _SampledBatchCpuView,
) -> tuple[int, ...]:
    return tuple(
        index
        for index, (status, step_count, choice_count) in enumerate(
            zip(
                cpu.status_codes,
                cpu.step_counts,
                cpu.choice_counts,
                strict=True,
            )
        )
        if status == 0 and step_count > 0 and choice_count > 0
    )


def _rejected_decisions(
    cpu: _SampledBatchCpuView,
) -> tuple[ModelRankDecisionResult, ...]:
    return tuple(
        Rejected(reason=_sample_rejection_reason(cpu=cpu, index=index))
        for index in range(len(cpu.policy_versions))
    )


def _stored_decisions(
    *,
    model_rank_index: int,
    cpu: _SampledBatchCpuView,
    traces: _ResponseTraceCpuView,
    valid_positions: tuple[int, ...],
    slots: tuple[int, ...],
    generations: tuple[int, ...],
) -> tuple[ModelRankDecisionResult, ...]:
    slot_by_position = {
        position: slots[index]
        for index, position in enumerate(valid_positions)
    }
    generation_by_position = {
        position: generations[index]
        for index, position in enumerate(valid_positions)
    }
    return tuple(
        _stored_decision_for_position(
            model_rank_index=model_rank_index,
            cpu=cpu,
            traces=traces,
            position=position,
            slot_by_position=slot_by_position,
            generation_by_position=generation_by_position,
        )
        for position in range(len(cpu.policy_versions))
    )


def _stored_decision_for_position(
    *,
    model_rank_index: int,
    cpu: _SampledBatchCpuView,
    traces: _ResponseTraceCpuView,
    position: int,
    slot_by_position: dict[int, int],
    generation_by_position: dict[int, int],
) -> ModelRankDecisionResult:
    slot = slot_by_position.get(position)
    generation = generation_by_position.get(position)
    if slot is None or generation is None:
        return Rejected(
            reason=_sample_rejection_reason(cpu=cpu, index=position)
        )
    step_count = cpu.step_counts[position]
    return Ok(
        value=ModelRankPolicyDecision(
            trace_token_ids=_trace_token_ids(
                traces=traces,
                position=position,
                step_count=step_count,
            ),
            decision_handle=DecisionHandle(
                model_rank_index=model_rank_index,
                policy_version=cpu.policy_versions[position],
                slot_index=slot,
                slot_generation=generation,
            ),
            choice_count=cpu.choice_counts[position],
        )
    )


def _response_trace_cpu_view(
    *,
    batch: SampledPolicyBatch,
    valid_positions: tuple[int, ...],
    step_counts: tuple[int, ...],
) -> _ResponseTraceCpuView:
    assert valid_positions
    max_step_count = max(
        step_counts[position] for position in valid_positions
    )
    assert max_step_count > 0
    selected_rows = torch.tensor(
        valid_positions,
        dtype=torch.long,
        device=batch.selected_token_ids_padded.device,
    )
    selected_tokens = (
        (
            batch.selected_token_ids_padded.index_select(
                dim=0, index=selected_rows
            )[:, :max_step_count]
        )
        .detach()
        .cpu()
    )
    return _ResponseTraceCpuView(
        valid_positions=valid_positions,
        selected_token_ids=selected_tokens,
    )


def _trace_token_ids(
    *, traces: _ResponseTraceCpuView, position: int, step_count: int
) -> tuple[int, ...]:
    assert step_count > 0
    trace_row = traces.valid_positions.index(position)
    return tuple(
        int(traces.selected_token_ids[trace_row, token_index].item())
        for token_index in range(step_count)
    )


def _sample_rejection_reason(
    *, cpu: _SampledBatchCpuView, index: int
) -> str:
    status = cpu.status_codes[index]
    if status != 0:
        return "policy sampling failed"
    if cpu.step_counts[index] <= 0:
        return "policy action trace is empty"
    if cpu.choice_counts[index] <= 0:
        return "policy action has no legal choices"
    return "policy sampling failed"
