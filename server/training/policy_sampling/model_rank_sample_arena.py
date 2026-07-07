"""Model-rank-owned device sample arena for PPO-ready decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from server import result as _result
from server.result import Rejected
from server.training.policy_sampling.records import (
    DecisionHandle,
    DeviceDecisionReplayRecord,
)
from server.training.ppo.replay_tensors import (
    PPOReplayTensorBatch,
    ReadyPPOBatch,
)
from server.training.returns import ReturnCommit
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tensorize import ObservationTensorBatch

_INITIAL_CAPACITY = 256


def _int_list() -> list[int]:
    return []


def _bool_list() -> list[bool]:
    return []


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
    _legal_token_masks: Tensor | None = None
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

    def store_batch(
        self, *, records: tuple[DeviceDecisionReplayRecord, ...]
    ) -> tuple[DecisionHandle, ...]:
        """Store a sampled decision batch and return public handles."""
        assert records
        return tuple(self.store(record=record) for record in records)

    def store(
        self, *, record: DeviceDecisionReplayRecord
    ) -> DecisionHandle:
        """Store one sampled decision and return its public handle."""
        assert record.old_value.device == self.device
        self._ensure_capacity_for(record)
        self._validate_record_shape(record)
        slot = self._claim_slot()
        generation = self._slot_generations[slot] + 1
        self._slot_generations[slot] = generation
        self._slot_active[slot] = True
        self._write_record(slot=slot, record=record)
        return DecisionHandle(
            model_rank_index=self.model_rank_index,
            policy_version=record.policy_version,
            slot_index=slot,
            slot_generation=generation,
        )

    def materialize_return_commit(
        self, *, commit: ReturnCommit
    ) -> _result.Ok[ReadyPPOBatch] | _result.Rejected:
        """Resolve committed handles into flat device PPO samples."""
        if commit.is_empty():
            return Rejected(reason="return commit has no decisions")
        slots_result = self._slot_indices_for_commit(commit)
        if isinstance(slots_result, Rejected):
            return slots_result
        slot_tensor = torch.tensor(
            slots_result.value, dtype=torch.long, device=self.device
        )
        step_counts = self._step_counts_tensor().index_select(
            dim=0, index=slot_tensor
        )
        max_step_count = _positive_tensor_max(step_counts)
        old_values = self._old_values_tensor().index_select(
            dim=0, index=slot_tensor
        )
        return_values = _float_tensor(
            commit.return_values, device=self.device
        )
        return _result.Ok(
            value=ReadyPPOBatch(
                policy_version=commit.policy_version,
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
                    sample_count=len(slots_result.value),
                    step_count=_positive_tensor_sum(step_counts),
                    max_step_count=max_step_count,
                    selected_token_ids_padded=(
                        self._selected_token_ids_tensor().index_select(
                            dim=0, index=slot_tensor
                        )[:, :max_step_count]
                    ),
                    legal_token_masks_padded=(
                        self._legal_token_masks_tensor().index_select(
                            dim=0, index=slot_tensor
                        )[:, :max_step_count, :]
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

    def discard_commit(self, *, commit: ReturnCommit) -> None:
        """Release committed slots after a successful PPO update."""
        for handle in commit.decision_handles:
            if handle.model_rank_index != self.model_rank_index:
                continue
            if not self._slot_is_current(handle):
                continue
            self._slot_active[handle.slot_index] = False
            self._free_slots.append(handle.slot_index)

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

    def _ensure_capacity_for(
        self, record: DeviceDecisionReplayRecord
    ) -> None:
        if self._capacity == 0:
            self._initialize_tensors(record=record)
        self._ensure_observation_token_capacity(record)
        if not self._free_slots:
            self._grow()

    def _initialize_tensors(
        self, *, record: DeviceDecisionReplayRecord
    ) -> None:
        self._capacity = _INITIAL_CAPACITY
        observation = record.observation_batch
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
        self._legal_token_masks = torch.zeros(
            (
                self._capacity,
                SEMANTIC_CODEC.max_argument_tokens,
                SEMANTIC_CODEC.argument_vocab_size,
            ),
            dtype=torch.bool,
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
        self._legal_token_masks = _grow_rows(
            self._legal_token_masks_tensor(), new_capacity=new_capacity
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
        self, record: DeviceDecisionReplayRecord
    ) -> None:
        token_count = int(
            record.observation_batch.component_ids.shape[1]
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

    def _validate_record_shape(
        self, record: DeviceDecisionReplayRecord
    ) -> None:
        assert record.selected_token_ids.shape[0] <= (
            SEMANTIC_CODEC.max_argument_tokens
        )
        assert record.legal_token_masks.shape[1] == (
            SEMANTIC_CODEC.argument_vocab_size
        )
        assert (
            record.observation_batch.component_ids.shape[2:]
            == (self._component_ids_tensor().shape[2:])
        )
        assert (
            record.observation_batch.numeric_values.shape[2:]
            == (self._numeric_values_tensor().shape[2:])
        )
        assert (
            record.observation_batch.numeric_masks.shape[2:]
            == (self._numeric_masks_tensor().shape[2:])
        )

    def _claim_slot(self) -> int:
        assert self._free_slots
        return self._free_slots.pop()

    def _write_record(
        self, *, slot: int, record: DeviceDecisionReplayRecord
    ) -> None:
        step_count = int(record.selected_token_ids.shape[0])
        token_count = int(
            record.observation_batch.component_ids.shape[1]
        )
        self._component_ids_tensor()[slot].zero_()
        self._component_ids_tensor()[slot, :token_count, :].copy_(
            record.observation_batch.component_ids[0]
        )
        self._numeric_values_tensor()[slot].zero_()
        self._numeric_values_tensor()[slot, :token_count, :].copy_(
            record.observation_batch.numeric_values[0]
        )
        self._numeric_masks_tensor()[slot].zero_()
        self._numeric_masks_tensor()[slot, :token_count, :].copy_(
            record.observation_batch.numeric_masks[0]
        )
        self._selected_token_ids_tensor()[slot].zero_()
        self._selected_token_ids_tensor()[slot, :step_count].copy_(
            record.selected_token_ids
        )
        self._legal_token_masks_tensor()[slot].zero_()
        self._legal_token_masks_tensor()[slot, :step_count, :].copy_(
            record.legal_token_masks
        )
        self._step_counts_tensor()[slot] = step_count
        self._old_log_probabilities_tensor()[slot] = (
            record.old_log_probability
        )
        self._old_values_tensor()[slot] = record.old_value
        self._policy_versions_tensor()[slot] = record.policy_version

    def _slot_indices_for_commit(
        self, commit: ReturnCommit
    ) -> _result.Ok[tuple[int, ...]] | Rejected:
        slots: list[int] = []
        for handle in commit.decision_handles:
            if handle.model_rank_index != self.model_rank_index:
                return Rejected(
                    reason="return commit targets the wrong model rank"
                )
            if handle.policy_version != commit.policy_version:
                return Rejected(
                    reason="return commit policy version mismatch"
                )
            if not self._slot_is_current(handle):
                return Rejected(
                    reason="return commit references missing replay"
                )
            policy_version = int(
                self._policy_versions_tensor()[handle.slot_index]
                .detach()
                .cpu()
                .item()
            )
            if policy_version != commit.policy_version:
                return Rejected(
                    reason="replay record policy version mismatch"
                )
            slots.append(handle.slot_index)
        return _result.Ok(value=tuple(slots))

    def _slot_is_current(self, handle: DecisionHandle) -> bool:
        if handle.slot_index >= self._capacity:
            return False
        if not self._slot_active[handle.slot_index]:
            return False
        return (
            self._slot_generations[handle.slot_index]
            == handle.slot_generation
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

    def _legal_token_masks_tensor(self) -> Tensor:
        assert self._legal_token_masks is not None
        return self._legal_token_masks

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


def _float_tensor(
    values: tuple[float, ...], *, device: torch.device
) -> Tensor:
    assert values
    return torch.tensor(values, dtype=torch.float32, device=device)
