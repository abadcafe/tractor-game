"""Device-resident PPO replay tensors for semantic token traces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tensorize import (
    ArgumentPrefixTensorBatch,
    ObservationTensorBatch,
)


@dataclass(frozen=True, slots=True)
class PPOReplayTensorBatch:
    """Recorded semantic-token replay tensors for one rollout."""

    sample_count: int
    step_count: int
    max_step_count: int
    selected_token_ids_padded: Tensor
    legal_token_masks_padded: Tensor
    step_mask: Tensor
    step_counts: Tensor

    def __post_init__(self) -> None:
        assert self.sample_count > 0
        assert self.step_count > 0
        assert self.max_step_count > 0
        assert self.selected_token_ids_padded.shape == (
            self.sample_count,
            self.max_step_count,
        )
        assert self.legal_token_masks_padded.shape == (
            self.sample_count,
            self.max_step_count,
            SEMANTIC_CODEC.argument_vocab_size,
        )
        assert self.step_mask.shape == (
            self.sample_count,
            self.max_step_count,
        )
        assert self.step_counts.shape == (self.sample_count,)
        assert self.selected_token_ids_padded.dtype == torch.long
        assert self.legal_token_masks_padded.dtype == torch.bool
        assert self.step_mask.dtype == torch.bool
        assert self.step_counts.dtype == torch.long


@dataclass(frozen=True, slots=True)
class RolloutTensorBatch:
    """Learner-ready rollout tensors on one torch device."""

    policy_version: int
    first_episode_id: int
    episode_count: int
    max_trajectory_length: int
    trajectory_count: int
    observation_batch: ObservationTensorBatch
    replay: PPOReplayTensorBatch
    old_log_probabilities: Tensor
    old_values: Tensor
    reward_after_step: Tensor
    terminal_rewards: Tensor
    trajectory_offsets: Tensor
    trajectory_team_indices: Tensor

    def __post_init__(self) -> None:
        assert self.policy_version >= 0
        assert self.first_episode_id >= 0
        assert self.episode_count >= 0
        assert self.max_trajectory_length > 0
        assert self.trajectory_count > 0
        assert self.old_log_probabilities.ndim == 1
        assert self.old_values.ndim == 1
        assert self.reward_after_step.ndim == 1
        assert self.terminal_rewards.ndim == 1
        assert self.trajectory_offsets.ndim == 1
        assert self.trajectory_team_indices.ndim == 1
        sample_count = self.replay.sample_count
        assert int(self.old_log_probabilities.shape[0]) == sample_count
        assert int(self.old_values.shape[0]) == sample_count
        assert int(self.reward_after_step.shape[0]) == sample_count
        assert int(self.observation_batch.component_ids.shape[0]) == (
            sample_count
        )
        assert int(self.trajectory_offsets.shape[0]) == (
            self.trajectory_count + 1
        )
        assert int(self.trajectory_team_indices.shape[0]) == (
            self.trajectory_count
        )
        assert int(self.terminal_rewards.shape[0]) == (
            self.trajectory_count
        )

    def transition_count(self) -> int:
        """Return the number of decision transitions."""
        return self.replay.sample_count

    def is_empty(self) -> bool:
        """Return whether this batch has no trainable transitions."""
        return self.transition_count() == 0


@dataclass(frozen=True, slots=True)
class ReplayPrefixTensorBatch:
    """Flat replay tensors for one minibatch."""

    active_positions: Tensor
    prefix_lengths: Tensor
    prefix_batch: ArgumentPrefixTensorBatch
    legal_token_masks: Tensor
    selected_token_ids: Tensor

    def __post_init__(self) -> None:
        row_count = int(self.active_positions.shape[0])
        assert row_count > 0
        assert self.active_positions.ndim == 1
        assert self.prefix_lengths.ndim == 1
        assert self.prefix_batch.argument_ids.ndim == 2
        assert self.prefix_batch.argument_masks.ndim == 2
        assert self.legal_token_masks.shape == (
            row_count,
            SEMANTIC_CODEC.argument_vocab_size,
        )
        assert self.selected_token_ids.shape == (row_count,)
        assert int(self.prefix_lengths.shape[0]) == row_count
        assert int(self.prefix_batch.argument_ids.shape[0]) == row_count
        assert (
            int(self.prefix_batch.argument_masks.shape[0]) == row_count
        )
        assert self.active_positions.dtype == torch.long
        assert self.prefix_lengths.dtype == torch.long
        assert self.legal_token_masks.dtype == torch.bool
        assert self.selected_token_ids.dtype == torch.long


def merge_rollout_tensor_batches(
    batches: tuple[RolloutTensorBatch, ...],
) -> RolloutTensorBatch:
    """Merge rollout tensor batches for one policy version."""
    assert batches
    policy_version = batches[0].policy_version
    assert all(
        batch.policy_version == policy_version for batch in batches
    )
    device = batches[0].old_values.device
    assert all(batch.old_values.device == device for batch in batches)
    max_tokens = max(
        int(batch.observation_batch.component_ids.shape[1])
        for batch in batches
    )
    return RolloutTensorBatch(
        policy_version=policy_version,
        first_episode_id=min(
            batch.first_episode_id for batch in batches
        ),
        episode_count=sum(batch.episode_count for batch in batches),
        max_trajectory_length=max(
            batch.max_trajectory_length for batch in batches
        ),
        trajectory_count=sum(
            batch.trajectory_count for batch in batches
        ),
        observation_batch=_merge_observation_batches(
            batches=batches, max_tokens=max_tokens
        ),
        replay=_merge_replay_batches(
            tuple(batch.replay for batch in batches)
        ),
        old_log_probabilities=torch.cat(
            [batch.old_log_probabilities for batch in batches], dim=0
        ),
        old_values=torch.cat([batch.old_values for batch in batches]),
        reward_after_step=torch.cat(
            [batch.reward_after_step for batch in batches], dim=0
        ),
        terminal_rewards=torch.cat(
            [batch.terminal_rewards for batch in batches], dim=0
        ),
        trajectory_offsets=_merge_trajectory_offsets(batches),
        trajectory_team_indices=torch.cat(
            [batch.trajectory_team_indices for batch in batches], dim=0
        ),
    )


def replay_prefix_tensor_batch(
    *,
    replay: PPOReplayTensorBatch,
    sample_indices: Tensor,
) -> ReplayPrefixTensorBatch | None:
    """Return one flat tensor batch for all replay prefixes."""
    assert sample_indices.ndim == 1
    if int(sample_indices.shape[0]) == 0:
        return None
    sample_step_counts = replay.step_counts.index_select(
        dim=0, index=sample_indices
    )
    argument_positions = torch.arange(
        replay.max_step_count,
        dtype=torch.long,
        device=sample_indices.device,
    ).unsqueeze(0)
    active_mask = argument_positions < sample_step_counts.unsqueeze(1)
    active_coordinates = torch.nonzero(active_mask, as_tuple=False)
    if int(active_coordinates.shape[0]) == 0:
        return None
    active_positions = active_coordinates[:, 0]
    argument_indices = active_coordinates[:, 1]
    active_sample_indices = sample_indices.index_select(
        dim=0, index=active_positions
    )
    selected_rows = replay.selected_token_ids_padded.index_select(
        dim=0, index=active_sample_indices
    )
    return ReplayPrefixTensorBatch(
        active_positions=active_positions,
        prefix_lengths=argument_indices,
        prefix_batch=_prefix_batch_for_active_prefixes(
            selected_rows=selected_rows,
            argument_indices=argument_indices,
            max_step_count=replay.max_step_count,
        ),
        legal_token_masks=replay.legal_token_masks_padded[
            active_sample_indices, argument_indices
        ],
        selected_token_ids=selected_rows.gather(
            dim=1, index=argument_indices.unsqueeze(1)
        ).squeeze(1),
    )


def _prefix_batch_for_active_prefixes(
    *,
    selected_rows: Tensor,
    argument_indices: Tensor,
    max_step_count: int,
) -> ArgumentPrefixTensorBatch:
    row_count = int(selected_rows.shape[0])
    assert row_count > 0
    prefix_ids = selected_rows[:, : max_step_count - 1]
    prefix_positions = torch.arange(
        max_step_count - 1,
        dtype=torch.long,
        device=selected_rows.device,
    ).unsqueeze(0)
    prefix_masks = prefix_positions < argument_indices.unsqueeze(1)
    safe_prefix_ids = torch.where(
        prefix_masks,
        prefix_ids,
        torch.zeros_like(prefix_ids),
    )
    bos = torch.full(
        (row_count, 1),
        SEMANTIC_CODEC.argument_bos_id,
        dtype=torch.long,
        device=selected_rows.device,
    )
    argument_ids = torch.cat((bos, safe_prefix_ids), dim=1)
    argument_masks = torch.cat(
        (
            torch.ones(
                (row_count, 1),
                dtype=torch.bool,
                device=selected_rows.device,
            ),
            prefix_masks,
        ),
        dim=1,
    )
    return ArgumentPrefixTensorBatch(
        argument_ids=argument_ids,
        argument_masks=argument_masks,
    )


def _merge_replay_batches(
    batches: tuple[PPOReplayTensorBatch, ...],
) -> PPOReplayTensorBatch:
    assert batches
    max_steps = max(batch.max_step_count for batch in batches)
    return PPOReplayTensorBatch(
        sample_count=sum(batch.sample_count for batch in batches),
        step_count=sum(batch.step_count for batch in batches),
        max_step_count=max_steps,
        selected_token_ids_padded=torch.cat(
            [
                _pad_step_columns(
                    batch.selected_token_ids_padded,
                    max_steps=max_steps,
                )
                for batch in batches
            ],
            dim=0,
        ),
        legal_token_masks_padded=torch.cat(
            [
                _pad_step_mask_columns(
                    batch.legal_token_masks_padded,
                    max_steps=max_steps,
                )
                for batch in batches
            ],
            dim=0,
        ),
        step_mask=torch.cat(
            [
                _pad_step_columns(batch.step_mask, max_steps=max_steps)
                for batch in batches
            ],
            dim=0,
        ),
        step_counts=torch.cat([batch.step_counts for batch in batches]),
    )


def _merge_observation_batches(
    *,
    batches: tuple[RolloutTensorBatch, ...],
    max_tokens: int,
) -> ObservationTensorBatch:
    return ObservationTensorBatch(
        component_ids=torch.cat(
            [
                _pad_observation_tokens(
                    batch.observation_batch.component_ids,
                    max_tokens=max_tokens,
                )
                for batch in batches
            ],
            dim=0,
        ),
        numeric_values=torch.cat(
            [
                _pad_observation_tokens(
                    batch.observation_batch.numeric_values,
                    max_tokens=max_tokens,
                )
                for batch in batches
            ],
            dim=0,
        ),
        numeric_masks=torch.cat(
            [
                _pad_observation_tokens(
                    batch.observation_batch.numeric_masks,
                    max_tokens=max_tokens,
                )
                for batch in batches
            ],
            dim=0,
        ),
    )


def _pad_observation_tokens(
    values: Tensor, *, max_tokens: int
) -> Tensor:
    current_tokens = int(values.shape[1])
    if current_tokens == max_tokens:
        return values
    padding = torch.zeros(
        (
            int(values.shape[0]),
            max_tokens - current_tokens,
            int(values.shape[2]),
        ),
        dtype=values.dtype,
        device=values.device,
    )
    return torch.cat((values, padding), dim=1)


def _merge_trajectory_offsets(
    batches: tuple[RolloutTensorBatch, ...],
) -> Tensor:
    pieces: list[Tensor] = []
    transition_base = 0
    for batch in batches:
        pieces.append(batch.trajectory_offsets[:-1] + transition_base)
        transition_base += batch.transition_count()
    last = torch.tensor(
        (transition_base,),
        dtype=torch.long,
        device=batches[0].trajectory_offsets.device,
    )
    return torch.cat((*pieces, last), dim=0)


def _pad_step_columns(values: Tensor, *, max_steps: int) -> Tensor:
    current = int(values.shape[1])
    if current == max_steps:
        return values
    padding = torch.zeros(
        (int(values.shape[0]), max_steps - current),
        dtype=values.dtype,
        device=values.device,
    )
    return torch.cat((values, padding), dim=1)


def _pad_step_mask_columns(values: Tensor, *, max_steps: int) -> Tensor:
    current = int(values.shape[1])
    if current == max_steps:
        return values
    padding = torch.zeros(
        (
            int(values.shape[0]),
            max_steps - current,
            int(values.shape[2]),
        ),
        dtype=values.dtype,
        device=values.device,
    )
    return torch.cat((values, padding), dim=1)
