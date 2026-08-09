"""Batched torch policy sampling over staged policy requests."""

from __future__ import annotations

import torch

from server.foundation.result import Ok, Rejected
from server.policy_model.actions.decoding import (
    ActionSampler,
)
from server.policy_model.network import PolicyModel
from server.training.device_execution import DeviceExecutionPolicy
from server.training.rollout_inference.batch import (
    DevicePolicyRequestBatch,
)
from server.training.rollout_inference.samples import (
    CompactPolicyDecisionBatch,
    ModelRankSampleArena,
    PolicySampleColumns,
)

type PolicySamplingResult = Ok[PolicySampleColumns] | Rejected
type PolicySamplingDecisionResult = (
    Ok[CompactPolicyDecisionBatch] | Rejected
)


def sample_policy_batch(
    *,
    model: PolicyModel,
    device: torch.device,
    requests: DevicePolicyRequestBatch,
    sampler: ActionSampler,
    execution_policy: DeviceExecutionPolicy,
) -> PolicySamplingResult:
    """Sample policy decisions for a staged request batch."""
    assert execution_policy.device == device
    _ = model.eval()
    with torch.no_grad(), execution_policy.autocast():
        observation_batch = requests.observation_batch
        encoding = model.encode_observations(observation_batch)
        logit_decoder = model.begin_action_decode_session(
            encoding,
            source_rows=torch.arange(
                encoding.batch_size,
                dtype=torch.long,
                device=encoding.device,
            ),
            max_steps=requests.padded_generation_steps,
        )
        action_result = sampler.sample(
            action_batch=requests.action_plan_batch,
            generation_step_counts=requests.generation_step_counts,
            sampling_thresholds=requests.sampling_thresholds,
            padded_generation_steps=requests.padded_generation_steps,
            logit_decoder=logit_decoder,
        )
        if isinstance(action_result, Rejected):
            return action_result
        action = action_result.value
        return Ok(
            value=PolicySampleColumns(
                policy_versions=requests.policy_versions,
                observation_batch=observation_batch,
                choice_ids_padded=action.choice_ids_padded,
                active_sample_indices=action.active_sample_indices,
                active_step_indices=action.active_step_indices,
                legal_choice_masks=action.legal_choice_masks,
                step_counts=action.step_counts,
                choice_counts=action.choice_counts,
                old_log_probabilities=action.log_probabilities,
            )
        )


def sample_policy_batch_into_arena(
    *,
    model: PolicyModel,
    device: torch.device,
    requests: DevicePolicyRequestBatch,
    sampler: ActionSampler,
    sample_arena: ModelRankSampleArena,
    execution_policy: DeviceExecutionPolicy,
) -> PolicySamplingDecisionResult:
    """Sample policy decisions and append replay tensors to an arena."""
    assert execution_policy.device == device
    _ = model.eval()
    with torch.no_grad(), execution_policy.autocast():
        observation_batch = requests.observation_batch
        encoding = model.encode_observations(observation_batch)
        logit_decoder = model.begin_action_decode_session(
            encoding,
            source_rows=torch.arange(
                encoding.batch_size,
                dtype=torch.long,
                device=encoding.device,
            ),
            max_steps=requests.padded_generation_steps,
        )
        action_result = sampler.sample(
            action_batch=requests.action_plan_batch,
            generation_step_counts=requests.generation_step_counts,
            sampling_thresholds=requests.sampling_thresholds,
            padded_generation_steps=requests.padded_generation_steps,
            logit_decoder=logit_decoder,
        )
        if isinstance(action_result, Rejected):
            return action_result
        action = action_result.value
        return sample_arena.store_sampled_result(
            policy_versions=requests.policy_versions,
            observation_batch=observation_batch,
            action_sample=action,
        )
