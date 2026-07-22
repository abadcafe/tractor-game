"""Batched torch policy sampling over staged policy requests."""

from __future__ import annotations

import torch

from server.foundation.result import Ok, Rejected
from server.training.model import ModelConfig, TractorPolicyModel
from server.training.policy_inference_batch import (
    DevicePolicyRequestBatch,
)
from server.training.policy_sampling import (
    CompactPolicyDecisionBatch,
    PolicySampleColumns,
)
from server.training.policy_sampling.model_rank_sample_arena import (
    ModelRankSampleArena,
)
from server.training.semantic_action_plan import (
    ActionSampler,
)
from server.training.tensor_finiteness import (
    NamedTensorCheck,
    reject_if_non_finite,
)

type PolicySamplingResult = Ok[PolicySampleColumns] | Rejected
type PolicySamplingDecisionResult = (
    Ok[CompactPolicyDecisionBatch] | Rejected
)


def sample_policy_batch(
    *,
    model: TractorPolicyModel,
    config: ModelConfig,
    device: torch.device,
    requests: DevicePolicyRequestBatch,
    sampler: ActionSampler,
) -> PolicySamplingResult:
    """Sample policy decisions for a staged request batch."""
    model.eval()
    with torch.no_grad():
        observation_batch = requests.observation_batch
        encoding = model.encode_observations(observation_batch)
        values = model.value_estimates(encoding)
        value_check = reject_if_non_finite(
            (
                NamedTensorCheck(
                    tensor=values,
                    reason="policy value estimate must be finite",
                ),
            )
        )
        if isinstance(value_check, Rejected):
            return value_check

        logit_decoder = model.begin_action_decode_session(
            encoding,
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
                old_values=values,
            )
        )


def sample_policy_batch_into_arena(
    *,
    model: TractorPolicyModel,
    config: ModelConfig,
    device: torch.device,
    requests: DevicePolicyRequestBatch,
    sampler: ActionSampler,
    sample_arena: ModelRankSampleArena,
) -> PolicySamplingDecisionResult:
    """Sample policy decisions and append replay tensors to an arena."""
    model.eval()
    with torch.no_grad():
        observation_batch = requests.observation_batch
        encoding = model.encode_observations(observation_batch)
        values = model.value_estimates(encoding)
        value_check = reject_if_non_finite(
            (
                NamedTensorCheck(
                    tensor=values,
                    reason="policy value estimate must be finite",
                ),
            )
        )
        if isinstance(value_check, Rejected):
            return value_check

        logit_decoder = model.begin_action_decode_session(
            encoding,
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
        return sample_arena.store_sampled_result(
            policy_versions=requests.policy_versions,
            observation_batch=observation_batch,
            action_sample=action_result.value,
            old_values=values,
        )
