"""Batched torch policy sampling over staged policy requests."""

from __future__ import annotations

import torch

from server.result import Ok, Rejected
from server.training.config import ModelConfig
from server.training.model import TractorPolicyModel
from server.training.policy_inference_batch import (
    DevicePolicyRequestBatch,
)
from server.training.policy_sampling import (
    ModelRankPolicyDecision,
    PolicySampleColumns,
)
from server.training.policy_sampling.model_rank_sample_arena import (
    ModelRankSampleArena,
)
from server.training.semantic_action_plan import (
    SemanticActionSampler,
)
from server.training.tensor_finiteness import (
    NamedTensorCheck,
    reject_if_non_finite,
)

type PolicySamplingResult = Ok[PolicySampleColumns] | Rejected
type PolicySamplingDecisionResult = (
    Ok[ModelRankPolicyDecision] | Rejected
)


def sample_policy_batch(
    *,
    model: TractorPolicyModel,
    config: ModelConfig,
    device: torch.device,
    requests: DevicePolicyRequestBatch,
    sampler: SemanticActionSampler,
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

        logit_decoder = model.begin_argument_decode_session(
            encoding,
            max_steps=requests.padded_generation_steps,
        )
        semantic_result = sampler.sample(
            action_batch=requests.action_plan_batch,
            generation_step_counts=requests.generation_step_counts,
            sampling_thresholds=requests.sampling_thresholds,
            padded_generation_steps=requests.padded_generation_steps,
            logit_decoder=logit_decoder,
        )
        if isinstance(semantic_result, Rejected):
            return semantic_result
        semantic = semantic_result.value
        return Ok(
            value=PolicySampleColumns(
                policy_versions=requests.policy_versions,
                observation_batch=observation_batch,
                selected_token_ids_padded=(
                    semantic.selected_token_ids_padded
                ),
                choice_token_ids=semantic.choice_token_ids,
                choice_masks=semantic.choice_masks,
                selected_choice_offsets=(
                    semantic.selected_choice_offsets
                ),
                step_counts=semantic.step_counts,
                choice_counts=semantic.choice_counts,
                old_log_probabilities=semantic.log_probabilities,
                old_values=values,
            )
        )


def sample_policy_batch_into_arena(
    *,
    model: TractorPolicyModel,
    config: ModelConfig,
    device: torch.device,
    requests: DevicePolicyRequestBatch,
    sampler: SemanticActionSampler,
    sample_arena: ModelRankSampleArena,
) -> tuple[PolicySamplingDecisionResult, ...]:
    """Sample policy decisions and append replay tensors to an arena."""
    batch_size = len(requests.policy_versions)
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
            return tuple(value_check for _ in range(batch_size))

        logit_decoder = model.begin_argument_decode_session(
            encoding,
            max_steps=requests.padded_generation_steps,
        )
        semantic_result = sampler.sample(
            action_batch=requests.action_plan_batch,
            generation_step_counts=requests.generation_step_counts,
            sampling_thresholds=requests.sampling_thresholds,
            padded_generation_steps=requests.padded_generation_steps,
            logit_decoder=logit_decoder,
        )
        if isinstance(semantic_result, Rejected):
            return tuple(semantic_result for _ in range(batch_size))
        return sample_arena.store_sampled_result(
            policy_versions=requests.policy_versions,
            observation_batch=observation_batch,
            semantic_sample=semantic_result.value,
            old_values=values,
        )
