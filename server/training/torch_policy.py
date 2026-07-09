"""Torch-backed TrainingPolicy implementation."""

from __future__ import annotations

import torch

from server.result import Ok, Rejected
from server.training.config import ModelConfig
from server.training.legal_actions import LegalActionIndex
from server.training.model import TractorPolicyModel
from server.training.observation import Observation
from server.training.policy import PolicyDecision
from server.training.policy_inference_batch import (
    CompletedPolicyResponse,
    PolicyRequestInput,
    PolicyRequestRoute,
    decode_policy_response,
    materialize_policy_request_inputs,
)
from server.training.policy_sampling.model_rank_sample_arena import (
    ModelRankSampleArena,
)
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_action_plan import (
    SemanticActionSampler,
)
from server.training.torch_sampler import sample_policy_batch_into_arena


class TorchTrainingPolicy:
    """Sample semantic argument traces from a torch model."""

    def __init__(
        self,
        *,
        model: TractorPolicyModel,
        config: ModelConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.sample_arena = ModelRankSampleArena(
            model_rank_index=0,
            device=device,
        )
        self.sampler = SemanticActionSampler.create(
            batch_capacity=1,
            device=device,
        )

    async def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected:
        request_result = materialize_policy_request_inputs(
            requests=(
                PolicyRequestInput(
                    route=PolicyRequestRoute(
                        worker_index=0,
                        request_id=decision_key.decision_index,
                    ),
                    observation=observation,
                    legal_actions=legal_actions,
                    decision_key=decision_key,
                ),
            ),
            batch_capacity=1,
            max_observation_tokens=self.config.max_tokens,
            device=self.device,
        )
        if isinstance(request_result, Rejected):
            return request_result
        decisions = sample_policy_batch_into_arena(
            model=self.model,
            config=self.config,
            device=self.device,
            requests=request_result.value,
            sampler=self.sampler,
            sample_arena=self.sample_arena,
        )
        assert len(decisions) == 1
        decision_result = decisions[0]
        if isinstance(decision_result, Rejected):
            return decision_result
        decision = decision_result.value
        return decode_policy_response(
            legal_actions=legal_actions,
            response=CompletedPolicyResponse(
                route=PolicyRequestRoute(
                    worker_index=0,
                    request_id=decision_key.decision_index,
                ),
                trace_token_ids=decision.trace_token_ids,
                decision_handle_model_rank=(
                    decision.decision_handle.model_rank_index
                ),
                decision_handle_policy_version=(
                    decision.decision_handle.policy_version
                ),
                decision_handle_row_index=(
                    decision.decision_handle.row_index
                ),
                choice_count=decision.choice_count,
            ),
        )
