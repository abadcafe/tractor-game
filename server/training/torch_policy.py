"""Torch-backed TrainingPolicy implementation."""

from __future__ import annotations

import torch

from server.foundation.result import Ok, Rejected
from server.training.config import ModelConfig
from server.training.legal_actions import LegalActionIndex
from server.training.model import TractorPolicyModel
from server.training.observation import Observation
from server.training.policy import PolicyDecision
from server.training.policy_inference_batch import (
    PolicyRequestCompiler,
    PolicyRequestInput,
    PolicyRequestRoute,
    build_completed_policy_responses,
    decode_policy_response,
    materialize_borrowed_policy_request_batch,
)
from server.training.policy_sampling.model_rank_sample_arena import (
    ModelRankSampleArena,
)
from server.training.sampling import PolicyDecisionKey
from server.training.semantic_action_plan import (
    ActionSampler,
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
        self.compiler = PolicyRequestCompiler(batch_capacity=1)
        self.sampler = ActionSampler.create(
            batch_capacity=1,
            device=device,
        )

    async def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected:
        route = PolicyRequestRoute(
            worker_index=0,
            request_id=decision_key.decision_index,
        )
        compiled_result = self.compiler.compile_batch(
            (
                PolicyRequestInput(
                    route=route,
                    observation=observation,
                    legal_actions=legal_actions,
                    decision_key=decision_key,
                ),
            ),
        )
        if isinstance(compiled_result, Rejected):
            return compiled_result
        request_result = materialize_borrowed_policy_request_batch(
            batch=compiled_result.value,
            device=self.device,
        )
        if isinstance(request_result, Rejected):
            return request_result
        decision_result = sample_policy_batch_into_arena(
            model=self.model,
            config=self.config,
            device=self.device,
            requests=request_result.value,
            sampler=self.sampler,
            sample_arena=self.sample_arena,
        )
        if isinstance(decision_result, Rejected):
            return decision_result
        response_result = build_completed_policy_responses(
            routes=(route,), decisions=decision_result.value
        )
        if isinstance(response_result, Rejected):
            return response_result
        assert len(response_result.value) == 1
        response = response_result.value[0]
        return decode_policy_response(
            legal_actions=legal_actions, response=response
        )
