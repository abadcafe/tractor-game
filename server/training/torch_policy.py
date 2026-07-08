"""Torch-backed TrainingPolicy implementation."""

from __future__ import annotations

import torch

from server.result import Ok, Rejected
from server.training.config import ModelConfig
from server.training.legal_actions import LegalActionIndex
from server.training.model import TractorPolicyModel
from server.training.observation import Observation
from server.training.policy import PolicyDecision
from server.training.policy_inference_wire import (
    CompletedPolicyResponse,
    PolicyRequestRoute,
    PolicyRequestWireBatch,
    build_policy_request_wire,
    decode_policy_response,
)
from server.training.policy_sampling.model_rank_sample_arena import (
    ModelRankSampleArena,
)
from server.training.runtime.model_rank.staging import (
    stage_policy_request_wires,
)
from server.training.sampling import PolicyDecisionKey
from server.training.torch_sampler import sample_policy_batch


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

    async def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected:
        request_result = build_policy_request_wire(
            max_observation_tokens=self.config.max_tokens,
            worker_index=0,
            request_id=decision_key.decision_index,
            observation=observation,
            legal_actions=legal_actions,
            decision_key=decision_key,
        )
        if isinstance(request_result, Rejected):
            return request_result
        staged_result = stage_policy_request_wires(
            requests=PolicyRequestWireBatch(
                requests=(request_result.value,)
            ),
            max_observation_tokens=self.config.max_tokens,
            device=self.device,
        )
        if isinstance(staged_result, Rejected):
            return staged_result
        sampled_result = sample_policy_batch(
            model=self.model,
            config=self.config,
            device=self.device,
            requests=staged_result.value.device_batch,
        )
        if isinstance(sampled_result, Rejected):
            return sampled_result
        decisions = self.sample_arena.store_sampled_batch(
            batch=sampled_result.value
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
                decision_handle_slot_index=(
                    decision.decision_handle.slot_index
                ),
                decision_handle_slot_generation=(
                    decision.decision_handle.slot_generation
                ),
                choice_count=decision.choice_count,
            ),
        )
