"""Torch-backed TrainingPolicy implementation."""

from __future__ import annotations

import torch

from server.result import Ok, Rejected
from server.training.config import ModelConfig
from server.training.legal_actions import LegalActionIndex
from server.training.model import TractorPolicyModel
from server.training.observation import Observation
from server.training.policy import PolicyDecision
from server.training.policy_request_frame import (
    CompletedPolicyResponseFrame,
    PolicyRequestBatchFrame,
    build_policy_request_frame,
    decode_policy_response,
)
from server.training.policy_sampling.replay_arena import (
    ModelRankReplayArena,
)
from server.training.sampling import PolicyDecisionKey
from server.training.torch_sampler import sample_policy_decisions


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
        self.replay_arena = ModelRankReplayArena(
            model_rank_index=0,
            device=device,
        )

    def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected:
        frame_result = build_policy_request_frame(
            observation=observation,
            legal_actions=legal_actions,
            decision_key=decision_key,
        )
        if isinstance(frame_result, Rejected):
            return frame_result
        results = sample_policy_decisions(
            model=self.model,
            config=self.config,
            device=self.device,
            requests=PolicyRequestBatchFrame(
                frames=(frame_result.value,)
            ),
        )
        assert len(results) == 1
        result = results[0]
        if isinstance(result, Rejected):
            return result
        sample = result.value
        handle = self.replay_arena.store(record=sample.replay_record)
        return decode_policy_response(
            legal_actions=legal_actions,
            response=CompletedPolicyResponseFrame(
                trace_token_ids=sample.trace_token_ids,
                decision_handle=handle,
                choice_count=sample.choice_count,
            ),
        )
