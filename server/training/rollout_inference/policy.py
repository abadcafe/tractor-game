"""Torch-backed TrainingPolicy implementation."""

from __future__ import annotations

from typing import final

import torch

from server.foundation.result import Ok, Rejected
from server.policy_model.actions import LegalActionSpace
from server.policy_model.actions.decoding import (
    ActionSampler,
)
from server.policy_model.network import PolicyModel
from server.policy_model.observation import Observation
from server.training.device_execution import DeviceExecutionPolicy
from server.training.rollout.policy import PolicyDecision
from server.training.rollout_inference.batch import (
    PolicyRequestCompiler,
    PolicyRequestInput,
    PolicyRequestRoute,
    build_completed_policy_responses,
    decode_policy_response,
    materialize_borrowed_policy_request_batch,
)
from server.training.rollout_inference.randomness import (
    TrainingDecisionKey,
)
from server.training.rollout_inference.sampler import (
    ObservationValueEvaluator,
    sample_policy_batch_into_arena,
)
from server.training.rollout_inference.samples import (
    ModelRankSampleArena,
)


@final
class TorchTrainingPolicy:
    """Sample semantic argument traces from a torch model."""

    def __init__(
        self,
        *,
        model: PolicyModel,
        value_evaluator: ObservationValueEvaluator,
        device: torch.device,
    ) -> None:
        self.model = model
        self.value_evaluator = value_evaluator
        self.device = device
        self.execution_policy = DeviceExecutionPolicy.for_device(device)
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
        legal_actions: LegalActionSpace,
        decision_key: TrainingDecisionKey,
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
            device=self.device,
            requests=request_result.value,
            sampler=self.sampler,
            sample_arena=self.sample_arena,
            value_evaluator=self.value_evaluator,
            execution_policy=self.execution_policy,
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
