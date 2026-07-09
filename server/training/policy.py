"""Policy interfaces used by TrainingPlayer and trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from server.result import Ok, Rejected
from server.training.legal_actions import LegalActionIndex
from server.training.observation import Observation
from server.training.policy_sampling.records import DecisionHandle
from server.training.sampling import (
    PolicyDecisionKey,
    policy_choice_threshold,
)
from server.training.semantic_action_plan import (
    SemanticActionSampler,
    SemanticArgumentLogitDecoder,
    action_plan_generation_step_count,
    compile_legal_action_frame,
    plan_batch_to_device,
    semantic_trace_from_token_ids,
)
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.semantic_actions.values import GeneratedAction


@dataclass(slots=True)
class _UniformArgumentLogitDecoder:
    batch_size: int
    device: torch.device

    def next_logits(self) -> torch.Tensor:
        return torch.zeros(
            (self.batch_size, SEMANTIC_CODEC.argument_vocab_size),
            dtype=torch.float32,
            device=self.device,
        )

    def advance(self, selected_token_ids: torch.Tensor) -> None:
        assert selected_token_ids.shape == (self.batch_size,)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Policy output plus a replay handle for one generated action."""

    action: GeneratedAction
    decision_handle: DecisionHandle
    choice_count: int

    def __post_init__(self) -> None:
        assert self.choice_count > 0


class TrainingPolicy(Protocol):
    """Policy abstraction consumed by TrainingPlayer."""

    async def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected: ...


class RandomTrainingPolicy:
    """Verified random semantic policy for smoke runs."""

    async def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected:
        device = torch.device("cpu")
        action_plan = compile_legal_action_frame(legal_actions)
        generation_step_count = action_plan_generation_step_count(
            action_plan
        )
        batch = plan_batch_to_device((action_plan,), device=device)
        thresholds = torch.tensor(
            (
                tuple(
                    policy_choice_threshold(
                        key=decision_key,
                        argument_index=argument_index,
                    )
                    for argument_index in range(generation_step_count)
                ),
            ),
            dtype=torch.float64,
            device=device,
        )
        sampler = SemanticActionSampler.create(
            batch_capacity=1, device=device
        )

        logit_decoder: SemanticArgumentLogitDecoder = (
            _UniformArgumentLogitDecoder(batch_size=1, device=device)
        )
        sample_result = sampler.sample(
            action_batch=batch,
            generation_step_counts=torch.tensor(
                (generation_step_count,),
                dtype=torch.long,
                device=device,
            ),
            sampling_thresholds=thresholds,
            padded_generation_steps=generation_step_count,
            logit_decoder=logit_decoder,
        )
        if isinstance(sample_result, Rejected):
            return sample_result
        sample = sample_result.value
        step_count = int(sample.step_counts[0].detach().cpu().item())
        trace_ids = tuple(
            int(
                sample.selected_token_ids_padded[0, index]
                .detach()
                .cpu()
                .item()
            )
            for index in range(step_count)
        )
        trace_result = semantic_trace_from_token_ids(trace_ids)
        if isinstance(trace_result, Rejected):
            return trace_result
        trace = trace_result.value
        decoded = legal_actions.decode(trace)
        assert isinstance(decoded, Ok)
        return Ok(
            value=PolicyDecision(
                action=decoded.value,
                decision_handle=DecisionHandle(
                    model_rank_index=0,
                    policy_version=decision_key.policy_version,
                    row_index=decision_key.decision_index,
                ),
                choice_count=int(
                    sample.choice_counts[0].detach().cpu().item()
                ),
            )
        )
