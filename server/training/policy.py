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
    uniform_choice_offset,
)
from server.training.semantic_action_plan import (
    action_trace_ids,
    advance_action_state,
    compile_legal_action_frame,
    initial_action_state,
    legal_token_mask,
    plan_batch_to_device,
    semantic_trace_from_token_ids,
)
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.semantic_actions.values import GeneratedAction


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

    def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected: ...


class RandomTrainingPolicy:
    """Verified random semantic policy for smoke runs."""

    def decide(
        self,
        observation: Observation,
        legal_actions: LegalActionIndex,
        decision_key: PolicyDecisionKey,
    ) -> Ok[PolicyDecision] | Rejected:
        device = torch.device("cpu")
        batch = plan_batch_to_device(
            (compile_legal_action_frame(legal_actions),), device=device
        )
        state = initial_action_state(batch)
        for _ in range(SEMANTIC_CODEC.max_argument_tokens):
            if bool(state.done.detach().cpu().item()):
                break
            mask = legal_token_mask(batch=batch, state=state)
            legal_ids = torch.nonzero(mask[0], as_tuple=False).squeeze(
                1
            )
            choice_count = int(legal_ids.shape[0])
            if choice_count <= 0:
                return Rejected(
                    reason="random policy has no legal token"
                )
            selected_argument_offset = uniform_choice_offset(
                key=decision_key,
                argument_index=int(state.step_counts[0].item()),
                choice_count=choice_count,
            )
            selected_token = legal_ids[selected_argument_offset].view(1)
            state = advance_action_state(
                batch=batch,
                state=state,
                selected_token_ids=selected_token,
                legal_mask=mask,
            )
        else:
            assert False
        trace_ids = action_trace_ids(state)[0]
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
                    slot_index=decision_key.decision_index,
                    slot_generation=0,
                ),
                choice_count=int(state.choice_counts[0].item()),
            )
        )
