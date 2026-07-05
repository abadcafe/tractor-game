"""Batched torch policy sampling over policy request frames."""

from __future__ import annotations

import torch
from torch import Tensor

from server.result import Ok, Rejected
from server.training.config import ModelConfig
from server.training.model import TractorPolicyModel
from server.training.policy_request_frame import (
    PolicyRequestBatchFrame,
    PolicyRequestFrame,
    policy_request_batch_to_device,
)
from server.training.policy_sampling import (
    DeviceDecisionReplayRecord,
    SampledPolicyDecision,
)
from server.training.semantic_action_plan import (
    action_prefix_batch,
    action_trace_ids,
    advance_action_state,
    initial_action_state,
    legal_token_mask,
    sample_legal_token_error_reason,
    sample_legal_tokens,
)
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tensor_finiteness import (
    NamedTensorCheck,
    reject_if_non_finite,
)
from server.training.tensorize import ObservationTensorBatch

type PolicySamplingResult = Ok[SampledPolicyDecision] | Rejected

_ERROR_UNTERMINATED = 1000


def sample_policy_decisions(
    *,
    model: TractorPolicyModel,
    config: ModelConfig,
    device: torch.device,
    requests: PolicyRequestBatchFrame,
) -> tuple[PolicySamplingResult, ...]:
    """Sample policy decisions for a batch of request frames."""
    frames = requests.frames
    model.eval()
    with torch.no_grad():
        request_batch = policy_request_batch_to_device(
            batch=requests,
            max_observation_tokens=config.max_tokens,
            device=device,
        )
        observation_batch = request_batch.observation_batch
        action_batch = request_batch.action_plan_batch
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
            return tuple(value_check for _ in frames)
        state = initial_action_state(action_batch)
        log_probabilities = torch.zeros(
            (len(frames),), dtype=values.dtype, device=device
        )
        error_code = torch.zeros((), dtype=torch.long, device=device)
        legal_masks: list[Tensor] = []
        for argument_index in range(SEMANTIC_CODEC.max_argument_tokens):
            mask = legal_token_mask(batch=action_batch, state=state)
            active = ~state.done
            scores = model.score_argument_prefixes(
                encoding,
                prefix=action_prefix_batch(
                    state, generated_token_count=argument_index
                ),
            )
            sampled = sample_legal_tokens(
                argument_logits=scores.argument_logits,
                legal_token_mask=mask,
                thresholds=request_batch.sampling_thresholds[
                    :, argument_index
                ],
            )
            error_code = _merge_error_code(
                error_code, sampled.error_code
            )
            log_probabilities = log_probabilities + torch.where(
                active,
                sampled.selected_log_probabilities,
                torch.zeros_like(sampled.selected_log_probabilities),
            )
            legal_masks.append(mask)
            state = advance_action_state(
                batch=action_batch,
                state=state,
                selected_token_ids=sampled.token_ids,
                legal_mask=mask,
            )
            if device.type == "cpu" and _cpu_all_done(state.done):
                break
        error_code = _set_error_if(
            error_code, (~state.done).any(), _ERROR_UNTERMINATED
        )
        error_value = _error_code_value(error_code)
        if error_value != 0:
            rejection = Rejected(reason=_error_reason(error_value))
            return tuple(rejection for _ in frames)
        mask_stack = torch.stack(legal_masks, dim=1)
        trace_ids = action_trace_ids(state)
        choice_counts = _int_tensor_values(state.choice_counts)
        return tuple(
            _finish_sample(
                request=request,
                observation_batch=observation_batch,
                state_index=index,
                trace_token_ids=trace_ids[index],
                legal_masks=mask_stack[index, : len(trace_ids[index])],
                old_log_probability=log_probabilities[index],
                old_value=values[index],
                choice_count=choice_counts[index],
            )
            for index, request in enumerate(frames)
        )


def _finish_sample(
    *,
    request: PolicyRequestFrame,
    observation_batch: ObservationTensorBatch,
    state_index: int,
    trace_token_ids: tuple[int, ...],
    legal_masks: Tensor,
    old_log_probability: Tensor,
    old_value: Tensor,
    choice_count: int,
) -> PolicySamplingResult:
    if choice_count <= 0:
        return Rejected(reason="policy action has no legal choices")
    return Ok(
        value=SampledPolicyDecision(
            trace_token_ids=trace_token_ids,
            replay_record=DeviceDecisionReplayRecord(
                policy_version=request.decision_key.policy_version,
                observation_batch=_observation_row(
                    observation_batch, index=state_index
                ),
                selected_token_ids=torch.tensor(
                    trace_token_ids,
                    dtype=torch.long,
                    device=old_value.device,
                ),
                legal_token_masks=legal_masks,
                old_log_probability=old_log_probability,
                old_value=old_value,
            ),
            choice_count=choice_count,
        )
    )


def _int_tensor_values(values: Tensor) -> tuple[int, ...]:
    cpu_values = values.detach().cpu()
    return tuple(
        int(cpu_values[index].item())
        for index in range(int(cpu_values.shape[0]))
    )


def _cpu_all_done(done: Tensor) -> bool:
    assert done.device.type == "cpu"
    return bool(done.all().item())


def _observation_row(
    batch: ObservationTensorBatch, *, index: int
) -> ObservationTensorBatch:
    assert index >= 0
    return ObservationTensorBatch(
        component_ids=batch.component_ids[index : index + 1],
        numeric_values=batch.numeric_values[index : index + 1],
        numeric_masks=batch.numeric_masks[index : index + 1],
    )


def _set_error_if(
    error_code: Tensor, condition: Tensor, code: int
) -> Tensor:
    assert error_code.shape == ()
    assert condition.shape == ()
    return torch.where(
        (error_code == 0) & condition,
        torch.full(
            (), code, dtype=torch.long, device=error_code.device
        ),
        error_code,
    )


def _merge_error_code(current: Tensor, incoming: Tensor) -> Tensor:
    assert current.shape == ()
    assert incoming.shape == ()
    return torch.where(
        (current == 0) & (incoming != 0), incoming, current
    )


def _error_code_value(error_code: Tensor) -> int:
    return int(error_code.detach().cpu().item())


def _error_reason(error_code: int) -> str:
    if error_code == _ERROR_UNTERMINATED:
        return "policy semantic action did not terminate"
    sampling_reason = sample_legal_token_error_reason(error_code)
    if sampling_reason is not None:
        return sampling_reason
    return "policy sampling failed"
