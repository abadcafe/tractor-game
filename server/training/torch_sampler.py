"""Batched torch policy sampling over staged policy requests."""

from __future__ import annotations

import torch
from torch import Tensor

from server.result import Ok, Rejected
from server.training.config import ModelConfig
from server.training.model import TractorPolicyModel
from server.training.policy_inference_wire import (
    DevicePolicyRequestBatch,
)
from server.training.policy_sampling import (
    SampledPolicyBatch,
)
from server.training.semantic_action_plan import (
    action_prefix_batch,
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

type PolicySamplingResult = Ok[SampledPolicyBatch] | Rejected

_ERROR_UNTERMINATED = 1000


def sample_policy_batch(
    *,
    model: TractorPolicyModel,
    config: ModelConfig,
    device: torch.device,
    requests: DevicePolicyRequestBatch,
) -> PolicySamplingResult:
    """Sample policy decisions for a staged request batch."""
    batch_size = len(requests.policy_versions)
    model.eval()
    with torch.no_grad():
        observation_batch = requests.observation_batch
        action_batch = requests.action_plan_batch
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
        state = initial_action_state(action_batch)
        log_probabilities = torch.zeros(
            (batch_size,), dtype=values.dtype, device=device
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
                thresholds=requests.sampling_thresholds[
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
            return Rejected(reason=_error_reason(error_value))
        mask_stack = _padded_legal_masks(
            legal_masks=tuple(legal_masks),
            batch_size=batch_size,
            device=device,
        )
        status_codes = torch.zeros(
            (batch_size,), dtype=torch.long, device=device
        )
        return Ok(
            value=SampledPolicyBatch(
                policy_versions=requests.policy_versions,
                status_codes=status_codes,
                observation_batch=observation_batch,
                selected_token_ids_padded=state.selected_token_ids,
                legal_token_masks_padded=mask_stack,
                step_counts=state.step_counts,
                choice_counts=state.choice_counts,
                old_log_probabilities=log_probabilities,
                old_values=values,
            )
        )


def _cpu_all_done(done: Tensor) -> bool:
    assert done.device.type == "cpu"
    return bool(done.all().item())


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


def _padded_legal_masks(
    *,
    legal_masks: tuple[Tensor, ...],
    batch_size: int,
    device: torch.device,
) -> Tensor:
    assert legal_masks
    mask_stack = torch.stack(legal_masks, dim=1)
    width = int(mask_stack.shape[1])
    if width == SEMANTIC_CODEC.max_argument_tokens:
        return mask_stack
    result = torch.zeros(
        (
            batch_size,
            SEMANTIC_CODEC.max_argument_tokens,
            SEMANTIC_CODEC.argument_vocab_size,
        ),
        dtype=torch.bool,
        device=device,
    )
    result[:, :width, :] = mask_stack
    return result
