"""Batched torch policy sampling over staged policy requests."""

from __future__ import annotations

from dataclasses import dataclass

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
    DeviceLegalChoices,
    action_prefix_batch,
    advance_action_state,
    initial_action_state,
    legal_token_choices,
    sample_legal_choices,
    sample_legal_token_error_reason,
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
        legal_choice_batches: list[DeviceLegalChoices] = []
        selected_choice_offsets: list[Tensor] = []
        for argument_index in range(SEMANTIC_CODEC.max_argument_tokens):
            choices = legal_token_choices(
                batch=action_batch, state=state
            )
            active = ~state.done
            scores = model.score_argument_prefixes(
                encoding,
                prefix=action_prefix_batch(
                    state, generated_token_count=argument_index
                ),
            )
            sampled = sample_legal_choices(
                argument_logits=scores.argument_logits,
                legal_choices=choices,
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
            legal_choice_batches.append(choices)
            selected_choice_offsets.append(
                sampled.selected_choice_offsets
            )
            state = advance_action_state(
                batch=action_batch,
                state=state,
                selected_token_ids=sampled.token_ids,
                choice_counts=choices.choice_counts,
            )
            if device.type == "cpu" and _cpu_all_done(state.done):
                break
        error_code = _set_error_if(
            error_code, (~state.done).any(), _ERROR_UNTERMINATED
        )
        error_value = _error_code_value(error_code)
        if error_value != 0:
            return Rejected(reason=_error_reason(error_value))
        replay_choices = _padded_legal_choices(
            legal_choices=tuple(legal_choice_batches),
            selected_choice_offsets=tuple(selected_choice_offsets),
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
                legal_choice_ids_padded=replay_choices.choice_ids,
                legal_choice_masks_padded=replay_choices.choice_masks,
                selected_choice_offsets_padded=(
                    replay_choices.selected_offsets
                ),
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


@dataclass(frozen=True, slots=True)
class _PaddedLegalChoices:
    choice_ids: Tensor
    choice_masks: Tensor
    selected_offsets: Tensor


def _padded_legal_choices(
    *,
    legal_choices: tuple[DeviceLegalChoices, ...],
    selected_choice_offsets: tuple[Tensor, ...],
    batch_size: int,
    device: torch.device,
) -> _PaddedLegalChoices:
    assert legal_choices
    assert len(legal_choices) == len(selected_choice_offsets)
    max_choice_count = max(
        _max_choice_count(choices.choice_counts)
        for choices in legal_choices
    )
    step_ids: list[Tensor] = []
    step_masks: list[Tensor] = []
    for choices in legal_choices:
        ids, masks = _padded_choice_step(
            choices=choices,
            batch_size=batch_size,
            max_choice_count=max_choice_count,
            device=device,
        )
        step_ids.append(ids)
        step_masks.append(masks)
    width = len(legal_choices)
    choice_ids = torch.zeros(
        (
            batch_size,
            SEMANTIC_CODEC.max_argument_tokens,
            max_choice_count,
        ),
        dtype=torch.int16,
        device=device,
    )
    choice_masks = torch.zeros(
        choice_ids.shape, dtype=torch.bool, device=device
    )
    selected_offsets = torch.zeros(
        (batch_size, SEMANTIC_CODEC.max_argument_tokens),
        dtype=torch.long,
        device=device,
    )
    choice_ids[:, :width, :] = torch.stack(step_ids, dim=1)
    choice_masks[:, :width, :] = torch.stack(step_masks, dim=1)
    selected_offsets[:, :width] = torch.stack(
        selected_choice_offsets, dim=1
    )
    return _PaddedLegalChoices(
        choice_ids=choice_ids,
        choice_masks=choice_masks,
        selected_offsets=selected_offsets,
    )


def _padded_choice_step(
    *,
    choices: DeviceLegalChoices,
    batch_size: int,
    max_choice_count: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    ids = torch.zeros(
        (batch_size, max_choice_count), dtype=torch.int16, device=device
    )
    masks = torch.zeros(
        (batch_size, max_choice_count), dtype=torch.bool, device=device
    )
    if int(choices.token_ids.shape[0]) == 0:
        return ids, masks
    local_offsets = torch.arange(
        int(choices.token_ids.shape[0]), dtype=torch.long, device=device
    ) - choices.choice_offsets.index_select(0, choices.row_indices)
    ids[choices.row_indices, local_offsets] = choices.token_ids.to(
        dtype=torch.int16
    )
    masks[choices.row_indices, local_offsets] = True
    return ids, masks


def _max_choice_count(choice_counts: Tensor) -> int:
    result = int(choice_counts.detach().cpu().max().item())
    assert result > 0
    return result
