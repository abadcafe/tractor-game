"""Current-policy evaluation of recorded PPO traces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server import result as _result
from server.training.model import (
    ObservationEncoding,
    TractorPolicyModel,
)
from server.training.ppo.minibatch import TensorizedPPOMinibatch
from server.training.ppo.profile import PPOProfileAccumulator
from server.training.ppo.replay_tensors import (
    PPOReplayTensorBatch,
)
from server.training.tensor_finiteness import (
    NamedTensorCheck,
    reject_if_non_finite,
)


@dataclass(frozen=True, slots=True)
class TraceBatchEval:
    """Current-model scores for a minibatch of recorded traces."""

    log_probabilities: Tensor
    values: Tensor
    entropies: Tensor


@dataclass(frozen=True, slots=True)
class ArgumentBatchEval:
    """Current-model scores for one batch of trace prefixes."""

    active_positions: Tensor
    log_probabilities: Tensor
    entropies: Tensor


def evaluate_trace_batch(
    *,
    model: TractorPolicyModel,
    minibatch: TensorizedPPOMinibatch,
    device: torch.device,
    profile: PPOProfileAccumulator,
) -> _result.Ok[TraceBatchEval] | _result.Rejected:
    """Evaluate recorded semantic traces under the current model."""
    assert not minibatch.is_empty()
    observation_batch = minibatch.observation_batch
    assert observation_batch is not None
    encode_start = profile.mark()
    encoding = model.encode_observations(observation_batch)
    profile.record_elapsed("observation_encode_seconds", encode_start)
    value_start = profile.mark()
    values = model.value_estimates(encoding)
    profile.record_elapsed("value_head_seconds", value_start)
    log_probability_sums = torch.zeros(
        (minibatch.local_count,), dtype=torch.float32, device=device
    )
    entropy_sums = torch.zeros(
        (minibatch.local_count,), dtype=torch.float32, device=device
    )
    replay = minibatch.replay
    assert replay is not None
    prefix_eval_result = _argument_batch_eval(
        model=model,
        encoding=encoding,
        replay=replay,
        profile=profile,
    )
    if isinstance(prefix_eval_result, _result.Rejected):
        return prefix_eval_result
    prefix_eval = prefix_eval_result.value
    log_probability_sums.index_add_(
        dim=0,
        index=prefix_eval.active_positions,
        source=prefix_eval.log_probabilities,
    )
    entropy_sums.index_add_(
        dim=0,
        index=prefix_eval.active_positions,
        source=prefix_eval.entropies,
    )
    return _result.Ok(
        value=TraceBatchEval(
            log_probabilities=log_probability_sums,
            values=values,
            entropies=entropy_sums,
        )
    )


def _argument_batch_eval(
    *,
    model: TractorPolicyModel,
    encoding: ObservationEncoding,
    replay: PPOReplayTensorBatch,
    profile: PPOProfileAccumulator,
) -> _result.Ok[ArgumentBatchEval] | _result.Rejected:
    assert int(replay.step_counts.shape[0]) > 0
    profile.record_argument_trace_lengths(replay.step_counts)
    decode_start = profile.mark()
    scores = model.score_argument_traces(
        encoding,
        selected_token_ids_padded=replay.selected_token_ids_padded,
        step_counts=replay.step_counts,
    )
    profile.record_elapsed("argument_decode_seconds", decode_start)
    active_positions = _active_sample_positions(
        step_mask=replay.step_mask
    )
    distribution_start = profile.mark()
    distribution_result = _evaluate_recorded_token_batch(
        argument_logits=scores.argument_logits[replay.step_mask],
        choice_token_ids=replay.choice_token_ids[replay.step_mask],
        choice_masks=replay.choice_masks[replay.step_mask],
        selected_choice_offsets=replay.selected_choice_offsets[
            replay.step_mask
        ],
    )
    if isinstance(distribution_result, _result.Rejected):
        return distribution_result
    distribution = distribution_result.value
    profile.record_elapsed(
        "argument_distribution_seconds", distribution_start
    )
    return _result.Ok(
        value=ArgumentBatchEval(
            active_positions=active_positions,
            log_probabilities=distribution.log_probabilities,
            entropies=distribution.entropies,
        )
    )


def _active_sample_positions(*, step_mask: Tensor) -> Tensor:
    assert step_mask.ndim == 2
    sample_positions = torch.arange(
        int(step_mask.shape[0]),
        dtype=torch.long,
        device=step_mask.device,
    ).unsqueeze(1)
    return sample_positions.expand_as(step_mask)[step_mask]


@dataclass(frozen=True, slots=True)
class _RecordedTokenEval:
    log_probabilities: Tensor
    entropies: Tensor


def _evaluate_recorded_token_batch(
    *,
    argument_logits: Tensor,
    choice_token_ids: Tensor,
    choice_masks: Tensor,
    selected_choice_offsets: Tensor,
) -> _result.Ok[_RecordedTokenEval] | _result.Rejected:
    assert choice_token_ids.ndim == 2
    assert choice_masks.shape == choice_token_ids.shape
    assert int(choice_token_ids.shape[0]) == int(
        argument_logits.shape[0]
    )
    assert selected_choice_offsets.shape == (
        int(argument_logits.shape[0]),
    )
    legal_logits = argument_logits.gather(
        dim=1, index=choice_token_ids.to(dtype=torch.long)
    )
    valid_logits = legal_logits[choice_masks]
    logits_check = reject_if_non_finite(
        (
            NamedTensorCheck(
                tensor=valid_logits,
                reason="policy argument logits must be finite",
            ),
        )
    )
    if isinstance(logits_check, _result.Rejected):
        return logits_check
    masked_logits = legal_logits.masked_fill(~choice_masks, -torch.inf)
    probabilities = torch.softmax(masked_logits, dim=1).masked_fill(
        ~choice_masks, 0.0
    )
    log_probabilities = torch.log_softmax(
        masked_logits, dim=1
    ).masked_fill(~choice_masks, 0.0)
    selected = log_probabilities.gather(
        dim=1, index=selected_choice_offsets.unsqueeze(1)
    ).squeeze(1)
    entropies = -(probabilities * log_probabilities).sum(dim=1)
    distribution_check = reject_if_non_finite(
        (
            NamedTensorCheck(
                tensor=probabilities,
                reason="policy argument distribution must be finite",
            ),
            NamedTensorCheck(
                tensor=log_probabilities,
                reason="policy argument distribution must be finite",
            ),
            NamedTensorCheck(
                tensor=selected,
                reason="policy argument distribution must be finite",
            ),
            NamedTensorCheck(
                tensor=entropies,
                reason="policy argument distribution must be finite",
            ),
        )
    )
    if isinstance(distribution_check, _result.Rejected):
        return distribution_check
    return _result.Ok(
        value=_RecordedTokenEval(
            log_probabilities=selected,
            entropies=entropies,
        )
    )
