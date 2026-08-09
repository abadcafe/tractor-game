"""Current-policy evaluation of recorded PPO traces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.policy_model.network import EncodedObservation, PolicyModel
from server.training.ppo.minibatch import TensorizedPPOMinibatch
from server.training.ppo.profile import PPOProfileAccumulator
from server.training.ppo.replay_tensors import PPOReplayTensorBatch
from server.training.ppo.validation import (
    PPO_TRACE_EVALUATION_FAILED,
    TensorValidationCheck,
    combine_validation_codes,
    non_finite_validation_code,
    validation_ok,
)


@dataclass(frozen=True, slots=True)
class TraceBatchEval:
    """Current-model scores for a minibatch of recorded traces."""

    log_probabilities: Tensor
    entropies: Tensor
    validation_code: Tensor


@dataclass(frozen=True, slots=True)
class ActionStepBatchEval:
    """Current-model scores for active action trace prefixes."""

    active_positions: Tensor
    log_probabilities: Tensor
    entropies: Tensor
    validation_code: Tensor


def evaluate_trace_batch(
    *,
    model: PolicyModel,
    minibatch: TensorizedPPOMinibatch,
    device: torch.device,
    profile: PPOProfileAccumulator,
) -> TraceBatchEval:
    """Evaluate recorded traces without synchronizing device checks."""
    assert not minibatch.is_empty()
    observation_batch = minibatch.observation_batch
    assert observation_batch is not None
    encode_start = profile.mark()
    policy_encoding = model.encode_observations(observation_batch)
    profile.record_elapsed(
        "policy_observation_encode_seconds",
        encode_start,
    )
    log_probability_sums = torch.zeros(
        (minibatch.local_count,), dtype=torch.float32, device=device
    )
    entropy_sums = torch.zeros(
        (minibatch.local_count,), dtype=torch.float32, device=device
    )
    replay = minibatch.replay
    assert replay is not None
    prefix_eval = _action_step_batch_eval(
        model=model,
        encoding=policy_encoding,
        replay=replay,
        profile=profile,
    )
    _ = log_probability_sums.index_add_(
        dim=0,
        index=prefix_eval.active_positions,
        source=prefix_eval.log_probabilities,
    )
    _ = entropy_sums.index_add_(
        dim=0,
        index=prefix_eval.active_positions,
        source=prefix_eval.entropies,
    )
    return TraceBatchEval(
        log_probabilities=log_probability_sums,
        entropies=entropy_sums,
        validation_code=prefix_eval.validation_code,
    )


def _action_step_batch_eval(
    *,
    model: PolicyModel,
    encoding: EncodedObservation,
    replay: PPOReplayTensorBatch,
    profile: PPOProfileAccumulator,
) -> ActionStepBatchEval:
    assert int(replay.step_counts.shape[0]) > 0
    profile.record_action_trace_lengths(replay.step_counts)
    decode_start = profile.mark()
    scores = model.score_action_traces(
        encoding,
        source_rows=torch.arange(
            int(replay.step_counts.shape[0]),
            dtype=torch.long,
            device=replay.step_counts.device,
        ),
        choice_ids_padded=replay.choice_ids_padded,
        step_counts=replay.step_counts,
    )
    profile.record_elapsed("action_decode_seconds", decode_start)
    active_positions = replay.active_sample_indices
    if replay.active_step_count == 0:
        empty = torch.empty(
            (0,), dtype=torch.float32, device=active_positions.device
        )
        return ActionStepBatchEval(
            active_positions=active_positions,
            log_probabilities=empty,
            entropies=empty,
            validation_code=validation_ok(active_positions.device),
        )
    distribution_start = profile.mark()
    distribution = _evaluate_recorded_choice_batch(
        choice_logits=scores.choice_logits[
            replay.active_sample_indices, replay.active_step_indices
        ],
        legal_choice_masks=replay.legal_choice_masks,
        selected_choice_ids=replay.choice_ids_padded[
            replay.active_sample_indices, replay.active_step_indices
        ],
    )
    profile.record_elapsed(
        "action_distribution_seconds", distribution_start
    )
    return ActionStepBatchEval(
        active_positions=active_positions,
        log_probabilities=distribution.log_probabilities,
        entropies=distribution.entropies,
        validation_code=distribution.validation_code,
    )


@dataclass(frozen=True, slots=True)
class _RecordedChoiceEval:
    log_probabilities: Tensor
    entropies: Tensor
    validation_code: Tensor


def _evaluate_recorded_choice_batch(
    *,
    choice_logits: Tensor,
    legal_choice_masks: Tensor,
    selected_choice_ids: Tensor,
) -> _RecordedChoiceEval:
    assert legal_choice_masks.shape == choice_logits.shape
    assert selected_choice_ids.shape == (int(choice_logits.shape[0]),)
    choice_logits = choice_logits.to(dtype=torch.float32)
    valid_logits = choice_logits[legal_choice_masks]
    logits_code = non_finite_validation_code(
        (
            TensorValidationCheck(
                tensor=valid_logits,
                code=PPO_TRACE_EVALUATION_FAILED,
            ),
        )
    )
    masked_logits = choice_logits.masked_fill(
        ~legal_choice_masks, -torch.inf
    )
    probabilities = torch.softmax(masked_logits, dim=1).masked_fill(
        ~legal_choice_masks, 0.0
    )
    log_probabilities = torch.log_softmax(
        masked_logits, dim=1
    ).masked_fill(~legal_choice_masks, 0.0)
    selected = log_probabilities.gather(
        dim=1, index=selected_choice_ids.unsqueeze(1)
    ).squeeze(1)
    entropies = -(probabilities * log_probabilities).sum(dim=1)
    distribution_code = non_finite_validation_code(
        (
            TensorValidationCheck(
                tensor=probabilities,
                code=PPO_TRACE_EVALUATION_FAILED,
            ),
            TensorValidationCheck(
                tensor=log_probabilities,
                code=PPO_TRACE_EVALUATION_FAILED,
            ),
            TensorValidationCheck(
                tensor=selected,
                code=PPO_TRACE_EVALUATION_FAILED,
            ),
            TensorValidationCheck(
                tensor=entropies,
                code=PPO_TRACE_EVALUATION_FAILED,
            ),
        )
    )
    return _RecordedChoiceEval(
        log_probabilities=selected,
        entropies=entropies,
        validation_code=combine_validation_codes(
            logits_code, distribution_code
        ),
    )
