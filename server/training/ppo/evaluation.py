"""Current-policy evaluation of recorded PPO traces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server import result as _result
from server.training.argument_distribution import (
    batched_argument_distribution,
)
from server.training.model import (
    ObservationEncoding,
    TractorPolicyModel,
)
from server.training.ppo.profile import PPOProfileAccumulator
from server.training.ppo.rollout import RolloutSample
from server.training.semantic_actions.arguments import (
    SemanticArgumentPrefix,
)
from server.training.tensorize import (
    stack_observation_batches,
    tensorize_argument_prefixes,
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

    log_probabilities: Tensor
    entropies: Tensor


@dataclass(frozen=True, slots=True)
class _TraceAccumulator:
    """Per-sample tensors collected from batched prefix forwards."""

    log_probabilities: tuple[Tensor, ...]
    entropies: tuple[Tensor, ...]


def evaluate_trace_batch(
    *,
    model: TractorPolicyModel,
    samples: tuple[RolloutSample, ...],
    device: torch.device,
    profile: PPOProfileAccumulator,
) -> _result.Ok[TraceBatchEval] | _result.Rejected:
    """Evaluate recorded semantic traces under the current model."""
    assert samples
    observation_batch_start = profile.mark()
    observation_batch = stack_observation_batches(
        batches=tuple(
            sample.transition.decision.observation_batch
            for sample in samples
        ),
        device=device,
    )
    profile.record_elapsed(
        "observation_batch_seconds", observation_batch_start
    )
    empty_prefixes = tuple(
        SemanticArgumentPrefix(arguments=()) for _ in samples
    )
    encode_start = profile.mark()
    encoding = model.encode_observations(observation_batch)
    profile.record_elapsed("observation_encode_seconds", encode_start)
    value_start = profile.mark()
    values = model.value_estimates(encoding)
    profile.record_elapsed("value_head_seconds", value_start)
    accumulators = tuple(
        _TraceAccumulator(log_probabilities=(), entropies=())
        for _ in samples
    )
    prefixes = list(empty_prefixes)
    max_trace_length = max(
        len(sample.transition.decision.action.semantic_trace.arguments)
        for sample in samples
    )
    for argument_index in range(max_trace_length):
        active_indices = tuple(
            index
            for index, sample in enumerate(samples)
            if argument_index
            < len(
                sample.transition.decision.action.semantic_trace.arguments
            )
        )
        step_eval_result = _argument_batch_eval(
            model=model,
            samples=samples,
            active_indices=active_indices,
            encoding=encoding,
            prefixes=tuple(prefixes[index] for index in active_indices),
            argument_index=argument_index,
            profile=profile,
        )
        if isinstance(step_eval_result, _result.Rejected):
            return step_eval_result
        step_eval = step_eval_result.value
        for row_index, sample_index in enumerate(active_indices):
            accumulator = accumulators[sample_index]
            accumulators = _replace_accumulator(
                accumulators,
                index=sample_index,
                accumulator=_TraceAccumulator(
                    log_probabilities=(
                        *accumulator.log_probabilities,
                        step_eval.log_probabilities[row_index],
                    ),
                    entropies=(
                        *accumulator.entropies,
                        step_eval.entropies[row_index],
                    ),
                ),
            )
            argument = samples[
                sample_index
            ].transition.decision.action.semantic_trace.arguments[
                argument_index
            ]
            if argument.kind == "select_face_count":
                current_prefix = prefixes[sample_index]
                prefixes[sample_index] = SemanticArgumentPrefix(
                    arguments=(*current_prefix.arguments, argument)
                )
    return _result.Ok(
        value=TraceBatchEval(
            log_probabilities=torch.stack(
                [
                    _sum_tensors(
                        accumulator.log_probabilities,
                        device=device,
                    )
                    for accumulator in accumulators
                ]
            ),
            values=values,
            entropies=torch.stack(
                [
                    _sum_tensors(
                        accumulator.entropies,
                        device=device,
                    )
                    for accumulator in accumulators
                ]
            ),
        )
    )


def _argument_batch_eval(
    *,
    model: TractorPolicyModel,
    samples: tuple[RolloutSample, ...],
    active_indices: tuple[int, ...],
    encoding: ObservationEncoding,
    prefixes: tuple[SemanticArgumentPrefix, ...],
    argument_index: int,
    profile: PPOProfileAccumulator,
) -> _result.Ok[ArgumentBatchEval] | _result.Rejected:
    assert active_indices
    select_start = profile.mark()
    active_encoding = model.select_observation_encoding(
        encoding,
        active_indices=active_indices,
    )
    profile.record_elapsed("argument_select_seconds", select_start)
    profile.record_argument_prefixes(prefixes)
    prefix_tensorize_start = profile.mark()
    prefix_batch = tensorize_argument_prefixes(
        prefixes=prefixes,
        device=encoding.memory.device,
    )
    profile.record_elapsed(
        "argument_prefix_tensorize_seconds",
        prefix_tensorize_start,
    )
    decode_start = profile.mark()
    scores = model.score_argument_prefixes(
        active_encoding,
        prefix=prefix_batch,
    )
    profile.record_elapsed("argument_decode_seconds", decode_start)
    distribution_start = profile.mark()
    distribution_result = batched_argument_distribution(
        argument_logits=scores.argument_logits,
        choice_steps=tuple(
            samples[
                sample_index
            ].transition.decision.choice_trace.steps[argument_index]
            for sample_index in active_indices
        ),
    )
    if isinstance(distribution_result, _result.Rejected):
        return distribution_result
    distribution = distribution_result.value
    profile.record_elapsed(
        "argument_distribution_seconds", distribution_start
    )
    return _result.Ok(
        value=ArgumentBatchEval(
            log_probabilities=distribution.selected_log_probabilities,
            entropies=distribution.entropies,
        )
    )


def _sum_tensors(
    values: tuple[Tensor, ...], *, device: torch.device
) -> Tensor:
    if not values:
        return torch.tensor(0.0, dtype=torch.float32, device=device)
    return torch.stack(list(values)).sum()


def _replace_accumulator(
    accumulators: tuple[_TraceAccumulator, ...],
    *,
    index: int,
    accumulator: _TraceAccumulator,
) -> tuple[_TraceAccumulator, ...]:
    result = list(accumulators)
    result[index] = accumulator
    return tuple(result)
