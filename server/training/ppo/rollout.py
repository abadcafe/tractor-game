"""PPO rollout sample construction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from server.training.ppo.math import (
    ValueStep,
    generalized_advantage_targets,
)
from server.training.trajectory import (
    DecisionTransition,
    RolloutBatch,
    TeamTrajectory,
)


@dataclass(frozen=True, slots=True)
class RolloutSample:
    """One decision annotated with PPO targets."""

    transition: DecisionTransition
    advantage: float
    return_value: float
    old_log_probability: float
    old_value_estimate: float


def rollout_samples(
    batch: RolloutBatch,
    *,
    gae_lambda: float,
) -> tuple[RolloutSample, ...]:
    """Convert collected team trajectories into PPO samples."""
    samples: list[RolloutSample] = []
    for trajectory in batch.trajectories:
        samples.extend(
            _team_samples(
                trajectory,
                gae_lambda=gae_lambda,
            )
        )
    return tuple(samples)


def normalize_advantages(
    samples: tuple[RolloutSample, ...],
) -> tuple[RolloutSample, ...]:
    """Normalize advantages within one PPO update batch."""
    assert samples
    mean = sum(sample.advantage for sample in samples) / len(samples)
    variance = sum(
        (sample.advantage - mean) * (sample.advantage - mean)
        for sample in samples
    ) / len(samples)
    stddev = math.sqrt(variance)
    if stddev <= 0.000001:
        return tuple(
            RolloutSample(
                transition=sample.transition,
                advantage=sample.advantage - mean,
                return_value=sample.return_value,
                old_log_probability=sample.old_log_probability,
                old_value_estimate=sample.old_value_estimate,
            )
            for sample in samples
        )
    return tuple(
        RolloutSample(
            transition=sample.transition,
            advantage=(sample.advantage - mean) / (stddev + 0.000001),
            return_value=sample.return_value,
            old_log_probability=sample.old_log_probability,
            old_value_estimate=sample.old_value_estimate,
        )
        for sample in samples
    )


def minibatches(
    samples: tuple[RolloutSample, ...],
    *,
    minibatch_size: int,
) -> tuple[tuple[RolloutSample, ...], ...]:
    """Split PPO samples into fixed-size minibatches."""
    assert minibatch_size > 0
    result: list[tuple[RolloutSample, ...]] = []
    for start in range(0, len(samples), minibatch_size):
        result.append(samples[start : start + minibatch_size])
    return tuple(result)


def shuffled_samples(
    samples: tuple[RolloutSample, ...],
) -> tuple[RolloutSample, ...]:
    """Return a torch-random permutation of PPO samples."""
    order = torch.randperm(len(samples))
    result: list[RolloutSample] = []
    for position in range(order.numel()):
        index = int(order[position].item())
        result.append(samples[index])
    return tuple(result)


def _team_samples(
    trajectory: TeamTrajectory,
    *,
    gae_lambda: float,
) -> tuple[RolloutSample, ...]:
    transitions = trajectory.transitions
    assert transitions
    value_steps = tuple(
        ValueStep(
            reward=transition.reward_after_step,
            value_estimate=transition.decision.value_estimate,
        )
        for transition in transitions
    )
    targets = generalized_advantage_targets(
        steps=value_steps,
        terminal_reward=trajectory.terminal_reward,
        gae_lambda=gae_lambda,
    )
    return tuple(
        RolloutSample(
            transition=transition,
            advantage=targets[index].advantage,
            return_value=targets[index].return_value,
            old_log_probability=transition.decision.log_probability,
            old_value_estimate=transition.decision.value_estimate,
        )
        for index, transition in enumerate(transitions)
    )
