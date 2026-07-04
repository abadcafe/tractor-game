"""PPO update result records."""

from __future__ import annotations

import math
from dataclasses import dataclass

from server.training.ppo.profile import (
    PPOUpdateProfile,
    ppo_update_profile_is_finite,
)


@dataclass(frozen=True, slots=True)
class PPOUpdateStats:
    """Scalar loss stats for metrics."""

    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approx_kl: float
    clip_fraction: float
    profile: PPOUpdateProfile


def ppo_update_stats_are_finite(stats: PPOUpdateStats) -> bool:
    """Return whether all scalar PPO diagnostics are finite."""
    return (
        math.isfinite(stats.policy_loss)
        and math.isfinite(stats.value_loss)
        and math.isfinite(stats.entropy)
        and math.isfinite(stats.total_loss)
        and math.isfinite(stats.approx_kl)
        and math.isfinite(stats.clip_fraction)
        and ppo_update_profile_is_finite(stats.profile)
    )
