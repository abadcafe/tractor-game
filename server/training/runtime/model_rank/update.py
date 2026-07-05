"""Model-rank update result data."""

from __future__ import annotations

from dataclasses import dataclass

from server.training.ppo import PPOUpdateStats
from server.training.runtime.state import RuntimeTrainingState


@dataclass(frozen=True, slots=True)
class ModelUpdateResult:
    """Result of one model-rank update."""

    update_stats: PPOUpdateStats
    state: RuntimeTrainingState
