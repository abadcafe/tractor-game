"""Runtime execution result types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainingLoopResult:
    """Completed training progress and final checkpoint manifest."""

    total_rounds: int
    total_samples: int
    total_updates: int
    checkpoint_path: Path
