"""Validated process configuration for model-backed players."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .search import SearchConfig

type AIDevice = Literal["cpu", "cuda", "mps"]


class AIConfig(BaseModel):
    """Checkpoint and compute budget shared by all AI players."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    checkpoint_path: Path
    device: AIDevice = "cpu"
    particle_count: int = Field(default=32, gt=0)
    direct_samples: int = Field(default=24, gt=0)
    candidate_samples: int = Field(default=24, gt=0)
    rollouts_per_particle: int = Field(default=2, gt=0)
    rollout_depth: int = Field(default=24, gt=0)

    def search_config(self) -> SearchConfig:
        """Return the rollout-only subset."""
        return SearchConfig(
            candidate_samples=self.candidate_samples,
            rollouts_per_particle=self.rollouts_per_particle,
            rollout_depth=self.rollout_depth,
        )

    @classmethod
    def from_env(cls) -> AIConfig:
        """Read one closed configuration from process environment."""
        run_dir = Path(
            os.environ.get("TRAINING_RUN_DIR", "training_runs")
        ).resolve()
        return cls(
            checkpoint_path=Path(
                os.environ.get(
                    "TRACTOR_AI_CHECKPOINT",
                    str(run_dir / "checkpoints" / "latest.json"),
                )
            ).resolve(),
            device=_device(os.environ.get("TRACTOR_AI_DEVICE", "cpu")),
            particle_count=_positive_env(
                "TRACTOR_AI_PARTICLES",
                32,
            ),
            direct_samples=_positive_env(
                "TRACTOR_AI_DIRECT_SAMPLES",
                24,
            ),
            candidate_samples=_positive_env(
                "TRACTOR_AI_CANDIDATE_SAMPLES",
                24,
            ),
            rollouts_per_particle=_positive_env(
                "TRACTOR_AI_ROLLOUTS_PER_PARTICLE",
                2,
            ),
            rollout_depth=_positive_env(
                "TRACTOR_AI_ROLLOUT_DEPTH",
                24,
            ),
        )


def _device(value: str) -> AIDevice:
    normalized = value.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        return "cuda"
    if normalized == "mps":
        return "mps"
    raise ValueError("TRACTOR_AI_DEVICE must be cpu, cuda, or mps")


def _positive_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


__all__ = ("AIConfig", "AIDevice")
