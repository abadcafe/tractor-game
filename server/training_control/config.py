"""Server-owned defaults for the thin training control adapter."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class TrainingControlConfig(BaseModel):
    """Validated server configuration, independent of training state."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    default_run_dir: Path
    stop_timeout_seconds: float = Field(gt=0.0)

    def resolve_run_dir(self, supplied: Path | None) -> Path:
        """Resolve an optional request directory against the default."""
        directory = (
            self.default_run_dir if supplied is None else supplied
        )
        return directory.resolve()


def training_control_config() -> TrainingControlConfig:
    """Load the adapter defaults from the Server environment."""
    directory = Path(
        os.environ.get("TRAINING_RUN_DIR", "training_runs")
    ).resolve()
    timeout = float(
        os.environ.get("TRAINING_STOP_TIMEOUT_SECONDS", "300")
    )
    return TrainingControlConfig(
        default_run_dir=directory,
        stop_timeout_seconds=timeout,
    )
