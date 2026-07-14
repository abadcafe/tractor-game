"""Server-owned defaults for the thin training control adapter."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class TrainingControlConfig(BaseModel):
    """Validated server configuration, independent of training state."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    default_run_dir: Path
    control_runtime_dir: Path
    stop_timeout_seconds: float = Field(gt=0.0)
    startup_timeout_seconds: float = Field(default=1800.0, gt=0.0)

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
        os.environ.get("TRAINING_STOP_TIMEOUT_SECONDS", "1800")
    )
    startup_timeout = float(
        os.environ.get("TRAINING_STARTUP_TIMEOUT_SECONDS", "1800")
    )
    configured_runtime = os.environ.get("TRAINING_CONTROL_RUNTIME_DIR")
    if configured_runtime is not None:
        runtime_dir = Path(configured_runtime).resolve()
    else:
        xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
        runtime_base = (
            Path(xdg_runtime)
            if xdg_runtime is not None
            else Path("/tmp") / f"tractor-game-{os.getuid()}"
        )
        runtime_dir = (runtime_base / "training").resolve()
    return TrainingControlConfig(
        default_run_dir=directory,
        control_runtime_dir=runtime_dir,
        stop_timeout_seconds=timeout,
        startup_timeout_seconds=startup_timeout,
    )
