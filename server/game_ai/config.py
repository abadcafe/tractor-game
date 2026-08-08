"""Closed local-or-remote deployment configuration for model AI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

type AIDevice = Literal["cpu", "cuda", "mps"]


class LocalAIConfig(BaseModel):
    """Checkpoint and execution device for local policy inference."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    mode: Literal["local"] = "local"
    checkpoint_path: Path
    device: AIDevice = "cpu"


class RemoteAIConfig(BaseModel):
    """Whole-controller endpoint hosted with accelerated inference."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    mode: Literal["remote"] = "remote"
    endpoint: HttpUrl
    request_timeout_seconds: float = Field(default=120.0, gt=0.0)


type AIConfig = Annotated[
    LocalAIConfig | RemoteAIConfig,
    Field(discriminator="mode"),
]


def ai_config_from_env() -> AIConfig:
    """Read exactly one explicit deployment shape."""
    mode = os.environ.get("TRACTOR_AI_MODE", "local").strip().lower()
    if mode == "remote":
        endpoint = os.environ.get("TRACTOR_AI_ENDPOINT")
        if endpoint is None:
            raise ValueError(
                "TRACTOR_AI_ENDPOINT is required in remote mode"
            )
        return RemoteAIConfig(
            endpoint=HttpUrl(endpoint),
            request_timeout_seconds=_positive_float_env(
                "TRACTOR_AI_REQUEST_TIMEOUT",
                120.0,
            ),
        )
    if mode != "local":
        raise ValueError("TRACTOR_AI_MODE must be local or remote")
    run_dir = Path(
        os.environ.get("TRAINING_RUN_DIR", "training_runs")
    ).resolve()
    return LocalAIConfig(
        checkpoint_path=Path(
            os.environ.get(
                "TRACTOR_AI_CHECKPOINT",
                str(run_dir / "checkpoints" / "latest.json"),
            )
        ).resolve(),
        device=_device(os.environ.get("TRACTOR_AI_DEVICE", "cpu")),
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


def _positive_float_env(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


__all__ = (
    "AIConfig",
    "AIDevice",
    "LocalAIConfig",
    "RemoteAIConfig",
    "ai_config_from_env",
)
