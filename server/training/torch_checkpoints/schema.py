"""Shared schema records for torch training checkpoints."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from server import result as _result
from server.training.config import ModelConfig, TrainConfig

CHECKPOINT_SCHEMA_VERSION = 15
CHECKPOINT_OBJECTS_DIR = "objects"
CHECKPOINT_STATE_FILENAME = "state.pt"


@dataclass(frozen=True, slots=True)
class TorchCheckpointMetadata:
    """Portable checkpoint metadata needed before model creation."""

    model_config: ModelConfig
    train_config: TrainConfig
    total_rounds: int
    total_updates: int

    def __post_init__(self) -> None:
        assert self.total_rounds >= 0
        assert self.total_updates >= 0


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    """Manifest record pointing at one immutable state payload."""

    checkpoint_id: str
    state_path: Path
    state_sha256: str
    metadata: TorchCheckpointMetadata


def checkpoint_corruption(path: Path, reason: str) -> _result.Rejected:
    """Build a rejected checkpoint-corruption result."""
    return _result.Rejected(
        reason=f"checkpoint corruption: {path}: {reason}"
    )


def sha256_file(path: Path) -> str:
    """Return the sha256 hex digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_checkpoint_file(
    path: Path,
) -> _result.Ok[str] | _result.Rejected:
    """Return a checkpoint file digest or a corruption rejection."""
    try:
        return _result.Ok(value=sha256_file(path))
    except FileNotFoundError:
        return checkpoint_corruption(path, "state file is missing")
    except OSError:
        return checkpoint_corruption(path, "state file is not readable")
