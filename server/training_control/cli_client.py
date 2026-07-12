"""Strict client for external training CLI summary commands."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from server.foundation import result as _result
from server.foundation.json_value import JsonObject

type TrainingRunState = Literal[
    "NOT_INITIALIZED", "BROKEN", "READY", "RUNNING"
]


class TrainingProcess(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    pid: int = Field(gt=0)
    name: str
    kernel_state: str
    executable: Path
    working_directory: Path
    run_dir: Path | None
    argv: tuple[str, ...]
    process_group_id: int = Field(gt=0)
    session_id: int = Field(gt=0)
    start_ticks: int = Field(ge=0)


class TrainingRunDetails(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    checkpoint_id: str
    checkpoint_path: Path
    state_size_bytes: int = Field(ge=0)
    model_config_values: JsonObject
    train_config_values: JsonObject
    total_rounds: int = Field(ge=0)
    total_samples: int = Field(ge=0)
    total_updates: int = Field(ge=0)


class CheckpointManifestRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str
    kind: Literal["latest", "archive", "invalid"]
    valid: bool
    error: str | None
    checkpoint_id: str | None
    state_path: str | None
    state_exists: bool
    state_size_bytes: int | None = Field(ge=0)
    modified_at_ms: int | None = Field(ge=0)
    state_modified_at_ms: int | None = Field(ge=0)
    state_sha256: str | None
    total_rounds: int | None = Field(ge=0)
    total_samples: int | None = Field(ge=0)
    total_updates: int | None = Field(ge=0)
    model_config_values: JsonObject | None
    train_config_values: JsonObject | None


class CheckpointObjectRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    checkpoint_id: str
    state_path: str
    valid: bool
    error: str | None
    state_size_bytes: int | None = Field(ge=0)
    state_modified_at_ms: int | None = Field(ge=0)
    referenced_by: tuple[str, ...]
    orphan: bool


class CheckpointCatalogRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    checkpoint_directory: Path
    manifests: tuple[CheckpointManifestRecord, ...]
    objects: tuple[CheckpointObjectRecord, ...]
    total_unique_state_bytes: int = Field(ge=0)


class TrainingCliSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[2]
    run_dir: Path
    state: TrainingRunState
    reason: str | None
    process: TrainingProcess | None
    details: TrainingRunDetails | None
    checkpoints: CheckpointCatalogRecord

    @model_validator(mode="after")
    def validate_state_payload(self) -> Self:
        if self.state == "RUNNING":
            assert self.process is not None
            assert self.reason is None and self.details is not None
        elif self.state == "READY":
            assert self.details is not None
            assert self.reason is None and self.process is None
        elif self.state == "BROKEN":
            assert self.reason is not None
            assert self.process is None and self.details is None
        else:
            assert self.reason is None
            assert self.process is None and self.details is None
        return self


class TrainingCliClient:
    """Execute summary without importing training producer modules."""

    def __init__(self, *, timeout_seconds: float = 120.0) -> None:
        assert timeout_seconds > 0.0
        self._timeout_seconds = timeout_seconds

    async def summary(
        self, run_dir: Path
    ) -> _result.Ok[TrainingCliSummary] | _result.Rejected:
        argv = (
            sys.executable,
            "-m",
            "server.training_cli",
            "--run-dir",
            str(run_dir.resolve()),
            "summary",
            "--format",
            "json",
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError:
            return _result.Rejected(
                reason="training summary command could not be started"
            )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self._timeout_seconds
            )
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        except TimeoutError:
            process.kill()
            await process.wait()
            return _result.Rejected(
                reason="training summary command timed out"
            )
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            return _result.Rejected(
                reason=detail or "training summary command failed"
            )
        try:
            summary = TrainingCliSummary.model_validate_json(stdout)
        except ValidationError:
            return _result.Rejected(
                reason="training summary command returned invalid JSON"
            )
        return _result.Ok(value=summary)


def same_process(left: TrainingProcess, right: TrainingProcess) -> bool:
    return (
        left.pid == right.pid and left.start_ticks == right.start_ticks
    )
