"""Run the short standalone training initialization command."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from server.foundation import result as _result


class TrainingInitialization(BaseModel):
    """Filesystem result of a completed initialization command."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    run_dir: Path
    checkpoint_path: Path


class TrainingInitControl:
    """Execute init without creating or managing a PID file."""

    async def initialize(
        self,
        *,
        run_dir: Path,
        command: tuple[str, ...],
        working_directory: Path,
    ) -> _result.Ok[TrainingInitialization] | _result.Rejected:
        assert command
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=working_directory,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError:
            return _result.Rejected(
                reason="training initialization could not be started"
            )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            reason = stderr.decode("utf-8", errors="replace").strip()
            return _result.Rejected(
                reason=reason or "training initialization failed"
            )
        canonical_run_dir = run_dir.resolve()
        return _result.Ok(
            value=TrainingInitialization(
                run_dir=canonical_run_dir,
                checkpoint_path=(
                    canonical_run_dir / "checkpoints" / "latest.json"
                ),
            )
        )
