"""PID-file lifecycle control for standalone training CLI processes."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from collections.abc import AsyncGenerator
from contextlib import suppress
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from server.foundation import result as _result
from server.training_control.process_inspection import (
    ProcessInspector,
    ProcessState,
    remove_training_pid,
    remove_training_pid_if_matches,
    write_training_pid,
)

_LOGGER = logging.getLogger(__name__)
_PROCESS_POLL_SECONDS = 1.0
_PROCESS_EXIT_POLL_SECONDS = 0.1
_FORCED_EXIT_TIMEOUT_SECONDS = 5.0
_PROCESS_LOG_NAME = "training-cli.log"


class StopResult(BaseModel):
    """Outcome of a PID-file stop request."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    forced: bool


class TrainingInitialization(BaseModel):
    """Filesystem result of a completed initialization command."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    run_dir: Path
    checkpoint_path: Path


class TrainingProcessControl:
    """Control initialization and resumed training with a PID file."""

    def __init__(self) -> None:
        self._inspector = ProcessInspector()
        self._locks: dict[Path, asyncio.Lock] = {}
        self._reapers: set[asyncio.Task[None]] = set()

    async def inspect(
        self, run_dir: Path
    ) -> _result.Ok[ProcessState] | _result.Rejected:
        return self._inspector.inspect(run_dir.resolve())

    async def initialize(
        self,
        *,
        run_dir: Path,
        command: tuple[str, ...],
        working_directory: Path,
    ) -> _result.Ok[TrainingInitialization] | _result.Rejected:
        """Execute CLI init synchronously without publishing a PID."""
        assert command
        canonical = run_dir.resolve()
        async with self._lock(canonical):
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=working_directory.resolve(),
                    stdin=subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except OSError as error:
                return _result.Rejected(
                    reason=(
                        "training initialization could not be started: "
                        f"{error}"
                    )
                )
            _stdout, stderr = await process.communicate()
            returncode = process.returncode
            assert returncode is not None
            if returncode != 0:
                error = stderr.decode("utf-8", errors="replace").strip()
                return _result.Rejected(
                    reason=error or "training initialization failed"
                )
        return _result.Ok(
            value=TrainingInitialization(
                run_dir=canonical,
                checkpoint_path=canonical
                / "checkpoints"
                / "latest.json",
            )
        )

    async def resume(
        self,
        *,
        run_dir: Path,
        command: tuple[str, ...],
        working_directory: Path,
    ) -> _result.Ok[None] | _result.Rejected:
        """Spawn one long-running CLI process and publish its PID."""
        assert command
        canonical = run_dir.resolve()
        async with self._lock(canonical):
            current = self._inspector.inspect(canonical)
            if isinstance(current, _result.Rejected):
                return current
            process_snapshot = current.value.process
            if process_snapshot is not None:
                return _result.Rejected(
                    reason=(
                        "training process is already running: PID "
                        f"{process_snapshot.pid}"
                    )
                )
            directory_result = _validate_resume_directory(canonical)
            if isinstance(directory_result, _result.Rejected):
                return directory_result
            log_path = canonical / _PROCESS_LOG_NAME
            try:
                output = log_path.open("wb", buffering=0)
            except OSError:
                return _result.Rejected(
                    reason=f"training log is unwritable: {log_path}"
                )
            try:
                try:
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        cwd=working_directory.resolve(),
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except OSError as error:
                    return _result.Rejected(
                        reason=(
                            "training process could not be started: "
                            f"{error}"
                        )
                    )
            finally:
                output.close()
            written = write_training_pid(canonical, process.pid)
            if isinstance(written, _result.Rejected):
                await _kill_spawned_process_group(process)
                return written
            self._start_reaper(process, canonical)
        return _result.Ok(value=None)

    async def stop(
        self,
        *,
        run_dir: Path,
        timeout_seconds: float,
    ) -> _result.Ok[StopResult] | _result.Rejected:
        """Request graceful shutdown, then force the process group."""
        assert timeout_seconds > 0.0
        canonical = run_dir.resolve()
        async with self._lock(canonical):
            current = self._inspector.inspect(canonical)
            if isinstance(current, _result.Rejected):
                return current
            process = current.value.process
            if process is None:
                removed = remove_training_pid(canonical)
                if isinstance(removed, _result.Rejected):
                    return removed
                return _result.Ok(value=StopResult(forced=False))
            pid = process.pid
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                removed = remove_training_pid_if_matches(canonical, pid)
                if isinstance(removed, _result.Rejected):
                    return removed
                return _result.Ok(value=StopResult(forced=False))
            except OSError:
                return _result.Rejected(
                    reason=f"SIGTERM failed for training PID {pid}"
                )
            exited = await _wait_for_process_group_exit(
                pid, timeout_seconds=timeout_seconds
            )
            if isinstance(exited, _result.Rejected):
                return exited
            forced = False
            if not exited.value:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    return _result.Rejected(
                        reason=(
                            "SIGKILL failed for training process group "
                            f"{pid}"
                        )
                    )
                else:
                    forced = True
                killed = await _wait_for_process_group_exit(
                    pid,
                    timeout_seconds=_FORCED_EXIT_TIMEOUT_SECONDS,
                )
                if isinstance(killed, _result.Rejected):
                    return killed
                if not killed.value:
                    return _result.Rejected(
                        reason="training process group did not exit"
                    )
            removed = remove_training_pid_if_matches(canonical, pid)
            if isinstance(removed, _result.Rejected):
                return removed
            return _result.Ok(value=StopResult(forced=forced))

    async def watch(
        self, run_dir: Path
    ) -> AsyncGenerator[
        _result.Ok[ProcessState] | _result.Rejected, None
    ]:
        """Yield initial state and subsequent PID-state changes."""
        canonical = run_dir.resolve()
        previous: ProcessState | None = None
        observed = False
        while True:
            result = self._inspector.inspect(canonical)
            if isinstance(result, _result.Rejected):
                yield result
                return
            if not observed or result.value != previous:
                previous = result.value
                observed = True
                yield result
            await asyncio.sleep(_PROCESS_POLL_SECONDS)

    async def close(self) -> None:
        """Stop local reapers without stopping resumed training."""
        tasks = tuple(self._reapers)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task

    def _lock(self, run_dir: Path) -> asyncio.Lock:
        return self._locks.setdefault(run_dir, asyncio.Lock())

    def _start_reaper(
        self, process: asyncio.subprocess.Process, run_dir: Path
    ) -> None:
        task = asyncio.create_task(self._reap(process, run_dir))
        self._reapers.add(task)
        task.add_done_callback(self._reapers.discard)

    async def _reap(
        self, process: asyncio.subprocess.Process, run_dir: Path
    ) -> None:
        await process.wait()
        async with self._lock(run_dir):
            removed = remove_training_pid_if_matches(
                run_dir, process.pid
            )
        if isinstance(removed, _result.Rejected):
            _LOGGER.error(
                "Training PID cleanup failed: %s", removed.reason
            )


def _validate_resume_directory(
    run_dir: Path,
) -> _result.Ok[None] | _result.Rejected:
    try:
        if not run_dir.is_dir():
            return _result.Rejected(
                reason=(
                    f"training run directory does not exist: {run_dir}"
                )
            )
    except OSError:
        return _result.Rejected(
            reason=f"training run directory is unreadable: {run_dir}"
        )
    return _result.Ok(value=None)


async def _kill_spawned_process_group(
    process: asyncio.subprocess.Process,
) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


async def _wait_for_process_group_exit(
    process_group_id: int, *, timeout_seconds: float
) -> _result.Ok[bool] | _result.Rejected:
    assert process_group_id > 0
    assert timeout_seconds > 0.0
    deadline = time.monotonic() + timeout_seconds
    while True:
        exists = _process_group_exists(process_group_id)
        if isinstance(exists, _result.Rejected):
            return exists
        if not exists.value:
            return _result.Ok(value=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return _result.Ok(value=False)
        await asyncio.sleep(min(_PROCESS_EXIT_POLL_SECONDS, remaining))


def _process_group_exists(
    process_group_id: int,
) -> _result.Ok[bool] | _result.Rejected:
    assert process_group_id > 0
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return _result.Ok(value=False)
    except PermissionError:
        return _result.Ok(value=True)
    except OSError:
        return _result.Rejected(
            reason=(
                "training process-group existence check failed: "
                f"{process_group_id}"
            )
        )
    return _result.Ok(value=True)
