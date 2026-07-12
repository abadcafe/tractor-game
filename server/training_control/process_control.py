"""Status-script-style detached training process control."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from server.foundation import result as _result
from server.training_control.cli_client import (
    TrainingCliClient,
    TrainingProcess,
    same_process,
)
from server.training_control.pid_file import (
    read_pid,
    remove_pid_if_matches,
    remove_stale_pid,
    write_pid,
)


@dataclass(frozen=True, slots=True)
class StopResult:
    """Observable result of one idempotent stop request."""

    forced: bool


class TrainingProcessControl:
    """Start and stop one PID-file-managed detached CLI process."""

    def __init__(
        self,
        *,
        cli_client: TrainingCliClient | None = None,
        startup_timeout_seconds: float = 5.0,
    ) -> None:
        assert startup_timeout_seconds > 0.0
        self._cli_client = cli_client or TrainingCliClient()
        self._startup_timeout_seconds = startup_timeout_seconds
        self._lock = asyncio.Lock()

    async def inspect(
        self, run_dir: Path
    ) -> _result.Ok[TrainingProcess | None] | _result.Rejected:
        """Return the process identity produced by CLI summary."""
        summary = await self._cli_client.summary(run_dir)
        if isinstance(summary, _result.Rejected):
            return summary
        if summary.value.state == "BROKEN":
            assert summary.value.reason is not None
            return _result.Rejected(reason=summary.value.reason)
        return _result.Ok(value=summary.value.process)

    async def start(
        self,
        *,
        run_dir: Path,
        command: tuple[str, ...],
        working_directory: Path,
    ) -> _result.Ok[TrainingProcess] | _result.Rejected:
        """Start a detached command and publish its PID."""
        assert command
        canonical_run_dir = run_dir.resolve()
        async with self._lock:
            summary = await self._cli_client.summary(canonical_run_dir)
            if isinstance(summary, _result.Rejected):
                return summary
            if summary.value.state != "READY":
                return _result.Rejected(
                    reason=(
                        "training run is not ready: "
                        f"{summary.value.state}"
                    )
                )
            existing_process = summary.value.process
            if existing_process is not None:
                return _result.Rejected(
                    reason=(
                        "training process is already running: "
                        f"PID {existing_process.pid}"
                    )
                )
            stale_result = _remove_stale_file(canonical_run_dir)
            if isinstance(stale_result, _result.Rejected):
                return stale_result
            spawn_result = await asyncio.to_thread(
                _spawn,
                canonical_run_dir,
                command,
                working_directory.resolve(),
            )
            if isinstance(spawn_result, _result.Rejected):
                return spawn_result
            process = spawn_result.value
            pid_result = write_pid(canonical_run_dir, process.pid)
            if isinstance(pid_result, _result.Rejected):
                await asyncio.to_thread(_kill_spawned_process, process)
                return pid_result
            inspection = await self._wait_for_start(
                canonical_run_dir, process
            )
            if isinstance(inspection, _result.Rejected):
                await asyncio.to_thread(_kill_spawned_process, process)
                remove_pid_if_matches(canonical_run_dir, process.pid)
                return inspection
            _start_reaper(process, canonical_run_dir)
            return inspection

    async def stop(
        self,
        *,
        run_dir: Path,
        timeout_seconds: float,
    ) -> _result.Ok[StopResult] | _result.Rejected:
        """Request graceful stop, then kill a verified process group."""
        assert timeout_seconds > 0.0
        canonical_run_dir = run_dir.resolve()
        async with self._lock:
            inspected = await self.inspect(canonical_run_dir)
            if isinstance(inspected, _result.Rejected):
                return inspected
            original = inspected.value
            if original is None:
                stale_result = _remove_stale_file(canonical_run_dir)
                if isinstance(stale_result, _result.Rejected):
                    return stale_result
                return _result.Ok(value=StopResult(forced=False))
            try:
                os.kill(original.pid, signal.SIGTERM)
            except ProcessLookupError:
                remove_pid_if_matches(canonical_run_dir, original.pid)
                return _result.Ok(value=StopResult(forced=False))
            except OSError:
                return _result.Rejected(
                    reason=(
                        "SIGTERM failed for training PID "
                        f"{original.pid}"
                    )
                )
            exited = await self._wait_for_exit(
                canonical_run_dir,
                original,
                timeout_seconds=timeout_seconds,
            )
            if isinstance(exited, _result.Rejected):
                return exited
            forced = False
            if not exited.value:
                current_result = await self.inspect(canonical_run_dir)
                if isinstance(current_result, _result.Rejected):
                    return current_result
                current = current_result.value
                if current is None:
                    remove_pid_if_matches(
                        canonical_run_dir, original.pid
                    )
                    return _result.Ok(value=StopResult(forced=False))
                if not same_process(original, current):
                    return _result.Rejected(
                        reason=(
                            "training PID identity changed while "
                            "stopping; "
                            "no force signal was sent"
                        )
                    )
                if current.process_group_id != current.pid:
                    return _result.Rejected(
                        reason=(
                            "training process is not its process-group "
                            "leader; "
                            "refusing unsafe force stop"
                        )
                    )
                try:
                    os.killpg(current.process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    return _result.Rejected(
                        reason=(
                            "SIGKILL failed for training process group "
                            f"{current.process_group_id}"
                        )
                    )
                forced = True
                killed = await self._wait_for_exit(
                    canonical_run_dir,
                    original,
                    timeout_seconds=5.0,
                )
                if isinstance(killed, _result.Rejected):
                    return killed
                if not killed.value:
                    return _result.Rejected(
                        reason="training process group did not exit"
                    )
            removed = remove_pid_if_matches(
                canonical_run_dir, original.pid
            )
            if isinstance(removed, _result.Rejected):
                return removed
            return _result.Ok(value=StopResult(forced=forced))

    async def _wait_for_start(
        self,
        run_dir: Path,
        process: subprocess.Popen[bytes],
    ) -> _result.Ok[TrainingProcess] | _result.Rejected:
        deadline = time.monotonic() + self._startup_timeout_seconds
        last_rejection: _result.Rejected | None = None
        while time.monotonic() < deadline:
            inspected = await self.inspect(run_dir)
            if isinstance(inspected, _result.Rejected):
                last_rejection = inspected
            elif inspected.value is not None:
                return _result.Ok(value=inspected.value)
            if process.poll() is not None:
                break
            await asyncio.sleep(0.05)
        if last_rejection is not None:
            return last_rejection
        return _result.Rejected(
            reason=(
                f"training process failed to start: PID {process.pid}"
            )
        )

    async def _wait_for_exit(
        self,
        run_dir: Path,
        original: TrainingProcess,
        *,
        timeout_seconds: float,
    ) -> _result.Ok[bool] | _result.Rejected:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            inspected = await self.inspect(run_dir)
            if isinstance(inspected, _result.Rejected):
                return inspected
            current = inspected.value
            if current is None:
                return _result.Ok(value=True)
            if not same_process(original, current):
                return _result.Rejected(
                    reason="training PID identity changed while waiting"
                )
            await asyncio.sleep(0.1)
        return _result.Ok(value=False)


def _remove_stale_file(
    run_dir: Path,
) -> _result.Ok[None] | _result.Rejected:
    pid_result = read_pid(run_dir)
    if isinstance(pid_result, _result.Rejected):
        return pid_result
    if pid_result.value is None:
        return _result.Ok(value=None)
    return remove_stale_pid(run_dir, pid_result.value)


def _spawn(
    run_dir: Path,
    command: tuple[str, ...],
    working_directory: Path,
) -> _result.Ok[subprocess.Popen[bytes]] | _result.Rejected:
    try:
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return _result.Rejected(
            reason="training process could not be started"
        )
    return _result.Ok(value=process)


def _kill_spawned_process(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _start_reaper(
    process: subprocess.Popen[bytes], run_dir: Path
) -> None:
    thread = threading.Thread(
        target=_reap_process,
        args=(process, run_dir),
        name=f"training-reaper-{process.pid}",
        daemon=True,
    )
    thread.start()


def _reap_process(
    process: subprocess.Popen[bytes], run_dir: Path
) -> None:
    process.wait()
    remove_pid_if_matches(run_dir, process.pid)
