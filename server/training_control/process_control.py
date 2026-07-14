"""Unified lifecycle control for standalone training CLI processes."""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import signal
import subprocess
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from server.foundation import result as _result
from server.training_control.config import training_control_config
from server.training_control.process_inspection import (
    ProcessInspector,
    ProcessSnapshot,
    ProcSnapshot,
    same_process,
)
from server.training_control.process_launch import (
    open_process_handle,
    read_control_message,
    signal_process_handle,
    spawn_process,
    terminate_process_group,
)
from server.training_control.process_owner import (
    ProcessOwner,
    TrainingCommand,
    increment_revision,
    lock_path,
    mark_owner_ready,
    prepare_control_directory,
    read_owner,
    read_revision,
    remove_owner_if_matches,
    write_owner,
)

_LOGGER = logging.getLogger(__name__)


class ProcessEnvelope(BaseModel):
    """Revisioned process state for control responses and streams."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    revision: int = Field(ge=0)
    process: ProcessSnapshot | None


class StopResult(BaseModel):
    """Terminal state returned by an idempotent stop request."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    forced: bool
    revision: int = Field(ge=0)
    process: None = None


class TrainingInitialization(BaseModel):
    """Filesystem result of a completed initialization command."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    run_dir: Path
    checkpoint_path: Path


class TrainingProcessControl:
    """Own initialize, resume, inspect, stop, and lifecycle watches."""

    def __init__(
        self,
        *,
        runtime_root: Path | None = None,
        proc_root: Path = Path("/proc"),
        startup_timeout_seconds: float | None = None,
    ) -> None:
        config = training_control_config()
        self._runtime_root = (
            config.control_runtime_dir
            if runtime_root is None
            else runtime_root.resolve()
        )
        self._inspector = ProcessInspector(
            runtime_root=self._runtime_root,
            proc_root=proc_root,
        )
        self._startup_timeout_seconds = (
            config.startup_timeout_seconds
            if startup_timeout_seconds is None
            else startup_timeout_seconds
        )
        assert self._startup_timeout_seconds > 0.0
        self._locks: dict[Path, asyncio.Lock] = {}
        self._conditions: dict[Path, asyncio.Condition] = {}
        self._reapers: set[asyncio.Task[None]] = set()

    async def inspect(
        self, run_dir: Path
    ) -> _result.Ok[ProcessEnvelope] | _result.Rejected:
        """Return direct /proc state and clean an exact stale owner."""
        canonical = run_dir.resolve()
        owner = read_owner(self._runtime_root, canonical)
        if isinstance(owner, _result.Rejected):
            return owner
        if owner.value is None:
            revision = read_revision(self._runtime_root, canonical)
            if isinstance(revision, _result.Rejected):
                return revision
            return _result.Ok(
                value=ProcessEnvelope(
                    revision=revision.value, process=None
                )
            )
        async with self._mutation_lock(canonical) as lock_result:
            if isinstance(lock_result, _result.Rejected):
                return lock_result
            return await self._inspect_locked(canonical)

    async def initialize(
        self,
        *,
        run_dir: Path,
        command: tuple[str, ...],
        working_directory: Path,
    ) -> _result.Ok[TrainingInitialization] | _result.Rejected:
        """Run initialization as a visible, managed process."""
        spawn_result = await self._start_process(
            run_dir=run_dir,
            command=command,
            working_directory=working_directory,
            command_kind="initialize",
            ready_write_fd=None,
            capture_output=True,
        )
        if isinstance(spawn_result, _result.Rejected):
            return spawn_result
        process, snapshot = spawn_result.value
        try:
            _stdout, stderr = await asyncio.to_thread(
                process.communicate
            )
        except asyncio.CancelledError:
            await asyncio.to_thread(
                terminate_process_group, process.pid
            )
            await asyncio.to_thread(process.wait)
            if snapshot is not None:
                cleanup = await self._finalize_owner(
                    run_dir.resolve(), snapshot
                )
                _log_cleanup_failure(cleanup)
            raise
        if snapshot is not None:
            cleanup = await self._finalize_owner(
                run_dir.resolve(), snapshot
            )
            if isinstance(cleanup, _result.Rejected):
                return cleanup
        if process.returncode != 0:
            error = stderr.decode("utf-8", errors="replace").strip()
            return _result.Rejected(
                reason=error or "training initialization failed"
            )
        canonical = run_dir.resolve()
        return _result.Ok(
            value=TrainingInitialization(
                run_dir=canonical,
                checkpoint_path=(
                    canonical / "checkpoints" / "latest.json"
                ),
            )
        )

    async def resume(
        self,
        *,
        run_dir: Path,
        command: tuple[str, ...],
        working_directory: Path,
    ) -> _result.Ok[ProcessEnvelope] | _result.Rejected:
        """Start training and wait until the CLI reports ready."""
        ready_read_fd, ready_write_fd = os.pipe()
        os.set_inheritable(ready_write_fd, True)
        controlled_command = (
            *command,
            "--ready-fd",
            str(ready_write_fd),
        )
        try:
            spawn_result = await self._start_process(
                run_dir=run_dir,
                command=controlled_command,
                working_directory=working_directory,
                command_kind="resume",
                ready_write_fd=ready_write_fd,
                capture_output=False,
            )
        finally:
            os.close(ready_write_fd)
        if isinstance(spawn_result, _result.Rejected):
            os.close(ready_read_fd)
            return spawn_result
        process, snapshot = spawn_result.value
        try:
            message_result = await asyncio.wait_for(
                asyncio.to_thread(read_control_message, ready_read_fd),
                timeout=self._startup_timeout_seconds,
            )
        except asyncio.CancelledError:
            await asyncio.to_thread(
                terminate_process_group, process.pid
            )
            await asyncio.to_thread(process.wait)
            if snapshot is not None:
                cleanup = await self._finalize_owner(
                    run_dir.resolve(), snapshot
                )
                _log_cleanup_failure(cleanup)
            raise
        except TimeoutError:
            message_result = _result.Rejected(
                reason="training readiness handshake timed out"
            )
        finally:
            os.close(ready_read_fd)
        if isinstance(message_result, _result.Rejected):
            await asyncio.to_thread(
                terminate_process_group, process.pid
            )
            await asyncio.to_thread(process.wait)
            if snapshot is not None:
                cleanup = await self._finalize_owner(
                    run_dir.resolve(), snapshot
                )
                _log_cleanup_failure(cleanup)
            return message_result
        if snapshot is None:
            await asyncio.to_thread(process.wait)
            return _result.Rejected(
                reason=(
                    "training process reported readiness after exiting"
                )
            )
        async with self._mutation_lock(
            run_dir.resolve()
        ) as lock_result:
            if isinstance(lock_result, _result.Rejected):
                return lock_result
            ready_result = mark_owner_ready(
                self._runtime_root,
                run_dir.resolve(),
                pid=snapshot.pid,
                start_ticks=snapshot.start_ticks,
            )
            if isinstance(ready_result, _result.Rejected):
                await asyncio.to_thread(
                    terminate_process_group, process.pid
                )
                await asyncio.to_thread(process.wait)
                removed = remove_owner_if_matches(
                    self._runtime_root,
                    run_dir.resolve(),
                    pid=snapshot.pid,
                    start_ticks=snapshot.start_ticks,
                )
                if isinstance(removed, _result.Rejected):
                    _log_cleanup_failure(removed)
                return ready_result
            publish_result = await self._publish(run_dir.resolve())
            if isinstance(publish_result, _result.Rejected):
                await asyncio.to_thread(
                    terminate_process_group, process.pid
                )
                await asyncio.to_thread(process.wait)
                removed = remove_owner_if_matches(
                    self._runtime_root,
                    run_dir.resolve(),
                    pid=snapshot.pid,
                    start_ticks=snapshot.start_ticks,
                )
                if isinstance(removed, _result.Rejected):
                    _log_cleanup_failure(removed)
                return publish_result
        self._start_reaper(process, run_dir.resolve(), snapshot)
        current = await self.inspect(run_dir)
        if isinstance(current, _result.Rejected):
            return current
        if current.value.process is None:
            return _result.Rejected(
                reason=(
                    "training process exited after readiness handshake"
                )
            )
        return current

    async def stop(
        self,
        *,
        run_dir: Path,
        timeout_seconds: float,
    ) -> _result.Ok[StopResult] | _result.Rejected:
        """Stop gracefully, then kill only a verified process group."""
        assert timeout_seconds > 0.0
        canonical = run_dir.resolve()
        initial = await self.inspect(canonical)
        if isinstance(initial, _result.Rejected):
            return initial
        original = initial.value.process
        if original is None:
            return _result.Ok(
                value=StopResult(
                    forced=False,
                    revision=initial.value.revision,
                )
            )
        handle_result = await asyncio.to_thread(
            open_process_handle, original.pid
        )
        if isinstance(handle_result, _result.Rejected):
            return handle_result
        handle = handle_result.value
        if handle is None:
            cleanup = await self._finalize_owner(canonical, original)
            if isinstance(cleanup, _result.Rejected):
                return cleanup
            empty = await self.inspect(canonical)
            if isinstance(empty, _result.Rejected):
                return empty
            return _result.Ok(
                value=StopResult(
                    forced=False, revision=empty.value.revision
                )
            )
        verified = self._inspector.inspect(canonical)
        if isinstance(verified, _result.Rejected):
            os.close(handle)
            return verified
        if verified.value is None:
            os.close(handle)
            cleanup = await self._finalize_owner(canonical, original)
            if isinstance(cleanup, _result.Rejected):
                return cleanup
            empty = await self.inspect(canonical)
            if isinstance(empty, _result.Rejected):
                return empty
            return _result.Ok(
                value=StopResult(
                    forced=False, revision=empty.value.revision
                )
            )
        if not same_process(original, verified.value):
            os.close(handle)
            return _result.Rejected(
                reason=(
                    "training PID identity changed before stop signal"
                )
            )
        signaled = await asyncio.to_thread(
            signal_process_handle, handle, signal.SIGTERM
        )
        os.close(handle)
        if isinstance(signaled, _result.Rejected):
            return signaled
        exited = await self._wait_for_exit(
            canonical, original, timeout_seconds=timeout_seconds
        )
        if isinstance(exited, _result.Rejected):
            return exited
        forced = False
        if not exited.value:
            current = await self.inspect(canonical)
            if isinstance(current, _result.Rejected):
                return current
            process = current.value.process
            if process is not None:
                if not same_process(original, process):
                    return _result.Rejected(
                        reason=(
                            "training PID identity changed while "
                            "stopping; "
                            "no force signal was sent"
                        )
                    )
                if process.process_group_id != process.pid:
                    return _result.Rejected(
                        reason=(
                            "training process is not its process-group "
                            "leader; refusing unsafe force stop"
                        )
                    )
                try:
                    os.killpg(process.process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    return _result.Rejected(
                        reason=(
                            "SIGKILL failed for training process group "
                            f"{process.process_group_id}"
                        )
                    )
                forced = True
                killed = await self._wait_for_exit(
                    canonical, original, timeout_seconds=5.0
                )
                if isinstance(killed, _result.Rejected):
                    return killed
                if not killed.value:
                    return _result.Rejected(
                        reason="training process group did not exit"
                    )
        cleanup = await self._finalize_owner(canonical, original)
        if isinstance(cleanup, _result.Rejected):
            return cleanup
        empty = await self.inspect(canonical)
        if isinstance(empty, _result.Rejected):
            return empty
        return _result.Ok(
            value=StopResult(
                forced=forced, revision=empty.value.revision
            )
        )

    async def watch(
        self, run_dir: Path, *, after_revision: int = -1
    ) -> AsyncGenerator[
        _result.Ok[ProcessEnvelope] | _result.Rejected, None
    ]:
        """Yield full snapshots, using revision only as invalidation."""
        canonical = run_dir.resolve()
        observed = after_revision
        while True:
            result = await self.inspect(canonical)
            if isinstance(result, _result.Rejected):
                yield result
                return
            if result.value.revision > observed:
                observed = result.value.revision
                yield result
                continue
            condition = self._condition(canonical)
            async with condition:
                try:
                    await asyncio.wait_for(
                        condition.wait(), timeout=1.0
                    )
                except TimeoutError:
                    pass

    async def close(self) -> None:
        """Stop server-owned watchers without stopping training."""
        tasks = tuple(self._reapers)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task

    async def _start_process(
        self,
        *,
        run_dir: Path,
        command: tuple[str, ...],
        working_directory: Path,
        command_kind: TrainingCommand,
        ready_write_fd: int | None,
        capture_output: bool,
    ) -> (
        _result.Ok[
            tuple[subprocess.Popen[bytes], ProcessSnapshot | None]
        ]
        | _result.Rejected
    ):
        assert command
        canonical = run_dir.resolve()
        async with self._mutation_lock(canonical) as lock_result:
            if isinstance(lock_result, _result.Rejected):
                return lock_result
            current = await self._inspect_locked(canonical)
            if isinstance(current, _result.Rejected):
                return current
            if current.value.process is not None:
                return _result.Rejected(
                    reason=(
                        "training process is already running: PID "
                        f"{current.value.process.pid}"
                    )
                )
            prepared = prepare_control_directory(
                self._runtime_root, canonical
            )
            if isinstance(prepared, _result.Rejected):
                return prepared
            spawn_result = await asyncio.to_thread(
                spawn_process,
                command,
                working_directory=working_directory.resolve(),
                ready_write_fd=ready_write_fd,
                capture_output=capture_output,
                output_directory=prepared.value,
            )
            if isinstance(spawn_result, _result.Rejected):
                return spawn_result
            process = spawn_result.value
            proc_result = await self._wait_for_proc(process)
            if isinstance(proc_result, _result.Rejected):
                await asyncio.to_thread(
                    terminate_process_group, process.pid
                )
                await asyncio.to_thread(process.wait)
                return proc_result
            proc = proc_result.value
            if proc is None:
                return _result.Ok(value=(process, None))
            owner = ProcessOwner(
                run_dir=canonical,
                pid=process.pid,
                start_ticks=proc.start_ticks,
                command=command_kind,
                ready=False,
            )
            owner_result = write_owner(self._runtime_root, owner)
            if isinstance(owner_result, _result.Rejected):
                await asyncio.to_thread(
                    terminate_process_group, process.pid
                )
                await asyncio.to_thread(process.wait)
                return owner_result
            inspected = self._inspector.inspect(canonical)
            if isinstance(inspected, _result.Rejected):
                await asyncio.to_thread(
                    terminate_process_group, process.pid
                )
                await asyncio.to_thread(process.wait)
                remove_owner_if_matches(
                    self._runtime_root,
                    canonical,
                    pid=owner.pid,
                    start_ticks=owner.start_ticks,
                )
                return inspected
            if inspected.value is None:
                await asyncio.to_thread(
                    terminate_process_group, process.pid
                )
                await asyncio.to_thread(process.wait)
                removed = remove_owner_if_matches(
                    self._runtime_root,
                    canonical,
                    pid=owner.pid,
                    start_ticks=owner.start_ticks,
                )
                if isinstance(removed, _result.Rejected):
                    return removed
                return _result.Rejected(
                    reason=(
                        "training process exited before ownership was "
                        "published"
                    )
                )
            publish_result = await self._publish(canonical)
            if isinstance(publish_result, _result.Rejected):
                await asyncio.to_thread(
                    terminate_process_group, process.pid
                )
                await asyncio.to_thread(process.wait)
                removed = remove_owner_if_matches(
                    self._runtime_root,
                    canonical,
                    pid=owner.pid,
                    start_ticks=owner.start_ticks,
                )
                if isinstance(removed, _result.Rejected):
                    return removed
                return publish_result
            return _result.Ok(value=(process, inspected.value))

    async def _wait_for_proc(
        self, process: subprocess.Popen[bytes]
    ) -> _result.Ok[ProcSnapshot | None] | _result.Rejected:
        deadline = time.monotonic() + min(
            self._startup_timeout_seconds, 5.0
        )
        last_error: _result.Rejected | None = None
        while time.monotonic() < deadline:
            result = self._inspector.inspect_pid(process.pid)
            if isinstance(result, _result.Rejected):
                last_error = result
            elif result.value is not None:
                return _result.Ok(value=result.value)
            if process.poll() is not None:
                return _result.Ok(value=None)
            await asyncio.sleep(0.01)
        if process.poll() is not None:
            return _result.Ok(value=None)
        if last_error is not None:
            return last_error
        return _result.Rejected(
            reason=(
                f"training process failed to start: PID {process.pid}"
            )
        )

    async def _inspect_locked(
        self, canonical: Path
    ) -> _result.Ok[ProcessEnvelope] | _result.Rejected:
        process_result = self._inspector.inspect(canonical)
        if isinstance(process_result, _result.Rejected):
            return process_result
        process = process_result.value
        owner_result = read_owner(self._runtime_root, canonical)
        if isinstance(owner_result, _result.Rejected):
            return owner_result
        owner = owner_result.value
        if process is None and owner is not None:
            removed = remove_owner_if_matches(
                self._runtime_root,
                canonical,
                pid=owner.pid,
                start_ticks=owner.start_ticks,
            )
            if isinstance(removed, _result.Rejected):
                return removed
            if removed.value:
                published = await self._publish(canonical)
                if isinstance(published, _result.Rejected):
                    return published
        revision_result = read_revision(self._runtime_root, canonical)
        if isinstance(revision_result, _result.Rejected):
            return revision_result
        return _result.Ok(
            value=ProcessEnvelope(
                revision=revision_result.value, process=process
            )
        )

    async def _wait_for_exit(
        self,
        run_dir: Path,
        original: ProcessSnapshot,
        *,
        timeout_seconds: float,
    ) -> _result.Ok[bool] | _result.Rejected:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            inspected = self._inspector.inspect(run_dir)
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

    async def _finalize_owner(
        self, run_dir: Path, process: ProcessSnapshot
    ) -> _result.Ok[None] | _result.Rejected:
        async with self._mutation_lock(run_dir) as lock_result:
            if isinstance(lock_result, _result.Rejected):
                return lock_result
            removed = remove_owner_if_matches(
                self._runtime_root,
                run_dir,
                pid=process.pid,
                start_ticks=process.start_ticks,
            )
            if isinstance(removed, _result.Ok) and removed.value:
                published = await self._publish(run_dir)
                if isinstance(published, _result.Rejected):
                    return published
            elif isinstance(removed, _result.Rejected):
                return removed
            return _result.Ok(value=None)

    def _start_reaper(
        self,
        process: subprocess.Popen[bytes],
        run_dir: Path,
        snapshot: ProcessSnapshot,
    ) -> None:
        async def reap() -> None:
            await asyncio.to_thread(process.wait)
            cleanup = await self._finalize_owner(run_dir, snapshot)
            _log_cleanup_failure(cleanup)

        task = asyncio.create_task(
            reap(), name=f"training-reaper-{process.pid}"
        )
        self._reapers.add(task)
        task.add_done_callback(self._reapers.discard)

    async def _publish(
        self, run_dir: Path
    ) -> _result.Ok[int] | _result.Rejected:
        revision = increment_revision(self._runtime_root, run_dir)
        if isinstance(revision, _result.Rejected):
            return revision
        condition = self._condition(run_dir)
        async with condition:
            condition.notify_all()
        return revision

    def _condition(self, run_dir: Path) -> asyncio.Condition:
        return self._conditions.setdefault(run_dir, asyncio.Condition())

    @asynccontextmanager
    async def _mutation_lock(
        self, run_dir: Path
    ) -> AsyncGenerator[_result.Ok[None] | _result.Rejected, None]:
        local_lock = self._locks.setdefault(run_dir, asyncio.Lock())
        async with local_lock:
            prepared = prepare_control_directory(
                self._runtime_root, run_dir
            )
            if isinstance(prepared, _result.Rejected):
                yield prepared
                return
            descriptor_result = await asyncio.to_thread(
                _acquire_file_lock,
                lock_path(self._runtime_root, run_dir),
            )
            if isinstance(descriptor_result, _result.Rejected):
                yield descriptor_result
                return
            descriptor = descriptor_result.value
            try:
                yield _result.Ok(value=None)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)


def _acquire_file_lock(
    path: Path,
) -> _result.Ok[int] | _result.Rejected:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        return _result.Rejected(
            reason=(
                f"training control lock could not be acquired: {path}"
            )
        )
    return _result.Ok(value=descriptor)


def _log_cleanup_failure(
    result: (_result.Ok[None] | _result.Ok[bool] | _result.Rejected),
) -> None:
    if isinstance(result, _result.Rejected):
        _LOGGER.error(
            "Training owner cleanup failed: %s", result.reason
        )
