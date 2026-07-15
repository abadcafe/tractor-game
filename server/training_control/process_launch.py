"""Low-level spawn and readiness pipe operations."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import psutil
from pydantic import BaseModel, ConfigDict, ValidationError

from server.foundation import result as _result
from server.training_control.process_inspection import (
    portable_start_token,
)


class _LinuxOsApi(Protocol):
    def pidfd_open(self, pid: int, flags: int = 0) -> int: ...


class _LinuxSignalApi(Protocol):
    def pidfd_send_signal(
        self,
        descriptor: int,
        requested_signal: int,
        siginfo: None = None,
        flags: int = 0,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _LinuxProcessHandle:
    descriptor: int


@dataclass(frozen=True, slots=True)
class _PortableProcessHandle:
    process: psutil.Process
    start_token: int


type ProcessHandle = _LinuxProcessHandle | _PortableProcessHandle

_LINUX_OS = cast(_LinuxOsApi, os)
_LINUX_SIGNAL = cast(_LinuxSignalApi, signal)


class _ControlMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["ready", "error"]
    error: str | None = None


def spawn_process(
    command: tuple[str, ...],
    *,
    working_directory: Path,
    ready_write_fd: int | None,
    capture_output: bool,
    output_directory: Path,
) -> _result.Ok[subprocess.Popen[bytes]] | _result.Rejected:
    """Spawn one process-group leader without discarding diagnostics."""
    output_handle = None
    try:
        if capture_output:
            stdout: int = subprocess.PIPE
            stderr: int = subprocess.PIPE
        else:
            output_handle = output_directory.joinpath(
                "cli-output.log"
            ).open("ab", buffering=0)
            stdout = output_handle.fileno()
            stderr = output_handle.fileno()
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            pass_fds=()
            if ready_write_fd is None
            else (ready_write_fd,),
            start_new_session=True,
        )
    except OSError as error:
        return _result.Rejected(
            reason=f"training process could not be started: {error}"
        )
    finally:
        if output_handle is not None:
            output_handle.close()
    return _result.Ok(value=process)


def read_control_message(
    descriptor: int,
) -> _result.Ok[None] | _result.Rejected:
    """Read one bounded, strict readiness result from the CLI."""
    chunks: list[bytes] = []
    size = 0
    while True:
        try:
            chunk = os.read(descriptor, 4096)
        except OSError as error:
            return _result.Rejected(
                reason=(
                    "training readiness pipe could not be read: "
                    f"{error}"
                )
            )
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > 65_536:
            return _result.Rejected(
                reason="training readiness message is too large"
            )
        if b"\n" in chunk:
            break
    data = b"".join(chunks).partition(b"\n")[0]
    if not data:
        return _result.Rejected(
            reason="training process exited before reporting readiness"
        )
    try:
        message = _ControlMessage.model_validate_json(data)
    except ValidationError:
        return _result.Rejected(
            reason="training readiness message is invalid"
        )
    if message.type == "error":
        return _result.Rejected(
            reason=message.error or "training startup failed"
        )
    if message.error is not None:
        return _result.Rejected(
            reason="training readiness message is invalid"
        )
    return _result.Ok(value=None)


def terminate_process_group(pid: int) -> None:
    """Kill a just-spawned process group whose PID identity is owned."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def open_process_handle(
    pid: int, start_token: int
) -> _result.Ok[ProcessHandle | None] | _result.Rejected:
    """Open an identity-bound handle, or none after process exit."""
    if not sys.platform.startswith("linux"):
        return _open_portable_process_handle(pid, start_token)
    try:
        return _result.Ok(
            value=_LinuxProcessHandle(
                descriptor=_LINUX_OS.pidfd_open(pid)
            )
        )
    except ProcessLookupError:
        return _result.Ok(value=None)
    except OSError as error:
        return _result.Rejected(
            reason=(
                f"training process handle could not be opened: {error}"
            )
        )


def signal_process_handle(
    handle: ProcessHandle, requested_signal: int
) -> _result.Ok[None] | _result.Rejected:
    """Signal the exact process lifetime referenced by a handle."""
    if isinstance(handle, _PortableProcessHandle):
        return _signal_portable_process_handle(handle, requested_signal)
    try:
        _LINUX_SIGNAL.pidfd_send_signal(
            handle.descriptor, requested_signal
        )
    except ProcessLookupError:
        return _result.Ok(value=None)
    except OSError as error:
        return _result.Rejected(
            reason=f"training process signal failed: {error}"
        )
    return _result.Ok(value=None)


def close_process_handle(handle: ProcessHandle) -> None:
    """Release operating-system resources owned by a process handle."""
    if isinstance(handle, _LinuxProcessHandle):
        os.close(handle.descriptor)


def _open_portable_process_handle(
    pid: int, start_token: int
) -> _result.Ok[ProcessHandle | None] | _result.Rejected:
    try:
        process = psutil.Process(pid)
        observed_token = portable_start_token(process)
    except psutil.NoSuchProcess, psutil.ZombieProcess:
        return _result.Ok(value=None)
    except psutil.Error, OSError, ValueError:
        return _result.Rejected(
            reason=(
                "training process handle could not be opened: "
                f"PID {pid}"
            )
        )
    if observed_token != start_token:
        return _result.Rejected(
            reason=(
                "training process identity changed before handle open"
            )
        )
    return _result.Ok(
        value=_PortableProcessHandle(
            process=process, start_token=start_token
        )
    )


def _signal_portable_process_handle(
    handle: _PortableProcessHandle, requested_signal: int
) -> _result.Ok[None] | _result.Rejected:
    try:
        if portable_start_token(handle.process) != handle.start_token:
            return _result.Rejected(
                reason="training process identity changed before signal"
            )
        handle.process.send_signal(requested_signal)
    except psutil.NoSuchProcess, psutil.ZombieProcess:
        return _result.Ok(value=None)
    except (psutil.Error, OSError, ValueError) as error:
        return _result.Rejected(
            reason=f"training process signal failed: {error}"
        )
    return _result.Ok(value=None)
