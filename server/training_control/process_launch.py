"""Low-level spawn and readiness pipe operations."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from server.foundation import result as _result


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
    pid: int,
) -> _result.Ok[int | None] | _result.Rejected:
    """Open a Linux pidfd, returning none only after process exit."""
    try:
        return _result.Ok(value=os.pidfd_open(pid))
    except ProcessLookupError:
        return _result.Ok(value=None)
    except OSError as error:
        return _result.Rejected(
            reason=(
                f"training process handle could not be opened: {error}"
            )
        )


def signal_process_handle(
    descriptor: int, requested_signal: int
) -> _result.Ok[None] | _result.Rejected:
    """Signal the exact process lifetime referenced by a pidfd."""
    try:
        signal.pidfd_send_signal(descriptor, requested_signal)
    except ProcessLookupError:
        return _result.Ok(value=None)
    except OSError as error:
        return _result.Rejected(
            reason=f"training process signal failed: {error}"
        )
    return _result.Ok(value=None)
