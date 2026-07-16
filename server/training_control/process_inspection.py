"""PID-file-backed process status and diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import psutil
from pydantic import BaseModel, ConfigDict, Field

from server.foundation import result as _result

PID_FILE_NAME = "training.pid"
_MAX_PID_FILE_BYTES = 64


class ProcessDetails(BaseModel):
    """Best-effort operating-system facts shown to the user."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["details"] = "details"
    started_at_ms: int = Field(ge=0)
    kernel_state: str
    executable: Path
    working_directory: Path
    argv: tuple[str, ...]
    process_group_id: int = Field(gt=0)
    unix_session_id: int = Field(gt=0)


class ProcessInspectionError(BaseModel):
    """A live PID whose diagnostics cannot be read."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["error"] = "error"
    error: str


type ProcessInspection = ProcessDetails | ProcessInspectionError


class ProcessSnapshot(BaseModel):
    """A live PID from the selected run's PID file."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    pid: int = Field(gt=0)
    inspection: ProcessInspection


class ProcessState(BaseModel):
    """Current PID-file process state for one training run."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    process: ProcessSnapshot | None


class ProcessInspector:
    """Expose the PID-file process without ownership checks."""

    def inspect(
        self, run_dir: Path
    ) -> _result.Ok[ProcessState] | _result.Rejected:
        canonical = run_dir.resolve()
        pid_result = read_training_pid(canonical)
        if isinstance(pid_result, _result.Rejected):
            return pid_result
        pid = pid_result.value
        if pid is None:
            return _result.Ok(value=ProcessState(process=None))
        exists_result = process_exists(pid)
        if isinstance(exists_result, _result.Rejected):
            return exists_result
        if not exists_result.value:
            return _result.Ok(value=ProcessState(process=None))
        details = _inspect_process(pid)
        if details is None:
            return _result.Ok(value=ProcessState(process=None))
        return _result.Ok(
            value=ProcessState(
                process=ProcessSnapshot(pid=pid, inspection=details)
            )
        )


def pid_file_path(run_dir: Path) -> Path:
    return run_dir.resolve() / PID_FILE_NAME


def read_training_pid(
    run_dir: Path,
) -> _result.Ok[int | None] | _result.Rejected:
    """Read one positive PID; malformed content is stale state."""
    path = pid_file_path(run_dir)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return _result.Ok(value=None)
    except OSError:
        return _result.Rejected(
            reason=f"training PID file is unreadable: {path}"
        )
    try:
        content = os.read(descriptor, _MAX_PID_FILE_BYTES + 1)
    except OSError:
        return _result.Rejected(
            reason=f"training PID file is unreadable: {path}"
        )
    finally:
        os.close(descriptor)
    if len(content) > _MAX_PID_FILE_BYTES:
        return _result.Ok(value=None)
    try:
        pid = int(content.decode("ascii").strip())
    except UnicodeDecodeError, ValueError:
        return _result.Ok(value=None)
    return _result.Ok(value=pid if pid > 0 else None)


def write_training_pid(
    run_dir: Path, pid: int
) -> _result.Ok[None] | _result.Rejected:
    """Atomically replace stale PID state after a successful spawn."""
    assert pid > 0
    canonical = run_dir.resolve()
    path = pid_file_path(canonical)
    temporary = canonical / f".{PID_FILE_NAME}.{os.getpid()}.tmp"
    descriptor: int | None = None
    try:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        payload = f"{pid}\n".encode("ascii")
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            assert written > 0
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        return _result.Rejected(
            reason=f"training PID file could not be written: {path}"
        )
    return _result.Ok(value=None)


def remove_training_pid_if_matches(
    run_dir: Path, pid: int
) -> _result.Ok[bool] | _result.Rejected:
    """Remove the PID file only when it contains the expected PID."""
    assert pid > 0
    current_result = read_training_pid(run_dir)
    if isinstance(current_result, _result.Rejected):
        return current_result
    if current_result.value != pid:
        return _result.Ok(value=False)
    path = pid_file_path(run_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        return _result.Ok(value=False)
    except OSError:
        return _result.Rejected(
            reason=f"training PID file could not be removed: {path}"
        )
    return _result.Ok(value=True)


def remove_training_pid(
    run_dir: Path,
) -> _result.Ok[None] | _result.Rejected:
    """Remove stale PID state during an explicit stop."""
    path = pid_file_path(run_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        return _result.Rejected(
            reason=f"training PID file could not be removed: {path}"
        )
    return _result.Ok(value=None)


def process_exists(
    pid: int,
) -> _result.Ok[bool] | _result.Rejected:
    """Implement PID existence with kill(2), treating EPERM as alive."""
    assert pid > 0
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return _result.Ok(value=False)
    except PermissionError:
        return _result.Ok(value=True)
    except OSError:
        return _result.Rejected(
            reason=f"training PID existence check failed: PID {pid}"
        )
    return _result.Ok(value=True)


def _inspect_process(pid: int) -> ProcessInspection | None:
    try:
        process = psutil.Process(pid)
        with process.oneshot():
            status = process.status()
            started_at_ms = round(process.create_time() * 1000)
            argv = tuple(process.cmdline())
            working_directory = Path(process.cwd()).resolve()
            executable = Path(process.exe()).resolve()
            process_group_id = os.getpgid(pid)
            unix_session_id = os.getsid(pid)
    except psutil.NoSuchProcess, ProcessLookupError:
        return None
    except psutil.Error, OSError, ValueError:
        return ProcessInspectionError(
            error=f"process information is unreadable: PID {pid}"
        )
    return ProcessDetails(
        started_at_ms=started_at_ms,
        kernel_state=status,
        executable=executable,
        working_directory=working_directory,
        argv=argv,
        process_group_id=process_group_id,
        unix_session_id=unix_session_id,
    )
