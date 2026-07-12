"""Exclusive PID-file ownership for the training control adapter."""

from __future__ import annotations

import os
from pathlib import Path

from server.foundation import result as _result

PID_FILENAME = "training.pid"


def pid_path(run_dir: Path) -> Path:
    """Return the control PID path for a run directory."""
    return run_dir / PID_FILENAME


def read_pid(
    run_dir: Path,
) -> _result.Ok[int | None] | _result.Rejected:
    """Read a positive PID without following a PID-file symlink."""
    path = pid_path(run_dir)
    try:
        if path.is_symlink():
            return _result.Rejected(
                reason=(
                    f"training PID file must not be a symlink: {path}"
                )
            )
        text = path.read_text(encoding="ascii")
    except FileNotFoundError:
        return _result.Ok(value=None)
    except UnicodeError, OSError:
        return _result.Rejected(
            reason=f"training PID file is unreadable: {path}"
        )
    stripped = text.strip()
    if not stripped.isdecimal():
        return _result.Rejected(
            reason=f"training PID file is invalid: {path}"
        )
    pid = int(stripped)
    if pid <= 0:
        return _result.Rejected(
            reason=f"training PID file is invalid: {path}"
        )
    return _result.Ok(value=pid)


def write_pid(
    run_dir: Path, pid: int
) -> _result.Ok[None] | _result.Rejected:
    """Exclusively create the control PID file."""
    assert pid > 0
    path = pid_path(run_dir)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        return _result.Rejected(
            reason=f"training PID file already exists: {path}"
        )
    except OSError:
        return _result.Rejected(
            reason=f"training PID file could not be created: {path}"
        )
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as target:
            target.write(f"{pid}\n")
            target.flush()
            os.fsync(target.fileno())
    except OSError:
        _remove_path(path)
        return _result.Rejected(
            reason=f"training PID file could not be written: {path}"
        )
    return _result.Ok(value=None)


def remove_pid_if_matches(
    run_dir: Path, pid: int
) -> _result.Ok[None] | _result.Rejected:
    """Remove the PID file only when it names the expected PID."""
    assert pid > 0
    read_result = read_pid(run_dir)
    if isinstance(read_result, _result.Rejected):
        return read_result
    if read_result.value != pid:
        return _result.Ok(value=None)
    path = pid_path(run_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        return _result.Ok(value=None)
    except OSError:
        return _result.Rejected(
            reason=f"training PID file could not be removed: {path}"
        )
    return _result.Ok(value=None)


def remove_stale_pid(
    run_dir: Path, pid: int
) -> _result.Ok[None] | _result.Rejected:
    """Remove a PID file already proven to refer to no live process."""
    return remove_pid_if_matches(run_dir, pid)


def _remove_path(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError, OSError:
        return
