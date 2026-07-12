"""Bounded readers for detached training process logs."""

from __future__ import annotations

from pathlib import Path

from server.foundation import result as _result
from server.training_control.process_control import (
    STDERR_FILENAME,
    STDOUT_FILENAME,
)


def read_log_tail(
    run_dir: Path,
    *,
    stream: str,
    max_bytes: int,
) -> _result.Ok[str] | _result.Rejected:
    """Read a bounded stdout or stderr suffix."""
    if stream not in ("stdout", "stderr"):
        return _result.Rejected(reason="invalid training log stream")
    if max_bytes <= 0 or max_bytes > 2_000_000:
        return _result.Rejected(
            reason="max_bytes must be between 1 and 2000000"
        )
    filename = (
        STDOUT_FILENAME if stream == "stdout" else STDERR_FILENAME
    )
    path = run_dir.resolve() / filename
    try:
        if path.is_symlink():
            return _result.Rejected(
                reason=f"training log must not be a symlink: {path}"
            )
        with path.open("rb") as source:
            source.seek(0, 2)
            size = source.tell()
            source.seek(max(0, size - max_bytes))
            data = source.read(max_bytes)
    except FileNotFoundError:
        return _result.Ok(value="")
    except OSError:
        return _result.Rejected(
            reason=f"training log is unreadable: {path}"
        )
    return _result.Ok(value=data.decode("utf-8", errors="replace"))
