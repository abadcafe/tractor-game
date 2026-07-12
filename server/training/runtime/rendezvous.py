"""File rendezvous creation for torch distributed process groups."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from server.foundation import result as _result


@dataclass(frozen=True, slots=True)
class FileRendezvous:
    """Torch distributed file init method."""

    init_method: str
    path: Path

    def __post_init__(self) -> None:
        assert self.init_method.startswith("file:///")
        assert self.path.is_absolute()


def create_file_rendezvous(
    run_dir: Path,
) -> _result.Ok[FileRendezvous] | _result.Rejected:
    """Create a file rendezvous path inside a run directory."""
    runtime_dir = run_dir / "runtime"
    try:
        runtime_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _result.Rejected(
            reason=f"failed to create distributed runtime dir: {exc}"
        )
    rendezvous_path = (
        runtime_dir / f"torch-distributed-{time.time_ns()}"
    ).resolve(strict=False)
    return _result.Ok(
        value=FileRendezvous(
            init_method=_torch_file_init_method(rendezvous_path),
            path=rendezvous_path,
        )
    )


def _torch_file_init_method(path: Path) -> str:
    # Torch file rendezvous reads the path literally and does not
    # URI-decode escaped characters such as spaces.
    return f"file://{path.as_posix()}"
