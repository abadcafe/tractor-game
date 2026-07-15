"""External owner records for managed training CLI processes."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from server.foundation import result as _result

type TrainingCommand = Literal["initialize", "resume"]


class ProcessOwner(BaseModel):
    """Persistent identity for exactly one process lifetime."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    run_dir: Path
    pid: int = Field(gt=0)
    start_ticks: int = Field(ge=0)
    command: TrainingCommand
    ready: bool


def control_directory(runtime_root: Path, run_dir: Path) -> Path:
    """Return a stable control directory outside the training run."""
    canonical = str(run_dir.resolve()).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return runtime_root.resolve() / digest


def owner_path(runtime_root: Path, run_dir: Path) -> Path:
    return control_directory(runtime_root, run_dir) / "owner.json"


def lock_path(runtime_root: Path, run_dir: Path) -> Path:
    return control_directory(runtime_root, run_dir) / "lock"


def revision_path(runtime_root: Path, run_dir: Path) -> Path:
    return control_directory(runtime_root, run_dir) / "revision"


def read_owner(
    runtime_root: Path, run_dir: Path
) -> _result.Ok[ProcessOwner | None] | _result.Rejected:
    """Read one strict owner record without following symlinks."""
    path = owner_path(runtime_root, run_dir)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "r", encoding="utf-8") as source:
            document = source.read()
    except FileNotFoundError:
        return _result.Ok(value=None)
    except UnicodeError, OSError:
        return _result.Rejected(
            reason=f"training owner is unreadable: {path}"
        )
    try:
        owner = ProcessOwner.model_validate_json(document)
    except ValidationError:
        return _result.Rejected(
            reason=f"training owner is invalid: {path}"
        )
    if owner.run_dir != run_dir.resolve():
        return _result.Rejected(
            reason=(
                f"training owner run directory does not match: {path}"
            )
        )
    return _result.Ok(value=owner)


def write_owner(
    runtime_root: Path, owner: ProcessOwner
) -> _result.Ok[None] | _result.Rejected:
    """Atomically publish an owner after process identity is known."""
    path = owner_path(runtime_root, owner.run_dir)
    directory_result = _prepare_directory(path.parent)
    if isinstance(directory_result, _result.Rejected):
        return directory_result
    temporary = path.parent / (
        f"owner.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(owner.model_dump(mode="json"), target)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except OSError:
        _remove_path(temporary)
        return _result.Rejected(
            reason=f"training owner could not be written: {path}"
        )
    return _result.Ok(value=None)


def remove_owner_if_matches(
    runtime_root: Path,
    run_dir: Path,
    *,
    pid: int,
    start_ticks: int,
) -> _result.Ok[bool] | _result.Rejected:
    """Remove an owner only for the exact recorded process lifetime."""
    result = read_owner(runtime_root, run_dir)
    if isinstance(result, _result.Rejected):
        return result
    owner = result.value
    if owner is None or (
        owner.pid != pid or owner.start_ticks != start_ticks
    ):
        return _result.Ok(value=False)
    path = owner_path(runtime_root, run_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        return _result.Ok(value=False)
    except OSError:
        return _result.Rejected(
            reason=f"training owner could not be removed: {path}"
        )
    return _result.Ok(value=True)


def mark_owner_ready(
    runtime_root: Path,
    run_dir: Path,
    *,
    pid: int,
    start_ticks: int,
) -> _result.Ok[None] | _result.Rejected:
    """Atomically mark the exact resume process as ready."""
    result = read_owner(runtime_root, run_dir)
    if isinstance(result, _result.Rejected):
        return result
    owner = result.value
    if owner is None or (
        owner.pid != pid or owner.start_ticks != start_ticks
    ):
        return _result.Rejected(
            reason="training owner changed before readiness"
        )
    if owner.ready:
        return _result.Ok(value=None)
    return write_owner(
        runtime_root, owner.model_copy(update={"ready": True})
    )


def read_revision(
    runtime_root: Path, run_dir: Path
) -> _result.Ok[int] | _result.Rejected:
    path = revision_path(runtime_root, run_dir)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "r", encoding="ascii") as source:
            value = source.read().strip()
    except FileNotFoundError:
        return _result.Ok(value=0)
    except UnicodeError, OSError:
        return _result.Rejected(
            reason=f"training process revision is unreadable: {path}"
        )
    if not value.isdecimal():
        return _result.Rejected(
            reason=f"training process revision is invalid: {path}"
        )
    return _result.Ok(value=int(value))


def increment_revision(
    runtime_root: Path, run_dir: Path
) -> _result.Ok[int] | _result.Rejected:
    previous = read_revision(runtime_root, run_dir)
    if isinstance(previous, _result.Rejected):
        return previous
    path = revision_path(runtime_root, run_dir)
    directory_result = _prepare_directory(path.parent)
    if isinstance(directory_result, _result.Rejected):
        return directory_result
    revision = previous.value + 1
    temporary = path.parent / (
        f"revision.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="ascii") as target:
            target.write(f"{revision}\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except OSError:
        _remove_path(temporary)
        return _result.Rejected(
            reason=(
                "training process revision could not be written: "
                f"{path}"
            )
        )
    return _result.Ok(value=revision)


def prepare_control_directory(
    runtime_root: Path, run_dir: Path
) -> _result.Ok[Path] | _result.Rejected:
    directory = control_directory(runtime_root, run_dir)
    result = _prepare_directory(directory)
    if isinstance(result, _result.Rejected):
        return result
    return _result.Ok(value=directory)


def _prepare_directory(
    directory: Path,
) -> _result.Ok[None] | _result.Rejected:
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        unsafe = directory.is_symlink() or not directory.is_dir()
        if unsafe:
            return _result.Rejected(
                reason=(
                    f"training control directory is unsafe: {directory}"
                )
            )
        directory.chmod(0o700)
    except OSError:
        return _result.Rejected(
            reason=(
                "training control directory could not be prepared: "
                f"{directory}"
            )
        )
    return _result.Ok(value=None)


def _remove_path(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError, OSError:
        return
