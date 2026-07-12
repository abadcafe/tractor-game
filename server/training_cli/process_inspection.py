"""CLI-owned Linux /proc inspection for a managed training process."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from server.foundation import result as _result

_PID_FILENAME = "training.pid"


class TrainingProcess(BaseModel):
    """Verified coordinator process facts read directly from /proc."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    pid: int
    name: str
    kernel_state: str
    executable: Path
    working_directory: Path
    run_dir: Path | None
    argv: tuple[str, ...]
    process_group_id: int
    session_id: int
    start_ticks: int


class ProcessInspector:
    """Inspect PID targets through a configurable proc filesystem."""

    def __init__(self, *, proc_root: Path = Path("/proc")) -> None:
        self._proc_root = proc_root

    def inspect(
        self, run_dir: Path
    ) -> _result.Ok[TrainingProcess | None] | _result.Rejected:
        canonical_run_dir = run_dir.resolve()
        pid_result = _read_pid(canonical_run_dir)
        if isinstance(pid_result, _result.Rejected):
            return pid_result
        pid = pid_result.value
        if pid is None:
            return _result.Ok(value=None)
        process_result = self.inspect_pid(pid)
        if isinstance(process_result, _result.Rejected):
            return process_result
        process = process_result.value
        if process is None or process.kernel_state == "Z":
            return _result.Ok(value=None)
        if (
            not _is_training_command(process.argv)
            or Path(process.argv[0]).resolve() != process.executable
            or process.run_dir != canonical_run_dir
        ):
            return _result.Rejected(
                reason=(
                    f"PID {pid} is not the managed training process "
                    "for "
                    f"{canonical_run_dir}: argv={process.argv!r}, "
                    f"cwd={process.working_directory}, "
                    f"exe={process.executable}"
                )
            )
        return _result.Ok(value=process)

    def inspect_pid(
        self, pid: int
    ) -> _result.Ok[TrainingProcess | None] | _result.Rejected:
        """Read one PID without consulting a run PID file."""
        assert pid > 0
        process_dir = self._proc_root / str(pid)
        try:
            command_bytes = (process_dir / "cmdline").read_bytes()
        except FileNotFoundError:
            return _result.Ok(value=None)
        except OSError:
            return _result.Rejected(
                reason=(
                    f"process information is unreadable: {process_dir}"
                )
            )
        if not command_bytes:
            return _result.Ok(value=None)
        argv_result = _decode_argv(command_bytes, process_dir)
        if isinstance(argv_result, _result.Rejected):
            return argv_result
        try:
            status_text = (process_dir / "status").read_text(
                encoding="utf-8"
            )
            stat_text = (process_dir / "stat").read_text(
                encoding="ascii"
            )
            working_directory = (process_dir / "cwd").resolve(
                strict=True
            )
            executable = (process_dir / "exe").resolve(strict=True)
        except FileNotFoundError:
            return _result.Ok(value=None)
        except UnicodeError, OSError:
            return _result.Rejected(
                reason=(
                    f"process information is unreadable: {process_dir}"
                )
            )
        status_result = _parse_status(status_text, process_dir)
        if isinstance(status_result, _result.Rejected):
            return status_result
        stat_result = _parse_stat(stat_text, process_dir)
        if isinstance(stat_result, _result.Rejected):
            return stat_result
        name, status_state = status_result.value
        stat_state, process_group_id, session_id, start_ticks = (
            stat_result.value
        )
        if status_state != stat_state:
            return _result.Rejected(
                reason=(
                    "process state changed during inspection: "
                    f"{process_dir}"
                )
            )
        run_dir = _run_dir(argv_result.value, working_directory)
        return _result.Ok(
            value=TrainingProcess(
                pid=pid,
                name=name,
                kernel_state=stat_state,
                executable=executable,
                working_directory=working_directory,
                run_dir=run_dir,
                argv=argv_result.value,
                process_group_id=process_group_id,
                session_id=session_id,
                start_ticks=start_ticks,
            )
        )


def same_process(left: TrainingProcess, right: TrainingProcess) -> bool:
    """Return whether observations identify one process lifetime."""
    return (
        left.pid == right.pid and left.start_ticks == right.start_ticks
    )


def _decode_argv(
    data: bytes, process_dir: Path
) -> _result.Ok[tuple[str, ...]] | _result.Rejected:
    try:
        argv = tuple(
            part.decode("utf-8")
            for part in data.rstrip(b"\0").split(b"\0")
            if part
        )
    except UnicodeDecodeError:
        return _result.Rejected(
            reason=f"process command is not valid UTF-8: {process_dir}"
        )
    if not argv:
        return _result.Rejected(
            reason=f"process command is empty: {process_dir}"
        )
    return _result.Ok(value=argv)


def _parse_status(
    text: str, process_dir: Path
) -> _result.Ok[tuple[str, str]] | _result.Rejected:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key] = value.strip()
    name = fields.get("Name")
    state_text = fields.get("State")
    if name is None or state_text is None or not state_text:
        return _result.Rejected(
            reason=f"process status is malformed: {process_dir}"
        )
    return _result.Ok(value=(name, state_text[0]))


def _parse_stat(
    text: str, process_dir: Path
) -> _result.Ok[tuple[str, int, int, int]] | _result.Rejected:
    command_end = text.rfind(")")
    if command_end < 0:
        return _result.Rejected(
            reason=f"process stat is malformed: {process_dir}"
        )
    fields = text[command_end + 1 :].strip().split()
    if len(fields) < 20:
        return _result.Rejected(
            reason=f"process stat is malformed: {process_dir}"
        )
    try:
        process_group_id = int(fields[2])
        session_id = int(fields[3])
        start_ticks = int(fields[19])
    except ValueError:
        return _result.Rejected(
            reason=f"process stat is malformed: {process_dir}"
        )
    return _result.Ok(
        value=(fields[0], process_group_id, session_id, start_ticks)
    )


def _is_training_command(argv: tuple[str, ...]) -> bool:
    return (
        len(argv) >= 7
        and argv[1:4] == ("-m", "server.training_cli", "--run-dir")
        and bool(argv[4])
        and argv[5] == "resume"
        and _is_managed_checkpoint_name(argv[6])
    )


def _is_managed_checkpoint_name(value: str) -> bool:
    if value == "latest.json":
        return True
    prefix = "update-"
    suffix = ".json"
    if not value.startswith(prefix) or not value.endswith(suffix):
        return False
    update = value[len(prefix) : -len(suffix)]
    return update.isdecimal() and not update.startswith("0")


def _run_dir(
    argv: tuple[str, ...], working_directory: Path
) -> Path | None:
    values: list[str] = []
    for index, part in enumerate(argv):
        if part == "--run-dir" and index + 1 < len(argv):
            values.append(argv[index + 1])
    if len(values) != 1 or not values[0]:
        return None
    path = Path(values[0])
    if not path.is_absolute():
        path = working_directory / path
    return path.resolve()


def _read_pid(
    run_dir: Path,
) -> _result.Ok[int | None] | _result.Rejected:
    path = run_dir / _PID_FILENAME
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
