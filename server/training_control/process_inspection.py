"""Cross-platform inspection for externally owned training processes."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import psutil
from pydantic import BaseModel, ConfigDict, Field

from server.foundation import result as _result
from server.training_control.process_owner import (
    ProcessOwner,
    TrainingCommand,
    read_owner,
)

type ProcessInspectionBackend = Literal["proc", "portable"]


class ProcessSnapshot(BaseModel):
    """Public facts for one exact managed CLI process lifetime."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    pid: int = Field(gt=0)
    start_ticks: int = Field(ge=0)
    started_at_ms: int = Field(ge=0)
    kernel_state: str
    executable: Path
    working_directory: Path
    run_dir: Path
    argv: tuple[str, ...]
    process_group_id: int = Field(gt=0)
    unix_session_id: int = Field(gt=0)
    command: TrainingCommand
    ready: bool


class ProcSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    pid: int
    start_ticks: int
    started_at_ms: int
    kernel_state: str
    executable: Path
    working_directory: Path
    run_dir: Path | None
    argv: tuple[str, ...]
    process_group_id: int
    unix_session_id: int
    cli_command: TrainingCommand | None


class ProcessInspector:
    """Resolve an external owner against stable OS process facts."""

    def __init__(
        self,
        *,
        runtime_root: Path,
        proc_root: Path | None = None,
        clock_ticks_per_second: int | None = None,
        backend: ProcessInspectionBackend | None = None,
    ) -> None:
        self._runtime_root = runtime_root.resolve()
        self._backend = (
            (
                "proc"
                if proc_root is not None
                or sys.platform.startswith("linux")
                else "portable"
            )
            if backend is None
            else backend
        )
        self._proc_root = (
            Path("/proc") if proc_root is None else proc_root
        )
        self._clock_ticks_per_second = (
            os.sysconf("SC_CLK_TCK")
            if clock_ticks_per_second is None
            else clock_ticks_per_second
        )
        assert self._clock_ticks_per_second > 0

    def inspect(
        self, run_dir: Path
    ) -> _result.Ok[ProcessSnapshot | None] | _result.Rejected:
        canonical = run_dir.resolve()
        owner_result = read_owner(self._runtime_root, canonical)
        if isinstance(owner_result, _result.Rejected):
            return owner_result
        owner = owner_result.value
        if owner is None:
            return _result.Ok(value=None)
        process_result = self.inspect_pid(owner.pid)
        if isinstance(process_result, _result.Rejected):
            return process_result
        process = process_result.value
        if process is None or process.kernel_state == "Z":
            return _result.Ok(value=None)
        mismatch = _identity_mismatch(owner, process)
        if mismatch is not None:
            return _result.Rejected(reason=mismatch)
        return _result.Ok(
            value=ProcessSnapshot(
                pid=process.pid,
                start_ticks=process.start_ticks,
                started_at_ms=process.started_at_ms,
                kernel_state=process.kernel_state,
                executable=process.executable,
                working_directory=process.working_directory,
                run_dir=canonical,
                argv=process.argv,
                process_group_id=process.process_group_id,
                unix_session_id=process.unix_session_id,
                command=owner.command,
                ready=owner.ready,
            )
        )

    def inspect_pid(
        self, pid: int
    ) -> _result.Ok[ProcSnapshot | None] | _result.Rejected:
        """Read one PID with a stable lifetime identity token."""
        assert pid > 0
        if self._backend == "portable":
            return _inspect_portable_pid(pid)
        return self._inspect_proc_pid(pid)

    def _inspect_proc_pid(
        self, pid: int
    ) -> _result.Ok[ProcSnapshot | None] | _result.Rejected:
        """Read a Linux PID twice to reject reuse during inspection."""
        process_dir = self._proc_root / str(pid)
        try:
            initial_stat_text = (process_dir / "stat").read_text(
                encoding="ascii"
            )
            command_bytes = (process_dir / "cmdline").read_bytes()
        except FileNotFoundError:
            return _result.Ok(value=None)
        except OSError:
            if not process_dir.exists():
                return _result.Ok(value=None)
            return _unreadable(process_dir)
        if not command_bytes:
            return _result.Ok(value=None)
        argv_result = _decode_argv(command_bytes, process_dir)
        if isinstance(argv_result, _result.Rejected):
            return argv_result
        initial = _parse_stat(
            initial_stat_text, process_dir, expected_pid=pid
        )
        if isinstance(initial, _result.Rejected):
            return initial
        if initial.value[0] == "Z":
            return _result.Ok(value=None)
        try:
            working_directory = (process_dir / "cwd").resolve(
                strict=True
            )
            executable = (process_dir / "exe").resolve(strict=True)
            final_stat_text = (process_dir / "stat").read_text(
                encoding="ascii"
            )
        except FileNotFoundError:
            return _result.Ok(value=None)
        except OSError:
            if not process_dir.exists():
                return _result.Ok(value=None)
            return _unreadable(process_dir)
        final = _parse_stat(
            final_stat_text, process_dir, expected_pid=pid
        )
        if isinstance(final, _result.Rejected):
            return final
        if initial.value[3] != final.value[3]:
            return _result.Rejected(
                reason=(
                    "process lifetime changed during inspection: "
                    f"{process_dir}"
                )
            )
        state, process_group_id, unix_session_id, start_ticks = (
            final.value
        )
        boot_result = _boot_time_ms(self._proc_root)
        if isinstance(boot_result, _result.Rejected):
            return boot_result
        argv = argv_result.value
        return _result.Ok(
            value=ProcSnapshot(
                pid=pid,
                start_ticks=start_ticks,
                started_at_ms=(
                    boot_result.value
                    + start_ticks * 1000 // self._clock_ticks_per_second
                ),
                kernel_state=state,
                executable=executable,
                working_directory=working_directory,
                run_dir=_run_dir(argv, working_directory),
                argv=argv,
                process_group_id=process_group_id,
                unix_session_id=unix_session_id,
                cli_command=_cli_command(argv),
            )
        )


def portable_start_token(process: psutil.Process) -> int:
    """Return a stable integer token for a portable process lifetime."""
    created_at = process.create_time()
    if created_at <= 0.0:
        raise ValueError("process creation time must be positive")
    return round(created_at * 1_000_000)


def _inspect_portable_pid(
    pid: int,
) -> _result.Ok[ProcSnapshot | None] | _result.Rejected:
    try:
        process = psutil.Process(pid)
        with process.oneshot():
            status = process.status()
            if status == psutil.STATUS_ZOMBIE:
                return _result.Ok(value=None)
            start_token = portable_start_token(process)
            started_at_ms = round(process.create_time() * 1000)
            argv = tuple(process.cmdline())
            working_directory = Path(process.cwd()).resolve()
            executable = Path(process.exe()).resolve()
            process_group_id = os.getpgid(pid)
            unix_session_id = os.getsid(pid)
    except (
        psutil.NoSuchProcess,
        psutil.ZombieProcess,
        ProcessLookupError,
    ):
        return _result.Ok(value=None)
    except psutil.Error, OSError, ValueError:
        return _result.Rejected(
            reason=f"process information is unreadable: PID {pid}"
        )
    if not argv:
        return _result.Ok(value=None)
    return _result.Ok(
        value=ProcSnapshot(
            pid=pid,
            start_ticks=start_token,
            started_at_ms=started_at_ms,
            kernel_state=status,
            executable=executable,
            working_directory=working_directory,
            run_dir=_run_dir(argv, working_directory),
            argv=argv,
            process_group_id=process_group_id,
            unix_session_id=unix_session_id,
            cli_command=_cli_command(argv),
        )
    )


def same_process(left: ProcessSnapshot, right: ProcessSnapshot) -> bool:
    return (
        left.pid == right.pid and left.start_ticks == right.start_ticks
    )


def _identity_mismatch(
    owner: ProcessOwner, process: ProcSnapshot
) -> str | None:
    expected_executable = Path(process.argv[0])
    if not expected_executable.is_absolute():
        expected_executable = (
            process.working_directory / expected_executable
        )
    if (
        process.start_ticks != owner.start_ticks
        or process.run_dir != owner.run_dir
        or process.cli_command != owner.command
        or expected_executable.resolve() != process.executable
    ):
        return (
            f"PID {owner.pid} is not the owned training process for "
            f"{owner.run_dir}: start_ticks={process.start_ticks}, "
            f"argv={process.argv!r}, cwd={process.working_directory}, "
            f"exe={process.executable}"
        )
    return None


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


def _parse_stat(
    text: str, process_dir: Path, *, expected_pid: int
) -> _result.Ok[tuple[str, int, int, int]] | _result.Rejected:
    command_start = text.find("(")
    command_end = text.rfind(")")
    if command_start < 0 or command_end <= command_start:
        return _malformed_stat(process_dir)
    fields = text[command_end + 1 :].strip().split()
    if len(fields) < 20:
        return _malformed_stat(process_dir)
    try:
        observed_pid = int(text[:command_start].strip())
        process_group_id = int(fields[2])
        unix_session_id = int(fields[3])
        start_ticks = int(fields[19])
    except ValueError:
        return _malformed_stat(process_dir)
    if observed_pid != expected_pid:
        return _result.Rejected(
            reason=f"process stat PID does not match: {process_dir}"
        )
    return _result.Ok(
        value=(
            fields[0],
            process_group_id,
            unix_session_id,
            start_ticks,
        )
    )


def _boot_time_ms(
    proc_root: Path,
) -> _result.Ok[int] | _result.Rejected:
    path = proc_root / "stat"
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except UnicodeError, OSError:
        return _result.Rejected(
            reason=f"process boot time is unreadable: {path}"
        )
    values = [
        line.split() for line in lines if line.startswith("btime ")
    ]
    if len(values) != 1 or len(values[0]) != 2:
        return _result.Rejected(
            reason=f"process boot time is malformed: {path}"
        )
    try:
        boot_seconds = int(values[0][1])
    except ValueError:
        return _result.Rejected(
            reason=f"process boot time is malformed: {path}"
        )
    if boot_seconds < 0:
        return _result.Rejected(
            reason=f"process boot time is malformed: {path}"
        )
    return _result.Ok(value=boot_seconds * 1000)


def _cli_command(argv: tuple[str, ...]) -> TrainingCommand | None:
    if len(argv) < 6:
        return None
    if argv[1:4] != ("-m", "server.training_cli", "--run-dir"):
        return None
    if not argv[4]:
        return None
    subcommand = argv[5]
    if subcommand == "init":
        return "initialize"
    if (
        subcommand == "resume"
        and len(argv) >= 7
        and _is_managed_checkpoint_name(argv[6])
    ):
        return "resume"
    return None


def _is_managed_checkpoint_name(value: str) -> bool:
    if value == "latest.json":
        return True
    if not value.startswith("update-") or not value.endswith(".json"):
        return False
    update = value[len("update-") : -len(".json")]
    return update.isdecimal() and not update.startswith("0")


def _run_dir(
    argv: tuple[str, ...], working_directory: Path
) -> Path | None:
    values = [
        argv[index + 1]
        for index, part in enumerate(argv[:-1])
        if part == "--run-dir"
    ]
    if len(values) != 1 or not values[0]:
        return None
    path = Path(values[0])
    if not path.is_absolute():
        path = working_directory / path
    return path.resolve()


def _unreadable(process_dir: Path) -> _result.Rejected:
    return _result.Rejected(
        reason=f"process information is unreadable: {process_dir}"
    )


def _malformed_stat(process_dir: Path) -> _result.Rejected:
    return _result.Rejected(
        reason=f"process stat is malformed: {process_dir}"
    )
