"""Black-box tests for strict cross-platform process inspection."""

import os
import sys
from pathlib import Path

from server.foundation.result import Ok, Rejected
from server.training_control.process_inspection import ProcessInspector
from server.training_control.process_owner import (
    ProcessOwner,
    write_owner,
)


def test_missing_external_owner_returns_no_process(
    tmp_path: Path,
) -> None:
    inspector = ProcessInspector(
        runtime_root=tmp_path / "control",
        proc_root=tmp_path / "proc",
        clock_ticks_per_second=100,
    )

    result = inspector.inspect(tmp_path / "run")

    assert isinstance(result, Ok)
    assert result.value is None


def test_owner_and_proc_produce_complete_snapshot(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    runtime_root = tmp_path / "control"
    proc_root = tmp_path / "proc"
    _write_process(proc_root, 123, run_dir, start_ticks=98_765)
    owner = ProcessOwner(
        run_dir=run_dir.resolve(),
        pid=123,
        start_ticks=98_765,
        command="resume",
    )
    assert isinstance(write_owner(runtime_root, owner), Ok)
    inspector = ProcessInspector(
        runtime_root=runtime_root,
        proc_root=proc_root,
        clock_ticks_per_second=100,
    )

    result = inspector.inspect(run_dir)

    assert isinstance(result, Ok)
    process = result.value
    assert process is not None
    assert process.pid == 123
    assert process.start_ticks == 98_765
    assert process.started_at_ms == 1_987_650
    assert process.kernel_state == "S"
    assert process.process_group_id == 123
    assert process.unix_session_id == 123
    assert process.run_dir == run_dir.resolve()
    assert process.command == "resume"


def test_pid_reuse_is_rejected_by_start_ticks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    runtime_root = tmp_path / "control"
    proc_root = tmp_path / "proc"
    _write_process(proc_root, 123, run_dir, start_ticks=222)
    assert isinstance(
        write_owner(
            runtime_root,
            ProcessOwner(
                run_dir=run_dir.resolve(),
                pid=123,
                start_ticks=111,
                command="resume",
            ),
        ),
        Ok,
    )
    inspector = ProcessInspector(
        runtime_root=runtime_root,
        proc_root=proc_root,
        clock_ticks_per_second=100,
    )

    result = inspector.inspect(run_dir)

    assert isinstance(result, Rejected)
    assert "not the owned training process" in result.reason


def test_portable_inspector_reads_current_process(
    tmp_path: Path,
) -> None:
    inspector = ProcessInspector(
        runtime_root=tmp_path / "control",
        backend="portable",
    )

    result = inspector.inspect_pid(os.getpid())

    assert isinstance(result, Ok)
    process = result.value
    assert process is not None
    assert process.pid == os.getpid()
    assert process.start_ticks > 0
    assert process.started_at_ms > 0
    assert process.argv
    assert process.executable.is_absolute()
    assert process.working_directory.is_absolute()


def _write_process(
    proc_root: Path, pid: int, run_dir: Path, *, start_ticks: int
) -> None:
    proc_root.mkdir(parents=True, exist_ok=True)
    proc_root.joinpath("stat").write_text(
        "cpu 1 2 3 4\nbtime 1000\n", encoding="ascii"
    )
    process_dir = proc_root / str(pid)
    process_dir.mkdir()
    argv = (
        sys.executable,
        "-m",
        "server.training_cli",
        "--run-dir",
        str(run_dir.resolve()),
        "resume",
        "latest.json",
    )
    process_dir.joinpath("cmdline").write_bytes(
        b"\0".join(part.encode("utf-8") for part in argv) + b"\0"
    )
    fields = ["S", "1", "123", "123"] + ["0"] * 15 + [str(start_ticks)]
    process_dir.joinpath("stat").write_text(
        f"{pid} (python worker) {' '.join(fields)}\n",
        encoding="ascii",
    )
    process_dir.joinpath("cwd").symlink_to(proc_root.parent)
    process_dir.joinpath("exe").symlink_to(sys.executable)
