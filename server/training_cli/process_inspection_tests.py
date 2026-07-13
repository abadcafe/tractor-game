"""Black-box tests for Linux training process inspection."""

import os
import threading
from pathlib import Path

from server.foundation.result import Ok, Rejected
from server.training_cli.process_inspection import ProcessInspector


def test_inspect_missing_pid_returns_none(tmp_path: Path) -> None:
    inspector = ProcessInspector(proc_root=tmp_path / "proc")

    result = inspector.inspect(tmp_path / "run")

    assert isinstance(result, Ok)
    assert result.value is None


def test_inspect_stale_pid_returns_none(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_pid(run_dir, 123)
    inspector = ProcessInspector(proc_root=tmp_path / "proc")

    result = inspector.inspect(run_dir)

    assert isinstance(result, Ok)
    assert result.value is None


def test_inspect_matching_training_process_returns_proc_data(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    proc_root = tmp_path / "proc"
    _write_pid(run_dir, 123)
    _write_process(proc_root, 123, run_dir)
    inspector = ProcessInspector(proc_root=proc_root)

    result = inspector.inspect(run_dir)

    assert isinstance(result, Ok)
    process = result.value
    assert process is not None
    assert process.pid == 123
    assert process.name == "python"
    assert process.kernel_state == "S"
    assert process.process_group_id == 123
    assert process.session_id == 123
    assert process.start_ticks == 98765
    assert process.run_dir == run_dir.resolve()


def test_inspect_uses_stat_state_when_status_snapshot_disagrees(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    proc_root = tmp_path / "proc"
    _write_pid(run_dir, 123)
    _write_process(
        proc_root,
        123,
        run_dir,
        state="S",
        status_state="R",
    )
    inspector = ProcessInspector(proc_root=proc_root)

    result = inspector.inspect(run_dir)

    assert isinstance(result, Ok)
    process = result.value
    assert process is not None
    assert process.kernel_state == "S"


def test_inspect_rejects_pid_recycled_during_proc_reads(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    proc_root = tmp_path / "proc"
    _write_pid(run_dir, 123)
    _write_process(proc_root, 123, run_dir)
    stat_path = proc_root / "123" / "stat"
    command_path = proc_root / "123" / "cmdline"
    command_bytes = command_path.read_bytes()
    command_path.unlink()
    os.mkfifo(command_path)
    writer = threading.Thread(
        target=_replace_stat_while_cmdline_is_open,
        args=(
            stat_path,
            command_path,
            _stat_text(123, state="S", start_ticks=123_456),
            command_bytes,
        ),
        daemon=True,
    )
    writer.start()
    inspector = ProcessInspector(proc_root=proc_root)

    result = inspector.inspect(run_dir)
    writer.join(timeout=1.0)

    assert not writer.is_alive()
    assert isinstance(result, Rejected)
    assert "process lifetime changed" in result.reason


def test_inspect_rejects_pid_owned_by_other_process(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    proc_root = tmp_path / "proc"
    _write_pid(run_dir, 123)
    _write_process(
        proc_root,
        123,
        run_dir,
        argv=("/usr/bin/python", "-c", "import time; time.sleep(10)"),
    )
    inspector = ProcessInspector(proc_root=proc_root)

    result = inspector.inspect(run_dir)

    assert isinstance(result, Rejected)
    assert "not the managed training process" in result.reason


def test_inspect_rejects_short_training_init_process(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    proc_root = tmp_path / "proc"
    _write_pid(run_dir, 123)
    _write_process(
        proc_root,
        123,
        run_dir,
        argv=(
            "/usr/bin/python",
            "-m",
            "server.training_cli",
            "--run-dir",
            str(run_dir),
            "init",
        ),
    )
    inspector = ProcessInspector(proc_root=proc_root)

    result = inspector.inspect(run_dir)

    assert isinstance(result, Rejected)
    assert "not the managed training process" in result.reason


def test_inspect_rejects_training_for_different_directory(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    other_dir = tmp_path / "other"
    proc_root = tmp_path / "proc"
    _write_pid(run_dir, 123)
    _write_process(proc_root, 123, other_dir)
    inspector = ProcessInspector(proc_root=proc_root)

    result = inspector.inspect(run_dir)

    assert isinstance(result, Rejected)
    assert "not the managed training process" in result.reason


def test_inspect_zombie_returns_none(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    proc_root = tmp_path / "proc"
    _write_pid(run_dir, 123)
    _write_process(proc_root, 123, run_dir, state="Z")
    inspector = ProcessInspector(proc_root=proc_root)

    result = inspector.inspect(run_dir)

    assert isinstance(result, Ok)
    assert result.value is None


def _write_process(
    proc_root: Path,
    pid: int,
    run_dir: Path,
    *,
    argv: tuple[str, ...] | None = None,
    state: str = "S",
    status_state: str | None = None,
) -> None:
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True)
    command = argv or (
        "/usr/bin/python",
        "-m",
        "server.training_cli",
        "--run-dir",
        str(run_dir.resolve()),
        "resume",
        "latest.json",
        "--max-samples",
        "100",
    )
    process_dir.joinpath("cmdline").write_bytes(
        b"\0".join(part.encode("utf-8") for part in command) + b"\0"
    )
    process_dir.joinpath("status").write_text(
        f"Name:\tpython\nState:\t{status_state or state} (sleeping)\n",
        encoding="utf-8",
    )
    process_dir.joinpath("stat").write_text(
        _stat_text(pid, state=state, start_ticks=98_765),
        encoding="ascii",
    )
    process_dir.joinpath("cwd").symlink_to(proc_root.parent)
    process_dir.joinpath("exe").symlink_to("/usr/bin/python")


def _write_pid(run_dir: Path, pid: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir.joinpath("training.pid").write_text(
        f"{pid}\n", encoding="ascii"
    )


def _stat_text(pid: int, *, state: str, start_ticks: int) -> str:
    fields = (
        [state, "1", "123", "123"] + ["0"] * 15 + [str(start_ticks)]
    )
    return f"{pid} (python worker) {' '.join(fields)}\n"


def _replace_stat_while_cmdline_is_open(
    stat_path: Path,
    command_path: Path,
    final_stat: str,
    command_bytes: bytes,
) -> None:
    with command_path.open("wb") as command_stream:
        stat_path.write_text(final_stat, encoding="ascii")
        command_stream.write(command_bytes)
