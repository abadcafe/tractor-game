"""Black-box tests for PID-file training process control."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from server.foundation.result import Ok, Rejected
from server.training_control.process_control import (
    TrainingProcessControl,
)
from server.training_control.process_inspection import (
    pid_file_path,
    read_training_pid,
)


@pytest.mark.asyncio
async def test_initialize_is_synchronous_and_never_writes_pid(
    tmp_path: Path,
) -> None:
    _write_fixture_cli(tmp_path)
    run_dir = tmp_path / "run"
    control = TrainingProcessControl()

    result = await control.initialize(
        run_dir=run_dir,
        command=_command(run_dir, "init"),
        working_directory=tmp_path,
    )

    assert isinstance(result, Ok)
    assert result.value.run_dir == run_dir.resolve()
    assert not pid_file_path(run_dir).exists()


@pytest.mark.asyncio
async def test_initialize_does_not_validate_or_replace_live_pid(
    tmp_path: Path,
) -> None:
    _write_fixture_cli(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pid_file_path(run_dir).write_text(
        f"{os.getpid()}\n", encoding="ascii"
    )
    control = TrainingProcessControl()

    result = await control.initialize(
        run_dir=run_dir,
        command=_command(run_dir, "init"),
        working_directory=tmp_path,
    )

    assert isinstance(result, Ok)
    assert pid_file_path(run_dir).read_text(encoding="ascii") == (
        f"{os.getpid()}\n"
    )


@pytest.mark.asyncio
async def test_resume_writes_pid_and_survives_controller_restart(
    tmp_path: Path,
) -> None:
    _write_fixture_cli(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    control = TrainingProcessControl()

    started = await control.resume(
        run_dir=run_dir,
        command=_command(run_dir, "resume", "latest.json"),
        working_directory=tmp_path,
    )

    assert isinstance(started, Ok)
    pid_result = read_training_pid(run_dir)
    assert isinstance(pid_result, Ok)
    pid = pid_result.value
    assert pid is not None
    await control.close()
    restarted = TrainingProcessControl()
    inspected = await restarted.inspect(run_dir)
    assert isinstance(inspected, Ok)
    process = inspected.value.process
    assert process is not None
    assert process.pid == pid

    stopped = await restarted.stop(run_dir=run_dir, timeout_seconds=2.0)
    assert isinstance(stopped, Ok)
    assert stopped.value.forced is False
    assert not pid_file_path(run_dir).exists()
    await restarted.close()


@pytest.mark.asyncio
async def test_watch_observes_pid_file_lifecycle_without_revision(
    tmp_path: Path,
) -> None:
    _write_fixture_cli(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    control = TrainingProcessControl()
    snapshots = control.watch(run_dir)

    initial = await anext(snapshots)
    assert isinstance(initial, Ok)
    assert initial.value.process is None
    started = await control.resume(
        run_dir=run_dir,
        command=_command(run_dir, "resume", "latest.json"),
        working_directory=tmp_path,
    )
    assert isinstance(started, Ok)
    running = await asyncio.wait_for(anext(snapshots), timeout=2.0)
    assert isinstance(running, Ok)
    assert running.value.process is not None
    stopped = await control.stop(run_dir=run_dir, timeout_seconds=2.0)
    assert isinstance(stopped, Ok)
    final = await asyncio.wait_for(anext(snapshots), timeout=2.0)
    assert isinstance(final, Ok)
    assert final.value.process is None

    await snapshots.aclose()
    await control.close()


@pytest.mark.asyncio
async def test_concurrent_resume_starts_exactly_one_process(
    tmp_path: Path,
) -> None:
    _write_fixture_cli(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    control = TrainingProcessControl()
    command = _command(run_dir, "resume", "latest.json")

    first, second = await asyncio.gather(
        control.resume(
            run_dir=run_dir,
            command=command,
            working_directory=tmp_path,
        ),
        control.resume(
            run_dir=run_dir,
            command=command,
            working_directory=tmp_path,
        ),
    )

    assert (
        sum(isinstance(result, Ok) for result in (first, second)) == 1
    )
    assert (
        sum(isinstance(result, Rejected) for result in (first, second))
        == 1
    )
    stopped = await control.stop(run_dir=run_dir, timeout_seconds=2.0)
    assert isinstance(stopped, Ok)
    await control.close()


@pytest.mark.asyncio
async def test_live_unrelated_pid_rejects_resume(
    tmp_path: Path,
) -> None:
    _write_fixture_cli(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pid_file_path(run_dir).write_text(
        f"{os.getpid()}\n", encoding="ascii"
    )
    control = TrainingProcessControl()

    result = await control.resume(
        run_dir=run_dir,
        command=_command(run_dir, "resume", "latest.json"),
        working_directory=tmp_path,
    )

    assert isinstance(result, Rejected)
    assert f"PID {os.getpid()}" in result.reason


@pytest.mark.asyncio
async def test_malformed_pid_is_overwritten_by_resume(
    tmp_path: Path,
) -> None:
    _write_fixture_cli(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pid_file_path(run_dir).write_text("invalid\n", encoding="ascii")
    control = TrainingProcessControl()

    started = await control.resume(
        run_dir=run_dir,
        command=_command(run_dir, "resume", "latest.json"),
        working_directory=tmp_path,
    )

    assert isinstance(started, Ok)
    pid_result = read_training_pid(run_dir)
    assert isinstance(pid_result, Ok)
    assert pid_result.value is not None
    stopped = await control.stop(run_dir=run_dir, timeout_seconds=2.0)
    assert isinstance(stopped, Ok)
    await control.close()


@pytest.mark.asyncio
async def test_natural_exit_removes_matching_pid_file(
    tmp_path: Path,
) -> None:
    _write_fixture_cli(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    control = TrainingProcessControl()

    started = await control.resume(
        run_dir=run_dir,
        command=(
            *_command(run_dir, "resume", "latest.json"),
            "--fixture-fail",
        ),
        working_directory=tmp_path,
    )

    assert isinstance(started, Ok)
    await _wait_for_pid_file_removal(run_dir)
    inspected = await control.inspect(run_dir)
    assert isinstance(inspected, Ok)
    assert inspected.value.process is None
    assert "fixture failure" in run_dir.joinpath(
        "training-cli.log"
    ).read_text(encoding="utf-8")
    await control.close()


@pytest.mark.asyncio
async def test_stop_forces_process_that_ignores_sigterm(
    tmp_path: Path,
) -> None:
    _write_fixture_cli(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    control = TrainingProcessControl()
    started = await control.resume(
        run_dir=run_dir,
        command=(
            *_command(run_dir, "resume", "latest.json"),
            "--fixture-ignore-term",
        ),
        working_directory=tmp_path,
    )
    assert isinstance(started, Ok)
    await _wait_for_path(run_dir / "fixture-ready")

    stopped = await control.stop(run_dir=run_dir, timeout_seconds=0.1)

    assert isinstance(stopped, Ok)
    assert stopped.value.forced is True
    assert not pid_file_path(run_dir).exists()
    await control.close()


@pytest.mark.asyncio
async def test_stop_removes_stale_pid_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pid_file_path(run_dir).write_text("2147483647\n", encoding="ascii")
    control = TrainingProcessControl()

    stopped = await control.stop(run_dir=run_dir, timeout_seconds=1.0)

    assert isinstance(stopped, Ok)
    assert stopped.value.forced is False
    assert not pid_file_path(run_dir).exists()


async def _wait_for_pid_file_removal(run_dir: Path) -> None:
    for _attempt in range(200):
        if not pid_file_path(run_dir).exists():
            return
        await asyncio.sleep(0.01)
    assert False, "training PID file was not removed"


async def _wait_for_path(path: Path) -> None:
    for _attempt in range(200):
        if path.exists():
            return
        await asyncio.sleep(0.01)
    assert False, f"fixture path was not created: {path}"


def _command(
    run_dir: Path, command: str, *arguments: str
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "server.training_cli",
        "--run-dir",
        str(run_dir.resolve()),
        command,
        *arguments,
    )


def _write_fixture_cli(tmp_path: Path) -> None:
    package = tmp_path / "server" / "training_cli"
    package.mkdir(parents=True)
    package.parent.joinpath("__init__.py").write_text(
        "", encoding="ascii"
    )
    package.joinpath("__init__.py").write_text("", encoding="ascii")
    package.joinpath("__main__.py").write_text(
        "import pathlib, signal, sys, time\n"
        "args = sys.argv[1:]\n"
        "run_dir = pathlib.Path(args[args.index('--run-dir') + 1])\n"
        "run_dir.mkdir(parents=True, exist_ok=True)\n"
        "stopped = False\n"
        "def stop(_signal, _frame):\n"
        "    global stopped\n"
        "    stopped = True\n"
        "if '--fixture-ignore-term' in args:\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "else:\n"
        "    signal.signal(signal.SIGTERM, stop)\n"
        "run_dir.joinpath('fixture-ready').write_text('ready')\n"
        "if 'resume' in args:\n"
        "    if '--fixture-fail' in args:\n"
        "        time.sleep(0.1)\n"
        "        print('fixture failure', file=sys.stderr)\n"
        "        raise SystemExit(2)\n"
        "    while not stopped:\n"
        "        time.sleep(0.01)\n",
        encoding="ascii",
    )
