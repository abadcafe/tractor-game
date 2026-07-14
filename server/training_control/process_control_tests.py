"""Black-box tests for the unified training process lifecycle."""

import asyncio
import sys
from pathlib import Path

import pytest

from server.foundation.result import Ok, Rejected
from server.training_control.process_control import (
    TrainingProcessControl,
)
from server.training_control.process_owner import owner_path


@pytest.mark.asyncio
async def test_resume_waits_for_ready_and_survives_controller_restart(
    tmp_path: Path,
) -> None:
    _write_fixture_cli(tmp_path)
    run_dir = tmp_path / "run"
    runtime_root = tmp_path / "control"
    command = _command(run_dir, "resume", "latest.json")
    control = TrainingProcessControl(
        runtime_root=runtime_root, startup_timeout_seconds=2.0
    )

    started = await control.resume(
        run_dir=run_dir,
        command=command,
        working_directory=tmp_path,
    )

    assert isinstance(started, Ok)
    process = started.value.process
    assert process is not None
    assert process.command == "resume"
    assert process.ready is True
    assert process.started_at_ms > 0
    assert owner_path(runtime_root, run_dir).is_file()
    assert not run_dir.joinpath("training.pid").exists()

    restarted = TrainingProcessControl(runtime_root=runtime_root)
    inspected = await restarted.inspect(run_dir)
    assert isinstance(inspected, Ok)
    assert inspected.value.process == process

    stopped = await restarted.stop(run_dir=run_dir, timeout_seconds=2.0)
    assert isinstance(stopped, Ok)
    assert stopped.value.forced is False
    assert stopped.value.process is None
    assert not owner_path(runtime_root, run_dir).exists()
    await control.close()


@pytest.mark.asyncio
async def test_resume_returns_original_handshake_error(
    tmp_path: Path,
) -> None:
    _write_fixture_cli(tmp_path)
    run_dir = tmp_path / "run"
    runtime_root = tmp_path / "control"
    control = TrainingProcessControl(
        runtime_root=runtime_root, startup_timeout_seconds=2.0
    )

    result = await control.resume(
        run_dir=run_dir,
        command=(
            *_command(run_dir, "resume", "latest.json"),
            "--fixture-fail",
        ),
        working_directory=tmp_path,
    )

    assert isinstance(result, Rejected)
    assert result.reason == "checkpoint metadata is corrupt"
    inspected = await control.inspect(run_dir)
    assert isinstance(inspected, Ok)
    assert inspected.value.process is None


@pytest.mark.asyncio
async def test_initialize_is_visible_and_stoppable(
    tmp_path: Path,
) -> None:
    _write_fixture_cli(tmp_path)
    run_dir = tmp_path / "run"
    runtime_root = tmp_path / "control"
    control = TrainingProcessControl(runtime_root=runtime_root)
    initialization = asyncio.create_task(
        control.initialize(
            run_dir=run_dir,
            command=(*_command(run_dir, "init"), "--fixture-wait"),
            working_directory=tmp_path,
        )
    )

    process = None
    for _attempt in range(100):
        inspected = await control.inspect(run_dir)
        assert isinstance(inspected, Ok)
        process = inspected.value.process
        if process is not None:
            break
        await asyncio.sleep(0.01)
    assert process is not None
    assert process.command == "initialize"
    assert process.ready is False

    stopped = await control.stop(run_dir=run_dir, timeout_seconds=2.0)
    initialized = await initialization

    assert isinstance(stopped, Ok)
    assert isinstance(initialized, Rejected)
    assert not owner_path(runtime_root, run_dir).exists()


@pytest.mark.asyncio
async def test_immediate_initialize_success_is_not_a_start_failure(
    tmp_path: Path,
) -> None:
    _write_fixture_cli(tmp_path)
    run_dir = tmp_path / "run"
    control = TrainingProcessControl(runtime_root=tmp_path / "control")

    result = await control.initialize(
        run_dir=run_dir,
        command=_command(run_dir, "init"),
        working_directory=tmp_path,
    )

    assert isinstance(result, Ok)
    assert result.value.run_dir == run_dir.resolve()


@pytest.mark.asyncio
async def test_resume_publishes_starting_before_ready(
    tmp_path: Path,
) -> None:
    _write_fixture_cli(tmp_path)
    run_dir = tmp_path / "run"
    control = TrainingProcessControl(
        runtime_root=tmp_path / "control",
        startup_timeout_seconds=2.0,
    )
    snapshots = control.watch(run_dir, after_revision=0)
    resume = asyncio.create_task(
        control.resume(
            run_dir=run_dir,
            command=(
                *_command(run_dir, "resume", "latest.json"),
                "--fixture-ready-delay",
            ),
            working_directory=tmp_path,
        )
    )

    starting = await asyncio.wait_for(anext(snapshots), timeout=2.0)
    assert isinstance(starting, Ok)
    assert starting.value.process is not None
    assert starting.value.process.ready is False
    started = await resume
    assert isinstance(started, Ok)
    assert started.value.process is not None
    assert started.value.process.ready is True
    assert started.value.revision > starting.value.revision

    await snapshots.aclose()
    await control.stop(run_dir=run_dir, timeout_seconds=2.0)
    await control.close()


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
        "import json, os, signal, sys, time\n"
        "args = sys.argv[1:]\n"
        "stopped = False\n"
        "def stop(_signal, _frame):\n"
        "    global stopped\n"
        "    stopped = True\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "if 'resume' in args:\n"
        "    fd = int(args[args.index('--ready-fd') + 1])\n"
        "    if '--fixture-fail' in args:\n"
        "        message = {'type': 'error', "
        "'error': 'checkpoint metadata is corrupt'}\n"
        "        os.write(fd, json.dumps(message).encode() + b'\\n')\n"
        "        os.close(fd)\n"
        "        raise SystemExit(2)\n"
        "    if '--fixture-ready-delay' in args:\n"
        "        time.sleep(0.1)\n"
        '    os.write(fd, b\'{"type": "ready"}\\n\')\n'
        "    os.close(fd)\n"
        "    while not stopped:\n"
        "        time.sleep(0.01)\n"
        "elif '--fixture-wait' in args:\n"
        "    while not stopped:\n"
        "        time.sleep(0.01)\n"
        "    raise SystemExit(2)\n",
        encoding="utf-8",
    )
