"""Black-box tests for detached training process control."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from server.foundation.result import Ok, Rejected
from server.training_control.pid_file import read_pid, write_pid
from server.training_control.process_control import (
    TrainingProcessControl,
)


@pytest.mark.asyncio
async def test_start_and_stop_detached_training_command(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _initialize_run(run_dir)
    fixture_package = tmp_path / "server" / "training_cli"
    fixture_package.mkdir(parents=True)
    fixture_package.parent.joinpath("__init__.py").write_text(
        "", encoding="ascii"
    )
    fixture_package.joinpath("__init__.py").write_text(
        "", encoding="ascii"
    )
    fixture_package.joinpath("__main__.py").write_text(
        "import signal, time\n"
        "stopped = False\n"
        "def stop(_signal, _frame):\n"
        "    global stopped\n"
        "    stopped = True\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "while not stopped:\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    command = (
        sys.executable,
        "-m",
        "server.training_cli",
        "--run-dir",
        str(run_dir.resolve()),
        "resume",
        "latest.json",
    )
    run_dir.joinpath("stdout.log").write_text(
        "previous session\n", encoding="utf-8"
    )
    control = TrainingProcessControl()

    started = await control.start(
        run_dir=run_dir,
        command=command,
        working_directory=tmp_path,
    )

    assert isinstance(started, Ok)
    assert started.value.run_dir == run_dir.resolve()
    assert isinstance(read_pid(run_dir), Ok)

    restarted_control = TrainingProcessControl()
    recovered = await restarted_control.inspect(run_dir)
    assert isinstance(recovered, Ok)
    assert recovered.value is not None

    stopped = await restarted_control.stop(
        run_dir=run_dir, timeout_seconds=2.0
    )

    assert isinstance(stopped, Ok)
    assert stopped.value.forced is False
    assert "previous session" not in run_dir.joinpath(
        "stdout.log"
    ).read_text(encoding="utf-8")
    pid_result = read_pid(run_dir)
    assert isinstance(pid_result, Ok)
    assert pid_result.value is None


@pytest.mark.asyncio
async def test_stop_missing_process_is_idempotent(
    tmp_path: Path,
) -> None:
    control = TrainingProcessControl()

    result = await control.stop(run_dir=tmp_path, timeout_seconds=0.1)

    assert isinstance(result, Ok)
    assert result.value.forced is False


@pytest.mark.asyncio
async def test_stop_rejects_pid_owned_by_current_test_process(
    tmp_path: Path,
) -> None:
    assert isinstance(write_pid(tmp_path, os.getpid()), Ok)
    control = TrainingProcessControl()

    result = await control.stop(run_dir=tmp_path, timeout_seconds=0.1)

    assert isinstance(result, Rejected)


def _initialize_run(run_dir: Path) -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "server.training_cli",
            "--run-dir",
            str(run_dir),
            "init",
            "--d-model",
            "4",
            "--layers",
            "1",
            "--heads",
            "1",
            "--max-tokens",
            "512",
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
