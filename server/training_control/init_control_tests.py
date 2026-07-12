"""Black-box tests for short training initialization control."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from server.foundation.result import Ok, Rejected
from server.training_control.init_control import TrainingInitControl


@pytest.mark.asyncio
async def test_initialize_waits_for_successful_command(
    tmp_path: Path,
) -> None:
    result = await TrainingInitControl().initialize(
        run_dir=tmp_path / "run",
        command=(sys.executable, "-c", "print('initialized')"),
        working_directory=tmp_path,
    )

    assert isinstance(result, Ok)
    assert result.value.run_dir == (tmp_path / "run").resolve()
    assert result.value.checkpoint_path == (
        tmp_path / "run" / "checkpoints" / "latest.json"
    )


@pytest.mark.asyncio
async def test_initialize_returns_command_error(tmp_path: Path) -> None:
    result = await TrainingInitControl().initialize(
        run_dir=tmp_path / "run",
        command=(
            sys.executable,
            "-c",
            "import sys; "
            "print('invalid config', file=sys.stderr); "
            "sys.exit(2)",
        ),
        working_directory=tmp_path,
    )

    assert isinstance(result, Rejected)
    assert result.reason == "invalid config"
