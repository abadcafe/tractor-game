"""Tests for external exact-lifetime process owners."""

from pathlib import Path

from server.foundation.result import Ok
from server.training_control.process_owner import (
    ProcessOwner,
    mark_owner_ready,
    owner_path,
    read_owner,
    remove_owner_if_matches,
    write_owner,
)


def test_owner_lives_outside_run_and_records_exact_identity(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    runtime_root = tmp_path / "runtime"
    owner = ProcessOwner(
        run_dir=run_dir.resolve(),
        pid=123,
        start_ticks=456,
        command="resume",
        ready=False,
    )

    written = write_owner(runtime_root, owner)
    read = read_owner(runtime_root, run_dir)

    assert isinstance(written, Ok)
    assert isinstance(read, Ok)
    assert read.value == owner
    assert owner_path(runtime_root, run_dir).is_file()
    assert not run_dir.joinpath("training.pid").exists()


def test_ready_and_remove_require_same_process_lifetime(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    runtime_root = tmp_path / "runtime"
    owner = ProcessOwner(
        run_dir=run_dir.resolve(),
        pid=123,
        start_ticks=456,
        command="resume",
        ready=False,
    )
    assert isinstance(write_owner(runtime_root, owner), Ok)

    unchanged = remove_owner_if_matches(
        runtime_root, run_dir, pid=123, start_ticks=999
    )
    ready = mark_owner_ready(
        runtime_root, run_dir, pid=123, start_ticks=456
    )
    removed = remove_owner_if_matches(
        runtime_root, run_dir, pid=123, start_ticks=456
    )

    assert isinstance(unchanged, Ok) and unchanged.value is False
    assert isinstance(ready, Ok)
    assert isinstance(removed, Ok) and removed.value is True
    assert not owner_path(runtime_root, run_dir).exists()
