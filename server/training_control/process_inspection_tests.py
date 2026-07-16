"""Tests for PID-file-backed process status."""

from __future__ import annotations

import os
from pathlib import Path

from server.foundation.result import Ok, Rejected
from server.training_control.process_inspection import (
    ProcessInspector,
    pid_file_path,
    read_training_pid,
    remove_training_pid_if_matches,
    write_training_pid,
)


def test_missing_pid_file_is_stopped(tmp_path: Path) -> None:
    result = ProcessInspector().inspect(tmp_path)

    assert isinstance(result, Ok)
    assert result.value.process is None


def test_live_pid_exposes_actual_process_details(
    tmp_path: Path,
) -> None:
    pid_file_path(tmp_path).write_text(
        f"{os.getpid()}\n", encoding="ascii"
    )

    result = ProcessInspector().inspect(tmp_path)

    assert isinstance(result, Ok)
    process = result.value.process
    assert process is not None
    assert process.pid == os.getpid()
    inspection = process.inspection
    assert inspection.kind == "details"
    assert inspection.started_at_ms > 0
    assert inspection.argv
    assert inspection.process_group_id > 0
    assert inspection.unix_session_id > 0


def test_malformed_pid_file_is_stopped(tmp_path: Path) -> None:
    path = pid_file_path(tmp_path)
    for content in ("", "not-a-pid\n", "0\n", "-12\n", "1\n2\n"):
        path.write_text(content, encoding="ascii")
        result = ProcessInspector().inspect(tmp_path)
        assert isinstance(result, Ok)
        assert result.value.process is None


def test_dead_pid_file_is_stopped(tmp_path: Path) -> None:
    pid_file_path(tmp_path).write_text("2147483647\n", encoding="ascii")

    result = ProcessInspector().inspect(tmp_path)

    assert isinstance(result, Ok)
    assert result.value.process is None


def test_pid_file_write_and_matching_removal(tmp_path: Path) -> None:
    written = write_training_pid(tmp_path, os.getpid())
    assert isinstance(written, Ok)
    read = read_training_pid(tmp_path)
    assert isinstance(read, Ok)
    assert read.value == os.getpid()

    unchanged = remove_training_pid_if_matches(
        tmp_path, os.getpid() + 1
    )
    assert isinstance(unchanged, Ok)
    assert unchanged.value is False
    removed = remove_training_pid_if_matches(tmp_path, os.getpid())
    assert isinstance(removed, Ok)
    assert removed.value is True
    assert not pid_file_path(tmp_path).exists()


def test_pid_file_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text(f"{os.getpid()}\n", encoding="ascii")
    pid_file_path(tmp_path).symlink_to(target)

    result = read_training_pid(tmp_path)

    assert isinstance(result, Rejected)
