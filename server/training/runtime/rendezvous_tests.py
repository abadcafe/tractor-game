"""Tests for torch distributed rendezvous creation."""

from __future__ import annotations

from pathlib import Path

import pytest

from server.result import Ok, Rejected
from server.training.runtime.rendezvous import create_file_rendezvous


def test_create_file_rendezvous_uses_absolute_file_uri(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    created = create_file_rendezvous(Path("training-runs/manual"))

    assert isinstance(created, Ok)
    assert created.value.path.is_absolute()
    assert created.value.path.parent == (
        tmp_path / "training-runs" / "manual" / "runtime"
    )
    assert created.value.init_method == (
        f"file://{created.value.path.as_posix()}"
    )


def test_create_file_rendezvous_preserves_unescaped_posix_path(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run with spaces"

    created = create_file_rendezvous(run_dir)

    assert isinstance(created, Ok)
    assert "run with spaces" in created.value.init_method
    assert "%20" not in created.value.init_method
    assert created.value.init_method == (
        f"file://{created.value.path.as_posix()}"
    )


def test_create_file_rendezvous_rejects_runtime_dir_create_failure(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.write_text("not a directory", encoding="utf-8")

    created = create_file_rendezvous(run_dir)

    assert isinstance(created, Rejected)
    assert "failed to create distributed runtime dir" in created.reason
