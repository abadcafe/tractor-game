"""Black-box tests for the training control PID file."""

from pathlib import Path

from server.foundation.result import Ok, Rejected
from server.training_control.pid_file import (
    read_pid,
    remove_pid_if_matches,
    write_pid,
)


def test_read_pid_missing_file_returns_none(tmp_path: Path) -> None:
    result = read_pid(tmp_path)

    assert isinstance(result, Ok)
    assert result.value is None


def test_read_pid_returns_positive_pid(tmp_path: Path) -> None:
    (tmp_path / "training.pid").write_text("123\n", encoding="ascii")

    result = read_pid(tmp_path)

    assert isinstance(result, Ok)
    assert result.value == 123


def test_read_pid_rejects_invalid_content(tmp_path: Path) -> None:
    (tmp_path / "training.pid").write_text(
        "not-a-pid\n", encoding="ascii"
    )

    result = read_pid(tmp_path)

    assert isinstance(result, Rejected)
    assert "invalid" in result.reason


def test_read_pid_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("123\n", encoding="ascii")
    (tmp_path / "training.pid").symlink_to(target)

    result = read_pid(tmp_path)

    assert isinstance(result, Rejected)
    assert "symlink" in result.reason


def test_write_pid_creates_exclusive_file(tmp_path: Path) -> None:
    result = write_pid(tmp_path, 456)

    assert isinstance(result, Ok)
    assert (tmp_path / "training.pid").read_text(
        encoding="ascii"
    ) == "456\n"


def test_write_pid_rejects_existing_file(tmp_path: Path) -> None:
    (tmp_path / "training.pid").write_text("123\n", encoding="ascii")

    result = write_pid(tmp_path, 456)

    assert isinstance(result, Rejected)
    assert (tmp_path / "training.pid").read_text(
        encoding="ascii"
    ) == "123\n"


def test_remove_pid_if_matches_removes_owned_file(
    tmp_path: Path,
) -> None:
    (tmp_path / "training.pid").write_text("123\n", encoding="ascii")

    result = remove_pid_if_matches(tmp_path, 123)

    assert isinstance(result, Ok)
    assert not (tmp_path / "training.pid").exists()


def test_remove_pid_if_matches_preserves_replaced_file(
    tmp_path: Path,
) -> None:
    (tmp_path / "training.pid").write_text("456\n", encoding="ascii")

    result = remove_pid_if_matches(tmp_path, 123)

    assert isinstance(result, Ok)
    assert (tmp_path / "training.pid").exists()
