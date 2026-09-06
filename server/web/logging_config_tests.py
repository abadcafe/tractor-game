"""Black-box tests for process-owned server logging."""

from __future__ import annotations

import io
import logging
import stat
from pathlib import Path

from server.web.logging_config import (
    configure_server_logging,
    shutdown_server_logging,
)


def test_configure_writes_info_to_console_and_server_file(
    tmp_path: Path,
) -> None:
    stream = io.StringIO()
    log_path = tmp_path / "logs" / "server.log"
    configure_server_logging(log_path=log_path, stderr=stream)
    logger = logging.getLogger("server.logging_test")

    logger.debug("hidden debug")
    logger.info("visible info")
    shutdown_server_logging()

    assert "visible info" in stream.getvalue()
    contents = log_path.read_text(encoding="utf-8")
    assert "visible info" in contents
    assert "hidden debug" not in contents


def test_configure_uses_public_log_permissions(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "server.log"
    configure_server_logging(
        log_path=log_path,
        stderr=io.StringIO(),
    )
    shutdown_server_logging()

    directory_mode = stat.S_IMODE(log_path.parent.stat().st_mode)
    file_mode = stat.S_IMODE(log_path.stat().st_mode)
    assert directory_mode == 0o755
    assert file_mode == 0o644


def test_configure_is_idempotent(tmp_path: Path) -> None:
    stream = io.StringIO()
    log_path = tmp_path / "logs" / "server.log"
    configure_server_logging(log_path=log_path, stderr=stream)
    configure_server_logging(log_path=log_path, stderr=stream)
    logger = logging.getLogger("server.logging_test")

    logger.info("single record")
    shutdown_server_logging()

    assert stream.getvalue().count("single record") == 1
    assert (
        log_path.read_text(encoding="utf-8").count("single record") == 1
    )
