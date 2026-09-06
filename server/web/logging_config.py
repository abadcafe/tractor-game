"""Own the web process logging handlers and their lifecycle."""

from __future__ import annotations

import logging
import os
import sys
from io import TextIOWrapper
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import IO, final, override

_CONSOLE_HANDLER_NAME = "tractor-server-stderr"
_FILE_HANDLER_NAME = "tractor-server-file"
_LOG_FORMAT = (
    "%(asctime)s %(levelname)s pid=%(process)d [%(name)s] %(message)s"
)
_ROTATION_BYTES = 64 * 1024 * 1024
_ROTATION_BACKUPS = 5


@final
class _ServerFileHandler(RotatingFileHandler):
    """Keep the configured public file mode after every rollover."""

    @override
    def _open(self) -> TextIOWrapper:
        stream = super()._open()
        os.chmod(self.baseFilename, 0o644)
        return stream


def configure_server_logging(
    *,
    log_path: Path | None = None,
    stderr: IO[str] | None = None,
) -> None:
    """Install exactly one console and one rotating file handler."""
    server_logger = logging.getLogger("server")
    if _owned_handlers(server_logger):
        return
    resolved = (log_path or Path("logs/server.log")).resolve()
    resolved.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    os.chmod(resolved.parent, 0o755)
    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler(
        sys.stderr if stderr is None else stderr
    )
    console.set_name(_CONSOLE_HANDLER_NAME)
    console.setLevel(_stderr_level())
    console.setFormatter(formatter)

    file_handler = _ServerFileHandler(
        resolved,
        maxBytes=_ROTATION_BYTES,
        backupCount=_ROTATION_BACKUPS,
        encoding="utf-8",
    )
    file_handler.set_name(_FILE_HANDLER_NAME)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    server_logger.setLevel(logging.DEBUG)
    server_logger.addHandler(console)
    server_logger.addHandler(file_handler)
    server_logger.propagate = False


def shutdown_server_logging() -> None:
    """Flush, close, and remove only handlers owned by this module."""
    server_logger = logging.getLogger("server")
    for handler in _owned_handlers(server_logger):
        server_logger.removeHandler(handler)
        handler.flush()
        handler.close()


def _owned_handlers(
    logger: logging.Logger,
) -> tuple[logging.Handler, ...]:
    names = frozenset((_CONSOLE_HANDLER_NAME, _FILE_HANDLER_NAME))
    return tuple(
        handler
        for handler in logger.handlers
        if handler.get_name() in names
    )


def _stderr_level() -> int:
    value = os.environ.get("TRACTOR_STDERR_LOG_LEVEL", "INFO")
    normalized = value.strip().upper()
    if normalized == "DEBUG":
        return logging.DEBUG
    if normalized == "INFO":
        return logging.INFO
    if normalized == "WARNING":
        return logging.WARNING
    if normalized == "ERROR":
        return logging.ERROR
    if normalized == "CRITICAL":
        return logging.CRITICAL
    raise ValueError(
        "TRACTOR_STDERR_LOG_LEVEL must be DEBUG, INFO, WARNING, "
        + "ERROR, or CRITICAL"
    )


__all__ = (
    "configure_server_logging",
    "shutdown_server_logging",
)
