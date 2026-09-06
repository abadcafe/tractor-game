"""Shared stderr logging for standalone server processes."""

from __future__ import annotations

import logging
import sys

_HANDLER_NAME = "tractor-runtime-stderr"
_LOG_FORMAT = (
    "%(asctime)s %(levelname)s pid=%(process)d [%(name)s] %(message)s"
)


def configure_stderr_logging() -> None:
    """Route INFO and higher server records to process stderr once."""
    logger = logging.getLogger("server")
    if any(
        handler.get_name() == _HANDLER_NAME
        for handler in logger.handlers
    ):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False


__all__ = ("configure_stderr_logging",)
