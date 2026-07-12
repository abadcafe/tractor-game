"""Server logging configuration."""

from __future__ import annotations

import logging

_SERVER_LOG_HANDLER_NAME = "tractor-server-stderr"


def configure_server_logging() -> None:
    """Route project loggers to stderr under uvicorn."""
    server_logger = logging.getLogger("server")
    server_logger.setLevel(logging.INFO)
    if not _has_named_handler(server_logger, _SERVER_LOG_HANDLER_NAME):
        handler = logging.StreamHandler()
        handler.set_name(_SERVER_LOG_HANDLER_NAME)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            )
        )
        server_logger.addHandler(handler)
    server_logger.propagate = False


def _has_named_handler(
    target_logger: logging.Logger, name: str
) -> bool:
    return any(
        handler.get_name() == name for handler in target_logger.handlers
    )
