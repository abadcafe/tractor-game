"""Tests for server logging configuration."""

import logging

from server.web.logging_config import configure_server_logging


def test_server_logger_has_visible_info_handler() -> None:
    configure_server_logging()
    server_logger = logging.getLogger("server")

    assert server_logger.isEnabledFor(logging.INFO)
    assert any(
        handler.level <= logging.INFO
        for handler in server_logger.handlers
    )
    assert server_logger.propagate is False
