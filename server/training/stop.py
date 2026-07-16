"""Signal-backed cooperative stop requests for the coordinator."""

from __future__ import annotations

import signal
import threading
from collections.abc import Generator
from contextlib import contextmanager
from types import FrameType


class TrainingStopRequest:
    """Thread-safe cooperative stop flag for training stages."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def request_stop(self) -> None:
        self._event.set()

    def is_requested(self) -> bool:
        return self._event.is_set()


@contextmanager
def training_stop_signals(
    request: TrainingStopRequest,
) -> Generator[None, None, None]:
    """Translate SIGINT and SIGTERM into a cooperative stop request."""

    def handle_stop_signal(
        _signum: int, _frame: FrameType | None
    ) -> None:
        request.request_stop()

    previous_sigint = signal.signal(signal.SIGINT, handle_stop_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, handle_stop_signal)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
