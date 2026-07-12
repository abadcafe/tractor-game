"""Tests for cooperative training stop requests."""

from __future__ import annotations

import signal

from server.training.stop import (
    TrainingStopRequest,
    training_stop_signals,
)


def test_sigterm_requests_stop_and_restores_handler() -> None:
    request = TrainingStopRequest()
    previous = signal.getsignal(signal.SIGTERM)

    with training_stop_signals(request):
        signal.raise_signal(signal.SIGTERM)
        assert request.is_requested()

    assert signal.getsignal(signal.SIGTERM) == previous
