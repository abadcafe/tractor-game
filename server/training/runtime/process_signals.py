"""Signal policy for training child processes."""

from __future__ import annotations

import signal
from multiprocessing.process import BaseProcess


def ignore_terminal_interrupt_in_child_process() -> None:
    """Let the coordinator own terminal Ctrl-C handling."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def start_child_process_ignoring_terminal_interrupt(
    process: BaseProcess,
) -> None:
    """Start one child with SIGINT ignored from spawn bootstrap."""
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        process.start()
    finally:
        signal.signal(signal.SIGINT, previous)
