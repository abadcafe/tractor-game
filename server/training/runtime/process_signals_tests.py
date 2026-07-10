"""Tests for training runtime process signal policy."""

from __future__ import annotations

import multiprocessing as mp
import signal
import subprocess
import sys
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess

from server.training.runtime.process_signals import (
    start_child_process_ignoring_terminal_interrupt,
)


class _StartFailure(Exception):
    """Test-only process start failure."""


class _StartFailureProcess(BaseProcess):
    def start(self) -> None:
        raise _StartFailure


def _send_current_sigint_handler_is_ignored(
    connection: Connection,
) -> None:
    connection.send(signal.getsignal(signal.SIGINT) == signal.SIG_IGN)
    connection.close()


def test_ignore_child_sigint_uses_ignore_handler() -> None:
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import signal\n"
                "from server.training.runtime.process_signals import "
                "ignore_terminal_interrupt_in_child_process\n"
                "ignore_terminal_interrupt_in_child_process()\n"
                "print(signal.getsignal(signal.SIGINT) "
                "== signal.SIG_IGN)\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "True"
    assert completed.stderr == ""


def test_start_child_process_ignores_sigint_and_restores_parent() -> (
    None
):
    context: SpawnContext = mp.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(
        target=_send_current_sigint_handler_is_ignored,
        args=(writer,),
    )
    previous = signal.getsignal(signal.SIGINT)

    try:
        start_child_process_ignoring_terminal_interrupt(process)
        writer.close()

        assert signal.getsignal(signal.SIGINT) == previous
        assert reader.poll(5.0)
        received = reader.recv()
        process.join(timeout=5.0)
        assert received is True
        assert process.exitcode == 0
    finally:
        reader.close()
        writer.close()
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)


def test_start_child_process_restores_sigint_when_start_fails() -> None:
    process = _StartFailureProcess()
    previous = signal.getsignal(signal.SIGINT)
    failed = False

    try:
        start_child_process_ignoring_terminal_interrupt(process)
    except _StartFailure:
        failed = True

    assert failed
    assert signal.getsignal(signal.SIGINT) == previous
