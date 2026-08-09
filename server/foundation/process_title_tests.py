"""Black-box tests for operating-system process titles."""

from __future__ import annotations

import subprocess
import sys

import psutil


def test_set_process_title_replaces_python_command_line() -> None:
    title = "tractor-title-test"
    source = (
        "from server.foundation.process_title "
        "import set_process_title; "
        "import time; "
        f"set_process_title('{title}'); "
        "print('ready', flush=True); "
        "time.sleep(30)"
    )
    process = subprocess.Popen(
        (sys.executable, "-c", source),
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        command_line = psutil.Process(process.pid).cmdline()
        assert command_line[0] == title
        assert all(not argument for argument in command_line[1:])
    finally:
        if process.poll() is None:
            process.terminate()
        _ = process.wait(timeout=5.0)
