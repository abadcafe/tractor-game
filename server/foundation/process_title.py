"""Operating-system process title ownership."""

from __future__ import annotations

from setproctitle import setproctitle


def set_process_title(title: str) -> None:
    """Replace the current process title visible to system tools."""
    assert title
    assert "\x00" not in title
    setproctitle(title)


__all__ = ("set_process_title",)
