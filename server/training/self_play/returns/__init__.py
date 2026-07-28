"""Worker-side return target construction."""

from server.training.self_play.returns.commit import (
    ReturnCommit,
    terminal_return_commit,
)

__all__ = (
    "ReturnCommit",
    "terminal_return_commit",
)
