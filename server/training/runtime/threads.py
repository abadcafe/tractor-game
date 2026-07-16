"""Torch thread configuration for training processes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from server.foundation import result as _result


@dataclass(frozen=True, slots=True)
class TorchThreadStatus:
    """Observed torch thread counts after applying a config."""

    requested_num_threads: int | None
    requested_num_interop_threads: int | None
    active_num_threads: int
    active_num_interop_threads: int


def apply_worker_torch_thread_config() -> (
    _result.Ok[TorchThreadStatus] | _result.Rejected
):
    """Apply the fixed one-thread policy for every worker process."""
    return apply_torch_thread_config(
        num_threads=1,
        num_interop_threads=1,
    )


def apply_torch_thread_config(
    *,
    num_threads: int | None,
    num_interop_threads: int | None,
) -> _result.Ok[TorchThreadStatus] | _result.Rejected:
    """Apply torch intra-op and inter-op thread counts."""
    assert num_threads is None or num_threads > 0
    assert num_interop_threads is None or num_interop_threads > 0
    try:
        if (
            num_threads is not None
            and num_threads != torch.get_num_threads()
        ):
            torch.set_num_threads(num_threads)
        if (
            num_interop_threads is not None
            and num_interop_threads != torch.get_num_interop_threads()
        ):
            torch.set_num_interop_threads(num_interop_threads)
    except RuntimeError:
        return _result.Rejected(
            reason=(
                "torch thread configuration must be applied before use"
            )
        )
    return _result.Ok(
        value=TorchThreadStatus(
            requested_num_threads=num_threads,
            requested_num_interop_threads=num_interop_threads,
            active_num_threads=torch.get_num_threads(),
            active_num_interop_threads=torch.get_num_interop_threads(),
        )
    )
