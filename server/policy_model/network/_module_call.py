"""Typed PyTorch module-call boundaries.

PyTorch types ``nn.Module.__call__`` dynamically even when ``forward``
is precise. These helpers preserve hook semantics while containing that
incomplete third-party type boundary in one place.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor


def call_tensor[**P](
    module: Callable[P, Tensor],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Tensor:
    """Call a module whose runtime contract returns one tensor."""
    return module(*args, **kwargs)


def call_attention[**P](
    module: Callable[P, tuple[Tensor, Tensor | None]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[Tensor, Tensor | None]:
    """Call an attention module through its two-result contract."""
    return module(*args, **kwargs)


__all__ = ("call_attention", "call_tensor")
