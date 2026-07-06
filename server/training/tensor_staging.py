"""Backend-specific tensor staging for training inference inputs."""

from __future__ import annotations

import torch
from torch import Tensor


def staged_tensor(
    values: object, *, dtype: torch.dtype, device: torch.device
) -> Tensor:
    """Create a tensor through the runtime's staging boundary."""
    if device.type != "cuda":
        return torch.tensor(values, dtype=dtype, device=device)
    cpu_tensor = torch.tensor(
        values,
        dtype=dtype,
        device=torch.device("cpu"),
        pin_memory=True,
    )
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        device_tensor = cpu_tensor.to(device=device, non_blocking=True)
    event = torch.cuda.Event()
    event.record(stream)
    torch.cuda.current_stream(device).wait_event(event)
    return device_tensor
