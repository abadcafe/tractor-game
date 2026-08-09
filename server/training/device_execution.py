"""Device execution policy shared by inference and PPO training."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DeviceExecutionPolicy:
    """Numerical execution mode selected from device capabilities."""

    device: torch.device
    uses_bfloat16: bool

    def __post_init__(self) -> None:
        assert self.uses_bfloat16 is False or self.device.type == "cuda"

    @classmethod
    def for_device(cls, device: torch.device) -> DeviceExecutionPolicy:
        """Select the fastest stable execution mode for a device."""
        return cls(
            device=device,
            uses_bfloat16=(
                device.type == "cuda" and torch.cuda.is_bf16_supported()
            ),
        )

    def autocast(self) -> AbstractContextManager[object]:
        """Return the model-compute autocast context."""
        if not self.uses_bfloat16:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )


def configure_accelerator_math(device: torch.device) -> None:
    """Configure process-wide accelerator math behavior."""
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")


__all__ = ("DeviceExecutionPolicy", "configure_accelerator_math")
