"""Single-owner synchronous PyTorch execution for inference requests."""

from __future__ import annotations

from pathlib import Path
from typing import final

import torch

from server.foundation.result import Ok, Rejected
from server.policy_model.actions import GeneratedAction
from server.policy_model.actions.decoding import ActionSampler
from server.policy_model.checkpoint import (
    load_policy_checkpoint,
)
from server.policy_model.network import PolicyModel

from ._batches import inference_batch_row_limit
from ._sample import sample_actions
from .contracts import PolicyDecisionRequest


@final
class TorchBatchExecutor:
    """Own one model and all mutable device sampling workspaces."""

    def __init__(
        self,
        *,
        model: PolicyModel,
        device: torch.device,
    ) -> None:
        self._model = model
        self._device = device
        self._max_batch_rows = inference_batch_row_limit(device.type)
        self._sampler: ActionSampler | None = None
        self._sampler_capacity = 0
        _ = self._model.eval()

    @classmethod
    def load(
        cls,
        *,
        checkpoint_path: Path,
        device_name: str,
    ) -> Ok[TorchBatchExecutor] | Rejected:
        """Load exactly the model declared by the current checkpoint."""
        device = _resolve_device(device_name)
        if isinstance(device, Rejected):
            return device
        loaded = load_policy_checkpoint(
            path=checkpoint_path,
            device=device.value,
        )
        if isinstance(loaded, Rejected):
            return loaded
        return Ok(
            cls(
                model=loaded.value.model,
                device=device.value,
            )
        )

    def decide(
        self,
        *,
        requests: tuple[PolicyDecisionRequest, ...],
    ) -> Ok[tuple[GeneratedAction, ...]] | Rejected:
        """Sample one complete action for every policy request."""
        assert requests
        capacity = len(requests)
        with torch.inference_mode():
            return sample_actions(
                model=self._model,
                device=self._device,
                sampler=self._sampler_for(capacity),
                requests=requests,
                max_batch_rows=self._max_batch_rows,
            )

    def _sampler_for(self, capacity: int) -> ActionSampler:
        assert capacity > 0
        if self._sampler is None or self._sampler_capacity < capacity:
            self._sampler = ActionSampler.create(
                batch_capacity=capacity,
                device=self._device,
            )
            self._sampler_capacity = capacity
        return self._sampler


def _resolve_device(name: str) -> Ok[torch.device] | Rejected:
    if name == "cpu":
        return Ok(torch.device("cpu"))
    if name == "cuda":
        if not torch.cuda.is_available():
            return Rejected(reason="AI CUDA device is unavailable")
        return Ok(torch.device("cuda"))
    if name == "mps":
        if not torch.backends.mps.is_available():
            return Rejected(reason="AI MPS device is unavailable")
        return Ok(torch.device("mps"))
    return Rejected(reason="AI device name is invalid")


__all__ = ("TorchBatchExecutor",)
