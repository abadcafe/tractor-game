"""Single-owner synchronous PyTorch execution for inference requests."""

from __future__ import annotations

from pathlib import Path

import torch

from server.foundation.result import Ok, Rejected
from server.training.model import TractorPolicyModel
from server.training.observation import Observation
from server.training.semantic_action_plan import ActionSampler
from server.training.torch_checkpoints.load import (
    load_inference_checkpoint,
)

from ..model import (
    ActionSampleRequest,
    ActionSamples,
    ActionScoreRequest,
)
from ._sample import sample_actions
from ._score import score_actions
from ._values import estimate_values


class TorchBatchExecutor:
    """Own one model and all mutable device sampling workspaces."""

    def __init__(
        self,
        *,
        model: TractorPolicyModel,
        device: torch.device,
    ) -> None:
        self._model = model
        self._device = device
        self._sampler: ActionSampler | None = None
        self._sampler_capacity = 0
        self._model.eval()

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
        loaded = load_inference_checkpoint(
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

    def sample(
        self,
        *,
        requests: tuple[ActionSampleRequest, ...],
    ) -> Ok[tuple[ActionSamples, ...]] | Rejected:
        """Execute one already-composed sample operation."""
        capacity = max(
            len(requests),
            sum(len(request.draws) for request in requests),
        )
        with torch.inference_mode():
            return sample_actions(
                model=self._model,
                device=self._device,
                sampler=self._sampler_for(capacity),
                requests=requests,
            )

    def score(
        self,
        *,
        requests: tuple[ActionScoreRequest, ...],
    ) -> Ok[tuple[float, ...]] | Rejected:
        """Execute one already-composed score operation."""
        with torch.inference_mode():
            return score_actions(
                model=self._model,
                device=self._device,
                sampler=self._sampler_for(len(requests)),
                requests=requests,
            )

    def values(
        self,
        *,
        observations: tuple[Observation, ...],
    ) -> Ok[tuple[float, ...]] | Rejected:
        """Execute one already-composed value operation."""
        with torch.inference_mode():
            return estimate_values(
                model=self._model,
                device=self._device,
                observations=observations,
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
