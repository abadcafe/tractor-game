"""Lazy public interface for the deep policy-training module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server.training.interface import (
        TrainingInitOptions,
        TrainingResumeOptions,
        TrainingService,
    )
    from server.training.stop import (
        TrainingStopRequest,
        training_stop_signals,
    )

__all__ = (
    "TrainingInitOptions",
    "TrainingResumeOptions",
    "TrainingService",
    "TrainingStopRequest",
    "training_stop_signals",
)


def __getattr__(name: str) -> object:
    """Load the training runtime only when its public API is used."""
    if name in (
        "TrainingInitOptions",
        "TrainingResumeOptions",
        "TrainingService",
    ):
        from server.training.interface import (
            TrainingInitOptions,
            TrainingResumeOptions,
            TrainingService,
        )

        if name == "TrainingInitOptions":
            return TrainingInitOptions
        if name == "TrainingResumeOptions":
            return TrainingResumeOptions
        return TrainingService
    if name in ("TrainingStopRequest", "training_stop_signals"):
        from server.training.stop import (
            TrainingStopRequest,
            training_stop_signals,
        )

        if name == "TrainingStopRequest":
            return TrainingStopRequest
        return training_stop_signals
    raise AttributeError(name)
