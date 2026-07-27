"""Application-scoped model ownership and seat controller creation."""

from __future__ import annotations

import random
import secrets
from typing import TYPE_CHECKING

from server.foundation.result import Ok, Rejected
from server.game import Seat

from .config import AIConfig
from .controller import AIController
from .model import AIControllerPort, ModelEvaluator

if TYPE_CHECKING:
    from .inference import InferenceRuntime


class AIService:
    """Load one checkpoint once and share its immutable model."""

    def __init__(self, config: AIConfig) -> None:
        self._config = config
        self._runtime: InferenceRuntime | None = None
        self._load_rejection: Rejected | None = None

    def controller(
        self,
        seat: Seat,
    ) -> Ok[AIControllerPort] | Rejected:
        """Create one seat-local controller over the shared model."""
        model_result = self._model_evaluator()
        if isinstance(model_result, Rejected):
            return model_result
        return Ok(
            AIController(
                seat=seat,
                model=model_result.value,
                particle_count=self._config.particle_count,
                direct_samples=self._config.direct_samples,
                search_config=self._config.search_config(),
                random_source=random.Random(secrets.randbits(128)),
            )
        )

    def _model_evaluator(
        self,
    ) -> Ok[ModelEvaluator] | Rejected:
        if self._runtime is not None:
            model: ModelEvaluator = self._runtime
            return Ok(model)
        if self._load_rejection is not None:
            return self._load_rejection
        from .inference import InferenceRuntime

        loaded = InferenceRuntime.load(
            checkpoint_path=self._config.checkpoint_path,
            device_name=self._config.device,
        )
        if isinstance(loaded, Rejected):
            self._load_rejection = loaded
            return loaded
        self._runtime = loaded.value
        model = loaded.value
        return Ok(model)

    async def close(self) -> None:
        """Close the inference runtime if the model was loaded."""
        runtime = self._runtime
        if runtime is None:
            return
        self._runtime = None
        await runtime.close()


__all__ = ("AIService",)
