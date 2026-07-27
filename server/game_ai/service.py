"""Application-scoped model ownership and seat controller creation."""

from __future__ import annotations

import random
import secrets
import threading

from server.foundation.result import Ok, Rejected
from server.game import Seat

from .config import AIConfig
from .controller import AIController
from .model import AIControllerPort, ModelEvaluator


class AIService:
    """Load one checkpoint once and share its immutable model."""

    def __init__(self, config: AIConfig) -> None:
        self._config = config
        self._model: ModelEvaluator | None = None
        self._load_rejection: Rejected | None = None
        self._load_lock = threading.Lock()

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
        with self._load_lock:
            if self._model is not None:
                return Ok(self._model)
            if self._load_rejection is not None:
                return self._load_rejection
            from .torch_model import TorchModelEvaluator

            loaded = TorchModelEvaluator.load_for_device_name(
                checkpoint_path=self._config.checkpoint_path,
                device_name=self._config.device,
            )
            if isinstance(loaded, Rejected):
                self._load_rejection = loaded
                return loaded
            model: ModelEvaluator = loaded.value
            self._model = model
            return Ok(model)


__all__ = ("AIService",)
