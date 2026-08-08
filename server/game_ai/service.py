"""Application-scoped local model or remote controller ownership."""

from __future__ import annotations

import random
import secrets
from typing import TYPE_CHECKING, final

import httpx

from server.foundation.result import Ok, Rejected
from server.game import Seat

from .config import AIConfig, LocalAIConfig, RemoteAIConfig
from .controller import AIController, AIControllerPort
from .remote import RemoteAIController

if TYPE_CHECKING:
    from server.policy_model.inference.runtime import InferenceRuntime


@final
class AIService:
    """Own exactly one configured AI deployment."""

    def __init__(self, config: AIConfig) -> None:
        self._config = config
        self._runtime: InferenceRuntime | None = None
        self._load_rejection: Rejected | None = None
        self._remote_client: httpx.AsyncClient | None = None

    def controller(
        self,
        seat: Seat,
    ) -> Ok[AIControllerPort] | Rejected:
        """Create a controller through the configured deployment."""
        if isinstance(self._config, RemoteAIConfig):
            return Ok(
                RemoteAIController(
                    seat=seat,
                    client=self._remote_http_client(),
                )
            )
        return self.local_controller(seat)

    def local_controller(
        self,
        seat: Seat,
    ) -> Ok[AIControllerPort] | Rejected:
        """Create an in-process controller or reject remote mode."""
        config = self._config
        if not isinstance(config, LocalAIConfig):
            return Rejected(
                reason=(
                    "AI endpoint cannot host sessions in remote mode"
                )
            )
        runtime_result = self._inference_runtime(config)
        if isinstance(runtime_result, Rejected):
            return runtime_result
        return Ok(
            AIController(
                seat=seat,
                model=runtime_result.value,
                random_source=random.Random(secrets.randbits(128)),
            )
        )

    def _inference_runtime(
        self,
        config: LocalAIConfig,
    ) -> Ok[InferenceRuntime] | Rejected:
        if self._runtime is not None:
            return Ok(self._runtime)
        if self._load_rejection is not None:
            return self._load_rejection
        from server.policy_model.inference.runtime import (
            InferenceRuntime,
        )

        loaded = InferenceRuntime.load(
            checkpoint_path=config.checkpoint_path,
            device_name=config.device,
        )
        if isinstance(loaded, Rejected):
            self._load_rejection = loaded
            return loaded
        self._runtime = loaded.value
        return Ok(loaded.value)

    def _remote_http_client(self) -> httpx.AsyncClient:
        config = self._config
        assert isinstance(config, RemoteAIConfig)
        if self._remote_client is None:
            self._remote_client = httpx.AsyncClient(
                base_url=str(config.endpoint).rstrip("/"),
                timeout=config.request_timeout_seconds,
            )
        return self._remote_client

    async def close(self) -> None:
        """Close the single configured inference transport."""
        runtime = self._runtime
        if runtime is not None:
            self._runtime = None
            await runtime.close()
        client = self._remote_client
        if client is not None:
            self._remote_client = None
            await client.aclose()


__all__ = ("AIService",)
