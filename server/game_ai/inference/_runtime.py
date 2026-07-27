"""Asynchronous single-owner inference runtime for the current model."""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from server.foundation.result import Ok, Rejected
from server.training.model import TractorPolicyModel
from server.training.observation import Observation

from ..model import (
    ActionSampleRequest,
    ActionSamples,
    ActionScoreRequest,
)
from ._executor import TorchBatchExecutor

type SampleResult = Ok[tuple[ActionSamples, ...]] | Rejected
type ScoreResult = Ok[tuple[float, ...]] | Rejected
type ValueResult = Ok[tuple[float, ...]] | Rejected


@dataclass(frozen=True, slots=True)
class _SampleCall:
    requests: tuple[ActionSampleRequest, ...]
    future: asyncio.Future[SampleResult]


@dataclass(frozen=True, slots=True)
class _ScoreCall:
    requests: tuple[ActionScoreRequest, ...]
    future: asyncio.Future[ScoreResult]


@dataclass(frozen=True, slots=True)
class _ValueCall:
    observations: tuple[Observation, ...]
    future: asyncio.Future[ValueResult]


@dataclass(frozen=True, slots=True)
class _StopCall:
    future: asyncio.Future[None]


type _InferenceCall = _SampleCall | _ScoreCall | _ValueCall | _StopCall


class InferenceRuntime:
    """Batch concurrent requests around one model-owning worker."""

    def __init__(self, executor: TorchBatchExecutor) -> None:
        self._executor = executor
        self._queue: asyncio.Queue[_InferenceCall] = asyncio.Queue()
        self._dispatcher: asyncio.Task[None] | None = None
        self._pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tractor-ai",
        )
        self._closed = False

    @classmethod
    def load(
        cls,
        *,
        checkpoint_path: Path,
        device_name: str,
    ) -> Ok[InferenceRuntime] | Rejected:
        """Load the current checkpoint into a new runtime."""
        executor = TorchBatchExecutor.load(
            checkpoint_path=checkpoint_path,
            device_name=device_name,
        )
        if isinstance(executor, Rejected):
            return executor
        return Ok(cls(executor.value))

    @classmethod
    def create(
        cls,
        *,
        model: TractorPolicyModel,
        device: torch.device,
    ) -> InferenceRuntime:
        """Create a runtime from an already constructed model."""
        return cls(TorchBatchExecutor(model=model, device=device))

    async def sample(
        self,
        *,
        requests: tuple[ActionSampleRequest, ...],
    ) -> SampleResult:
        """Schedule explicit action draws."""
        assert requests
        if self._closed:
            return Rejected(reason="AI inference runtime is closed")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[SampleResult] = loop.create_future()
        self._start_dispatcher()
        self._queue.put_nowait(
            _SampleCall(requests=requests, future=future)
        )
        return await future

    async def score(
        self,
        *,
        requests: tuple[ActionScoreRequest, ...],
    ) -> ScoreResult:
        """Schedule teacher-forced action scores."""
        assert requests
        if self._closed:
            return Rejected(reason="AI inference runtime is closed")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ScoreResult] = loop.create_future()
        self._start_dispatcher()
        self._queue.put_nowait(
            _ScoreCall(requests=requests, future=future)
        )
        return await future

    async def values(
        self,
        *,
        observations: tuple[Observation, ...],
    ) -> ValueResult:
        """Schedule observation value estimates."""
        assert observations
        if self._closed:
            return Rejected(reason="AI inference runtime is closed")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ValueResult] = loop.create_future()
        self._start_dispatcher()
        self._queue.put_nowait(
            _ValueCall(observations=observations, future=future)
        )
        return await future

    async def close(self) -> None:
        """Drain queued work and stop the model-owning worker."""
        if self._closed:
            return
        self._closed = True
        dispatcher = self._dispatcher
        if dispatcher is not None:
            loop = asyncio.get_running_loop()
            stopped: asyncio.Future[None] = loop.create_future()
            self._queue.put_nowait(_StopCall(future=stopped))
            await stopped
            await dispatcher
        self._pool.shutdown(wait=True)

    def _start_dispatcher(self) -> None:
        if self._dispatcher is None:
            self._dispatcher = asyncio.create_task(self._dispatch())

    async def _dispatch(self) -> None:
        stopping: list[_StopCall] = []
        while not stopping:
            first = await self._queue.get()
            calls = [first]
            while not self._queue.empty():
                calls.append(self._queue.get_nowait())
            sample_calls = tuple(
                call for call in calls if isinstance(call, _SampleCall)
            )
            score_calls = tuple(
                call for call in calls if isinstance(call, _ScoreCall)
            )
            value_calls = tuple(
                call for call in calls if isinstance(call, _ValueCall)
            )
            stopping.extend(
                call for call in calls if isinstance(call, _StopCall)
            )
            if sample_calls:
                await self._execute_samples(sample_calls)
            if score_calls:
                await self._execute_scores(score_calls)
            if value_calls:
                await self._execute_values(value_calls)
        for call in stopping:
            if not call.future.cancelled():
                call.future.set_result(None)

    async def _execute_samples(
        self, calls: tuple[_SampleCall, ...]
    ) -> None:
        requests = tuple(
            request for call in calls for request in call.requests
        )
        loop = asyncio.get_running_loop()
        operation = functools.partial(
            self._executor.sample,
            requests=requests,
        )
        result = await loop.run_in_executor(self._pool, operation)
        if isinstance(result, Rejected):
            for call in calls:
                if not call.future.cancelled():
                    call.future.set_result(result)
            return
        offset = 0
        for call in calls:
            end = offset + len(call.requests)
            if not call.future.cancelled():
                call.future.set_result(Ok(result.value[offset:end]))
            offset = end
        assert offset == len(result.value)

    async def _execute_scores(
        self, calls: tuple[_ScoreCall, ...]
    ) -> None:
        requests = tuple(
            request for call in calls for request in call.requests
        )
        loop = asyncio.get_running_loop()
        operation = functools.partial(
            self._executor.score,
            requests=requests,
        )
        result = await loop.run_in_executor(self._pool, operation)
        if isinstance(result, Rejected):
            for call in calls:
                if not call.future.cancelled():
                    call.future.set_result(result)
            return
        offset = 0
        for call in calls:
            end = offset + len(call.requests)
            if not call.future.cancelled():
                call.future.set_result(Ok(result.value[offset:end]))
            offset = end
        assert offset == len(result.value)

    async def _execute_values(
        self, calls: tuple[_ValueCall, ...]
    ) -> None:
        observations = tuple(
            observation
            for call in calls
            for observation in call.observations
        )
        loop = asyncio.get_running_loop()
        operation = functools.partial(
            self._executor.values,
            observations=observations,
        )
        result = await loop.run_in_executor(self._pool, operation)
        if isinstance(result, Rejected):
            for call in calls:
                if not call.future.cancelled():
                    call.future.set_result(result)
            return
        offset = 0
        for call in calls:
            end = offset + len(call.observations)
            if not call.future.cancelled():
                call.future.set_result(Ok(result.value[offset:end]))
            offset = end
        assert offset == len(result.value)


__all__ = ("InferenceRuntime",)
