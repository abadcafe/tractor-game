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

from ..queries import (
    ActionProposalRequest,
    ActionProposals,
    ActionSamples,
    PolicySampleRequest,
    PolicyScoreRequest,
)
from ._cpu import cpu_execution_plan, initialize_cpu_worker
from ._executor import TorchBatchExecutor

type SampleResult = Ok[tuple[ActionSamples, ...]] | Rejected
type ProposalResult = Ok[tuple[ActionProposals, ...]] | Rejected
type ScoreResult = Ok[tuple[float, ...]] | Rejected
type ValueResult = Ok[tuple[float, ...]] | Rejected


@dataclass(frozen=True, slots=True)
class _ProposalCall:
    requests: tuple[ActionProposalRequest, ...]
    future: asyncio.Future[ProposalResult]


@dataclass(frozen=True, slots=True)
class _SampleCall:
    requests: tuple[PolicySampleRequest, ...]
    future: asyncio.Future[SampleResult]


@dataclass(frozen=True, slots=True)
class _ScoreCall:
    requests: tuple[PolicyScoreRequest, ...]
    future: asyncio.Future[ScoreResult]


@dataclass(frozen=True, slots=True)
class _ValueCall:
    observations: tuple[Observation, ...]
    future: asyncio.Future[ValueResult]


@dataclass(frozen=True, slots=True)
class _StopCall:
    future: asyncio.Future[None]


type _InferenceCall = (
    _ProposalCall | _SampleCall | _ScoreCall | _ValueCall | _StopCall
)


class InferenceRuntime:
    """Batch concurrent requests around one model-owning worker."""

    def __init__(self, executor: TorchBatchExecutor) -> None:
        self._executor = executor
        self._queue: asyncio.Queue[_InferenceCall] = asyncio.Queue()
        self._dispatcher: asyncio.Task[None] | None = None
        self._active_calls: tuple[_InferenceCall, ...] = ()
        self._failure: BaseException | None = None
        self._pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tractor-ai",
            initializer=initialize_cpu_worker,
            initargs=(cpu_execution_plan(),),
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
        return cls(
            TorchBatchExecutor(
                model=model,
                device=device,
            )
        )

    async def propose(
        self,
        *,
        requests: tuple[ActionProposalRequest, ...],
    ) -> ProposalResult:
        """Schedule structured no-replacement root proposals."""
        assert requests
        self._raise_dispatcher_failure()
        if self._closed:
            return Rejected(reason="AI inference runtime is closed")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ProposalResult] = loop.create_future()
        self._start_dispatcher()
        self._queue.put_nowait(
            _ProposalCall(requests=requests, future=future)
        )
        return await future

    async def sample(
        self,
        *,
        requests: tuple[PolicySampleRequest, ...],
    ) -> SampleResult:
        """Schedule explicit action draws."""
        assert requests
        self._raise_dispatcher_failure()
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
        requests: tuple[PolicyScoreRequest, ...],
    ) -> ScoreResult:
        """Schedule teacher-forced action scores."""
        assert requests
        self._raise_dispatcher_failure()
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
        self._raise_dispatcher_failure()
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
        try:
            if dispatcher is not None:
                if not dispatcher.done():
                    loop = asyncio.get_running_loop()
                    stopped: asyncio.Future[None] = loop.create_future()
                    self._queue.put_nowait(_StopCall(future=stopped))
                    await stopped
                await dispatcher
        finally:
            self._pool.shutdown(wait=True)

    def _start_dispatcher(self) -> None:
        if self._dispatcher is None:
            self._dispatcher = asyncio.create_task(self._dispatch())
            self._dispatcher.add_done_callback(
                self._dispatcher_finished
            )

    def _dispatcher_finished(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        failure = task.exception()
        if failure is None:
            return
        self._failure = failure
        for call in self._active_calls:
            _fail_call(call, failure)
        self._active_calls = ()
        while not self._queue.empty():
            _fail_call(self._queue.get_nowait(), failure)

    def _raise_dispatcher_failure(self) -> None:
        if self._failure is not None:
            raise self._failure

    async def _dispatch(self) -> None:
        stopping: list[_StopCall] = []
        while not stopping:
            first = await self._queue.get()
            calls = [first]
            while not self._queue.empty():
                calls.append(self._queue.get_nowait())
            self._active_calls = tuple(calls)
            sample_calls = tuple(
                call for call in calls if isinstance(call, _SampleCall)
            )
            proposal_calls = tuple(
                call
                for call in calls
                if isinstance(call, _ProposalCall)
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
            if proposal_calls:
                await self._execute_proposals(proposal_calls)
            if sample_calls:
                await self._execute_samples(sample_calls)
            if score_calls:
                await self._execute_scores(score_calls)
            if value_calls:
                await self._execute_values(value_calls)
        for call in stopping:
            if not call.future.cancelled():
                call.future.set_result(None)
        self._active_calls = ()

    async def _execute_proposals(
        self, calls: tuple[_ProposalCall, ...]
    ) -> None:
        requests = tuple(
            request for call in calls for request in call.requests
        )
        loop = asyncio.get_running_loop()
        operation = functools.partial(
            self._executor.propose,
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


def _fail_call(call: _InferenceCall, failure: BaseException) -> None:
    if not call.future.done():
        call.future.set_exception(failure)


__all__ = ("InferenceRuntime",)
