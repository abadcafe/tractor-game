"""Asynchronous single-owner runtime for production decisions."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, final

import torch

from server.foundation.result import Ok, Rejected
from server.policy_model.actions import GeneratedAction
from server.policy_model.network import PolicyModel

from ._cpu import cpu_execution_plan, initialize_cpu_worker
from ._executor import TorchBatchExecutor
from .contracts import PolicyDecisionRequest

type DecisionResult = Ok[GeneratedAction] | Rejected

_LOGGER = logging.getLogger(__name__)


class InferenceTaskOwner(Protocol):
    """Process capability that supervises the inference dispatcher."""

    def create_task(
        self,
        coroutine: Coroutine[object, object, None],
        *,
        name: str,
    ) -> asyncio.Task[None]:
        """Start one named task under process ownership."""
        ...


@dataclass(frozen=True, slots=True)
class _DecisionCall:
    request: PolicyDecisionRequest
    future: asyncio.Future[DecisionResult]


@dataclass(frozen=True, slots=True)
class _StopCall:
    future: asyncio.Future[None]


type _InferenceCall = _DecisionCall | _StopCall


@final
class InferenceRuntime:
    """Batch concurrent decisions around one model-owning worker."""

    def __init__(
        self,
        executor: TorchBatchExecutor,
        task_owner: InferenceTaskOwner,
    ) -> None:
        self._executor = executor
        self._task_owner = task_owner
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

    @property
    def model_id(self) -> str:
        """Return the loaded checkpoint or in-memory model identity."""
        return self._executor.model_id

    @property
    def device_type(self) -> str:
        """Return the concrete inference device type."""
        return self._executor.device_type

    @classmethod
    def load(
        cls,
        *,
        checkpoint_path: Path,
        device_name: str,
        task_owner: InferenceTaskOwner,
    ) -> Ok[InferenceRuntime] | Rejected:
        """Load the exact current checkpoint into a new runtime."""
        executor = TorchBatchExecutor.load(
            checkpoint_path=checkpoint_path,
            device_name=device_name,
        )
        if isinstance(executor, Rejected):
            return executor
        return Ok(cls(executor.value, task_owner))

    @classmethod
    def create(
        cls,
        *,
        model: PolicyModel,
        device: torch.device,
        task_owner: InferenceTaskOwner,
    ) -> InferenceRuntime:
        """Create a runtime from an already constructed model."""
        return cls(
            TorchBatchExecutor(
                model=model,
                device=device,
                model_id="in-memory",
            ),
            task_owner,
        )

    async def decide(
        self,
        *,
        request: PolicyDecisionRequest,
    ) -> DecisionResult:
        """Schedule one complete policy decision."""
        self._raise_dispatcher_failure()
        if self._closed:
            return Rejected(reason="AI inference runtime is closed")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[DecisionResult] = loop.create_future()
        self._start_dispatcher()
        self._queue.put_nowait(
            _DecisionCall(request=request, future=future)
        )
        return await future

    async def close(self) -> None:
        """Drain queued decisions and stop the model-owning worker."""
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
            self._dispatcher = self._task_owner.create_task(
                self._dispatch(),
                name=f"inference:{self._executor.model_id}:dispatcher",
            )
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
        _LOGGER.critical(
            "runtime.task task=inference-dispatcher model_id=%s "
            + "device=%s error=%s: %s",
            self._executor.model_id,
            self._executor.device_type,
            type(failure).__name__,
            failure,
            exc_info=(
                type(failure),
                failure,
                failure.__traceback__,
            ),
        )
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
            decisions = tuple(
                call
                for call in calls
                if isinstance(call, _DecisionCall)
            )
            stopping.extend(
                call for call in calls if isinstance(call, _StopCall)
            )
            if decisions:
                await self._execute_decisions(decisions)
        for call in stopping:
            if not call.future.cancelled():
                call.future.set_result(None)
        self._active_calls = ()

    async def _execute_decisions(
        self,
        calls: tuple[_DecisionCall, ...],
    ) -> None:
        requests = tuple(call.request for call in calls)
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        operation = functools.partial(
            self._executor.decide,
            requests=requests,
        )
        result = await loop.run_in_executor(self._pool, operation)
        _LOGGER.debug(
            "inference.batch model_id=%s device=%s "
            + "batch_size=%d elapsed_ms=%.3f",
            self._executor.model_id,
            self._executor.device_type,
            len(calls),
            (time.perf_counter() - started) * 1000.0,
        )
        if isinstance(result, Rejected):
            for call in calls:
                if not call.future.cancelled():
                    call.future.set_result(result)
            return
        assert len(result.value) == len(calls)
        for call, action in zip(calls, result.value, strict=True):
            if not call.future.cancelled():
                call.future.set_result(Ok(action))


def _fail_call(
    call: _InferenceCall,
    failure: BaseException,
) -> None:
    if not call.future.done():
        call.future.set_exception(failure)


__all__ = ("InferenceRuntime", "InferenceTaskOwner")
