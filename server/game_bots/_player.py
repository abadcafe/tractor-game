"""Player lifecycle and submission loop for bots."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import final, override

from server.foundation.result import Ok, Rejected
from server.game import CommandRejected, commands
from server.game_runtime.player import (
    BotPlayerDescription,
    BotPolicyName,
    CommandDecoder,
    PlayerDescription,
    PlayerPort,
    PlayerView,
    UserId,
)

from ._policy import DecisionPolicy, DecisionRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _New:
    pass


@dataclass(frozen=True, slots=True)
class _Running:
    port: PlayerPort
    queue: asyncio.Queue[PlayerView]
    task: asyncio.Task[None]
    ready: asyncio.Event


@dataclass(frozen=True, slots=True)
class _Stopped:
    pass


type _Lifecycle = _New | _Running | _Stopped


@final
class _TypedCommandDecoder(CommandDecoder):
    def __init__(self, command: commands.Command) -> None:
        self._command = command

    @override
    def decode(
        self,
    ) -> Ok[commands.Command] | CommandRejected:
        return Ok(self._command)


@final
class BotPlayer:
    """Run one DecisionPolicy behind a stable runtime player."""

    def __init__(
        self,
        *,
        policy_name: BotPolicyName,
        policy: DecisionPolicy,
    ) -> None:
        self._policy_name: BotPolicyName = policy_name
        self._policy = policy
        self._lifecycle: _Lifecycle = _New()

    def lobby_status(
        self,
        requester: UserId | None,
    ) -> PlayerDescription:
        """Return the bot policy without human-only fields."""
        del requester
        return BotPlayerDescription(
            kind="bot",
            policy=self._policy_name,
        )

    async def start(self, port: PlayerPort) -> None:
        """Start and initialize one background decision loop."""
        assert isinstance(self._lifecycle, _New)
        queue: asyncio.Queue[PlayerView] = asyncio.Queue()
        ready = asyncio.Event()
        task = asyncio.create_task(self._run(port, queue, ready))
        self._lifecycle = _Running(
            port=port,
            queue=queue,
            task=task,
            ready=ready,
        )
        ready_task = asyncio.create_task(ready.wait())
        done, _pending = await asyncio.wait(
            {ready_task, task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task in done:
            if not ready_task.done():
                _ = ready_task.cancel()
                try:
                    await ready_task
                except asyncio.CancelledError:
                    pass
            task.result()
            raise AssertionError("bot task stopped before ready")
        assert ready_task in done

    async def update(self, view: PlayerView) -> None:
        """Queue every complete view without losing policy history."""
        lifecycle = self._lifecycle
        assert isinstance(lifecycle, _Running)
        lifecycle.queue.put_nowait(view)

    async def stop(self) -> None:
        """Cancel pending decisions and prevent later submissions."""
        lifecycle = self._lifecycle
        if isinstance(lifecycle, _Stopped):
            return
        self._lifecycle = _Stopped()
        if isinstance(lifecycle, _New):
            return
        _ = lifecycle.task.cancel()
        if lifecycle.task is asyncio.current_task():
            return
        try:
            await lifecycle.task
        except asyncio.CancelledError:
            pass

    async def _run(
        self,
        port: PlayerPort,
        queue: asyncio.Queue[PlayerView],
        ready: asyncio.Event,
    ) -> None:
        await port.request_view()
        while True:
            view = await queue.get()
            observed = self._policy.observe(view)
            if isinstance(observed, Rejected):
                logger.error(
                    "bot policy observation rejected: %s",
                    observed.reason,
                )
                ready.set()
                continue
            action = view.snapshot.awaiting_action
            if action is None:
                ready.set()
                continue
            if action == "next_round":
                await port.submit(
                    view.seq,
                    _TypedCommandDecoder(commands.ConfirmRound()),
                )
                ready.set()
                continue
            decision = await self._policy.decide(
                DecisionRequest(
                    view=view,
                    action=action,
                )
            )
            if isinstance(decision, Rejected):
                logger.error(
                    "bot policy decision rejected: %s",
                    decision.reason,
                )
                ready.set()
                continue
            await port.submit(
                view.seq,
                _TypedCommandDecoder(decision.value),
            )
            ready.set()


__all__ = ("BotPlayer",)
