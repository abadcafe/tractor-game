"""Deterministic fan-out for async control commands."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from server.foundation import result as _result
from server.foundation.result import Ok, Rejected


class ControlCommandSender[CommandT](Protocol):
    """Endpoint capable of sending one typed control command."""

    async def send_command(
        self, command: CommandT
    ) -> _result.Ok[None] | _result.Rejected: ...


@dataclass(frozen=True, slots=True)
class ControlCommandBroadcastFailure[TargetT]:
    """A deterministic command broadcast failure with sent targets."""

    sent_targets: tuple[TargetT, ...]
    failed_target: TargetT
    rejection: Rejected

    def __post_init__(self) -> None:
        assert self.rejection.reason


async def broadcast_control_commands[TargetT, CommandT](
    *,
    targets: tuple[TargetT, ...],
    sender: Callable[[TargetT], ControlCommandSender[CommandT]],
    command: Callable[[TargetT], CommandT],
) -> (
    _result.Ok[tuple[TargetT, ...]]
    | ControlCommandBroadcastFailure[TargetT]
):
    """Send one command to each target concurrently.

    All sends finish before return. Cleanup can then address every child
    that received a command. Failure reporting stays in target order.
    """
    assert targets
    tasks: list[asyncio.Task[_result.Ok[None] | _result.Rejected]] = []
    for target in targets:
        tasks.append(
            asyncio.create_task(
                sender(target).send_command(command(target))
            )
        )
    gathered = await asyncio.gather(*tasks)
    results: tuple[_result.Ok[None] | _result.Rejected, ...] = tuple(
        gathered
    )
    sent_targets = tuple(
        target
        for target, result in zip(targets, results, strict=True)
        if isinstance(result, Ok)
    )
    for target, result in zip(targets, results, strict=True):
        if isinstance(result, Rejected):
            return ControlCommandBroadcastFailure(
                sent_targets=sent_targets,
                failed_target=target,
                rejection=result,
            )
    return Ok(value=sent_targets)
