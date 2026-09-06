"""Session-owned player loop for automatic policies."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import final, override

from server.foundation.result import Ok
from server.game import CommandRejected, commands
from server.game_runtime.player import (
    BotPlayerDescription,
    BotPolicyName,
    CommandDecoder,
    PlayerDescription,
    PlayerFailure,
    PlayerInbox,
    PlayerPort,
    PlayerView,
    UserId,
)

from ._policy import (
    DecisionPolicy,
    DecisionRequest,
    DecisionUnavailable,
)

_LOGGER = logging.getLogger(__name__)
_SLOW_DECISION_SECONDS = 5.0


@final
class _TypedCommandDecoder(CommandDecoder):
    def __init__(self, command: commands.Command) -> None:
        self._command = command

    @override
    def decode(
        self,
    ) -> Ok[commands.Command] | CommandRejected:
        return Ok(self._command)


@dataclass(frozen=True, slots=True)
class _DecisionContext:
    game_id: str
    view: PlayerView
    action: str
    started: float


@final
class BotPlayer:
    """Serve one DecisionPolicy without owning background work."""

    def __init__(
        self,
        *,
        policy_name: BotPolicyName,
        policy: DecisionPolicy,
    ) -> None:
        self._policy_name: BotPolicyName = policy_name
        self._policy = policy

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

    async def run(
        self,
        port: PlayerPort,
        inbox: PlayerInbox,
    ) -> None:
        """Observe lossless views and submit one chosen command."""
        await port.player_ready()
        await port.request_view()
        initialized = False
        while True:
            view = await inbox.receive()
            if view is None:
                return
            if view.status == "failed":
                if not initialized:
                    await port.player_initialized()
                    initialized = True
                continue
            self._policy.observe(view)
            action = view.snapshot.awaiting_action
            if not initialized and action != "next_round":
                await port.player_initialized()
                initialized = True
            if action is None:
                continue
            if action == "next_round":
                await port.submit(
                    view.seq,
                    _TypedCommandDecoder(commands.ConfirmRound()),
                )
                continue
            request = DecisionRequest(view=view, action=action)
            context = _DecisionContext(
                game_id=port.game_id.value,
                view=view,
                action=action,
                started=time.perf_counter(),
            )
            warning = asyncio.get_running_loop().call_later(
                _SLOW_DECISION_SECONDS,
                _log_slow_decision,
                self._policy_name,
                context,
            )
            try:
                decision = await self._policy.decide(request)
            finally:
                warning.cancel()
            elapsed_ms = (
                time.perf_counter() - context.started
            ) * 1000.0
            if isinstance(decision, DecisionUnavailable):
                _LOGGER.error(
                    "bot.decision game_id=%s seat=%s seq=%d "
                    + "action=%s policy=%s elapsed_ms=%.3f error=%s",
                    port.game_id.value,
                    view.viewer.value,
                    view.seq,
                    action,
                    self._policy_name,
                    elapsed_ms,
                    decision.error,
                )
                await port.report_failure(
                    PlayerFailure(error=decision.error)
                )
                continue
            _LOGGER.info(
                "bot.decision game_id=%s seat=%s seq=%d "
                + "action=%s policy=%s elapsed_ms=%.3f",
                port.game_id.value,
                view.viewer.value,
                view.seq,
                action,
                self._policy_name,
                elapsed_ms,
            )
            await port.submit(
                view.seq,
                _TypedCommandDecoder(decision.value),
            )


def _log_slow_decision(
    policy_name: BotPolicyName,
    context: _DecisionContext,
) -> None:
    elapsed_ms = (time.perf_counter() - context.started) * 1000.0
    _LOGGER.warning(
        "bot.decision game_id=%s seat=%s seq=%d action=%s "
        + "policy=%s elapsed_ms=%.3f "
        + "error=decision exceeded slow threshold",
        context.game_id,
        context.view.viewer.value,
        context.view.seq,
        context.action,
        policy_name,
        elapsed_ms,
    )


__all__ = ("BotPlayer",)
