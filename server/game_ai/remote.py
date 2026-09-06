"""Whole-controller request protocol and accelerated remote boundary."""

from __future__ import annotations

import asyncio
import logging
import random
import secrets
import time
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Protocol, Self

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from server.foundation.result import Ok, Rejected
from server.game import Seat, commands
from server.game.rules.cards import CardId
from server.game.snapshots import PlayerSnapshot

from .controller import AIControllerPort, AIUnavailable

_LOGGER = logging.getLogger(__name__)
_RETRYABLE_STATUS_CODES = frozenset((408, 425, 429, 500, 502, 503, 504))

type RemoteCommandKind = Literal[
    "reveal_bid",
    "pass_bid",
    "bury",
    "stir",
    "pass_stir",
    "play",
]


class RemoteObservation(BaseModel):
    """One complete contiguous player view."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    seq: int = Field(ge=0)
    snapshot: PlayerSnapshot
    error: str | None = None


class RemoteCommand(BaseModel):
    """Closed command wire used only by model-backed controllers."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    kind: RemoteCommandKind
    card_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_card_shape(self) -> Self:
        needs_cards = self.kind in {
            "reveal_bid",
            "bury",
            "stir",
            "play",
        }
        if bool(self.card_ids) != needs_cards:
            raise ValueError(
                "remote command card payload does not match its kind"
            )
        return self

    def to_domain(self) -> commands.Command:
        """Decode the exact typed game command."""
        ids = tuple(CardId(value) for value in self.card_ids)
        if self.kind == "reveal_bid":
            assert ids
            return commands.RevealBid(card_ids=ids)
        if self.kind == "pass_bid":
            assert not ids
            return commands.PassBid()
        if self.kind == "bury":
            assert ids
            return commands.Bury(card_ids=ids)
        if self.kind == "stir":
            assert ids
            return commands.Stir(card_ids=ids)
        if self.kind == "pass_stir":
            assert not ids
            return commands.PassStir()
        assert self.kind == "play"
        assert ids
        return commands.Play(card_ids=ids)

    @classmethod
    def from_domain(cls, command: commands.Command) -> RemoteCommand:
        """Encode one model-supported game command."""
        if isinstance(command, commands.RevealBid):
            return cls(
                kind="reveal_bid",
                card_ids=tuple(command.card_ids),
            )
        if isinstance(command, commands.PassBid):
            return cls(kind="pass_bid")
        if isinstance(command, commands.Bury):
            return cls(
                kind="bury",
                card_ids=tuple(command.card_ids),
            )
        if isinstance(command, commands.Stir):
            return cls(
                kind="stir",
                card_ids=tuple(command.card_ids),
            )
        if isinstance(command, commands.PassStir):
            return cls(kind="pass_stir")
        assert isinstance(command, commands.Play)
        return cls(kind="play", card_ids=tuple(command.card_ids))


class RemoteDecisionRequest(BaseModel):
    """All unsent observations plus one idempotent decision query."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    session_id: str = Field(min_length=16, max_length=128)
    seat: Seat
    observations: tuple[RemoteObservation, ...]
    seq: int = Field(ge=0)
    snapshot: PlayerSnapshot

    @model_validator(mode="after")
    def _validate_observations(self) -> Self:
        for index in range(1, len(self.observations)):
            previous = self.observations[index - 1]
            current = self.observations[index]
            if current.seq != previous.seq + 1:
                raise ValueError(
                    "remote observations must be contiguous"
                )
        if self.observations:
            latest = self.observations[-1]
            if (
                latest.seq != self.seq
                or latest.snapshot != self.snapshot
            ):
                raise ValueError(
                    "remote decision must match its latest observation"
                )
        return self


class RemoteDecisionResponse(BaseModel):
    """Exactly one command or domain error."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )

    command: RemoteCommand | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        if (self.command is None) == (self.error is None):
            raise ValueError(
                "remote decision must contain one command or one error"
            )
        return self


class LocalControllerFactory(Protocol):
    """Factory surface accepted by the remote session registry."""

    def local_controller(
        self,
        seat: Seat,
    ) -> Ok[AIControllerPort] | Rejected:
        """Create a controller only when this process owns a model."""
        ...


class RemoteTaskOwner(Protocol):
    """Application lifecycle surface for shared remote decisions."""

    def create_task[ResultT](
        self,
        coroutine: Coroutine[object, object, ResultT],
        *,
        name: str,
    ) -> asyncio.Task[ResultT]:
        """Start one named task owned by the application."""
        ...


@dataclass(slots=True)
class _RemoteSession:
    seat: Seat
    controller: AIControllerPort
    last_decision_seq: int | None = None
    last_command: RemoteCommand | None = None
    last_observation: RemoteObservation | None = None
    in_flight_seq: int | None = None
    in_flight: asyncio.Task[RemoteDecisionResponse] | None = None


def _session_map() -> dict[str, _RemoteSession]:
    return {}


@dataclass(slots=True)
class RemoteSessionRegistry:
    """Process-local idempotent controller sessions for remote games."""

    factory: LocalControllerFactory
    task_owner: RemoteTaskOwner
    _sessions: dict[str, _RemoteSession] = field(
        default_factory=_session_map
    )

    async def decide(
        self,
        request: RemoteDecisionRequest,
    ) -> RemoteDecisionResponse:
        """Apply observations then execute or replay one decision."""
        session = self._sessions.get(request.session_id)
        if session is None:
            created = self.factory.local_controller(request.seat)
            if isinstance(created, Rejected):
                return RemoteDecisionResponse(error=created.reason)
            session = _RemoteSession(
                seat=request.seat,
                controller=created.value,
            )
            self._sessions[request.session_id] = session
        elif session.seat != request.seat:
            return RemoteDecisionResponse(
                error="AI session seat does not match"
            )
        if session.last_decision_seq == request.seq:
            assert session.last_command is not None
            return RemoteDecisionResponse(command=session.last_command)
        in_flight = session.in_flight
        if in_flight is not None:
            if session.in_flight_seq != request.seq:
                return RemoteDecisionResponse(
                    error="AI session has another decision in progress"
                )
            try:
                return await asyncio.shield(in_flight)
            finally:
                if in_flight.done() and session.in_flight is in_flight:
                    session.in_flight = None
                    session.in_flight_seq = None
        for observation in request.observations:
            previous = session.last_observation
            if previous is not None and observation.seq == previous.seq:
                if observation != previous:
                    return RemoteDecisionResponse(
                        error="AI session received conflicting state"
                    )
                continue
            if (
                previous is not None
                and observation.seq != previous.seq + 1
            ):
                return RemoteDecisionResponse(
                    error="AI session missed a state sequence"
                )
            session.controller.observe(
                seq=observation.seq,
                snapshot=observation.snapshot,
                error=observation.error,
            )
            session.last_observation = observation
        if (
            session.last_decision_seq is not None
            and request.seq < session.last_decision_seq
        ):
            return RemoteDecisionResponse(
                error="AI decision sequence moved backwards"
            )
        task = self.task_owner.create_task(
            self._decide_new(session=session, request=request),
            name=(
                f"ai-remote-{request.session_id[:8]}-seq-{request.seq}"
            ),
        )
        session.in_flight_seq = request.seq
        session.in_flight = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done() and session.in_flight is task:
                session.in_flight = None
                session.in_flight_seq = None

    @staticmethod
    async def _decide_new(
        *,
        session: _RemoteSession,
        request: RemoteDecisionRequest,
    ) -> RemoteDecisionResponse:
        """Execute exactly one new controller decision."""
        decided = await session.controller.decide(
            seq=request.seq,
            snapshot=request.snapshot,
        )
        if isinstance(decided, AIUnavailable):
            return RemoteDecisionResponse(error=decided.error)
        command = RemoteCommand.from_domain(decided.value)
        session.last_decision_seq = request.seq
        session.last_command = command
        return RemoteDecisionResponse(command=command)

    def clear(self) -> None:
        """Forget all remote controller sessions."""
        self._sessions.clear()


def _observation_list() -> list[RemoteObservation]:
    return []


@dataclass(frozen=True, slots=True)
class RemoteRetryPolicy:
    """Bounded retry timing for one idempotent remote decision."""

    deadline_seconds: float = 120.0
    attempt_timeout_seconds: float = 30.0
    initial_delay_seconds: float = 0.25
    maximum_delay_seconds: float = 4.0

    def __post_init__(self) -> None:
        assert self.deadline_seconds > 0.0
        assert self.attempt_timeout_seconds > 0.0
        assert self.initial_delay_seconds >= 0.0
        assert self.maximum_delay_seconds >= self.initial_delay_seconds


@dataclass(slots=True)
class RemoteAIController:
    """Buffer observations and query one whole remote controller."""

    seat: Seat
    client: httpx.AsyncClient
    session_id: str = field(
        default_factory=lambda: secrets.token_urlsafe(24)
    )
    retry: RemoteRetryPolicy = field(default_factory=RemoteRetryPolicy)
    random_source: random.Random = field(
        default_factory=random.SystemRandom,
        repr=False,
    )
    _pending: list[RemoteObservation] = field(
        default_factory=_observation_list
    )
    _last_observation: RemoteObservation | None = None

    def observe(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
        error: str | None,
    ) -> None:
        """Buffer one contiguous view without network round trips."""
        observation = RemoteObservation(
            seq=seq,
            snapshot=snapshot,
            error=error,
        )
        previous = self._last_observation
        if previous is not None and seq == previous.seq:
            assert observation == previous, (
                "remote AI received conflicting duplicate state"
            )
            return
        if previous is not None and seq != previous.seq + 1:
            raise AssertionError(
                "remote AI missed a game-state sequence"
            )
        self._pending.append(observation)
        self._last_observation = observation

    async def decide(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[commands.Command] | AIUnavailable:
        """Send pending views and await one remote policy decision."""
        previous = self._last_observation
        if (
            previous is None
            or previous.seq != seq
            or previous.snapshot != snapshot
        ):
            raise AssertionError(
                "remote AI decision does not match latest view"
            )
        request = RemoteDecisionRequest(
            session_id=self.session_id,
            seat=self.seat,
            observations=tuple(self._pending),
            seq=seq,
            snapshot=snapshot,
        )
        response = await self._request(request)
        if isinstance(response, AIUnavailable):
            return response
        try:
            decoded = RemoteDecisionResponse.model_validate_json(
                response.content
            )
        except ValidationError as error:
            return AIUnavailable(
                error=f"remote AI response is invalid: {error}"
            )
        self._pending.clear()
        if decoded.error is not None:
            return AIUnavailable(error=decoded.error)
        assert decoded.command is not None
        return Ok(decoded.command.to_domain())

    async def _request(
        self,
        request: RemoteDecisionRequest,
    ) -> httpx.Response | AIUnavailable:
        started = time.monotonic()
        delay = self.retry.initial_delay_seconds
        attempt = 1
        error_message = "remote AI deadline exceeded"
        while True:
            remaining = self.retry.deadline_seconds - (
                time.monotonic() - started
            )
            if remaining <= 0.0:
                return AIUnavailable(error=error_message)
            timeout = min(
                self.retry.attempt_timeout_seconds,
                remaining,
            )
            try:
                response = await asyncio.wait_for(
                    self.client.post(
                        "/api/ai/decision",
                        content=request.model_dump_json(
                            exclude_computed_fields=True
                        ),
                        headers={"content-type": "application/json"},
                    ),
                    timeout=timeout,
                )
            except (TimeoutError, httpx.RequestError) as error:
                error_message = f"remote AI request failed: {error}"
            else:
                if response.status_code < 400:
                    return response
                error_message = (
                    "remote AI request failed: HTTP "
                    f"{response.status_code}"
                )
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    return AIUnavailable(error=error_message)
            remaining = self.retry.deadline_seconds - (
                time.monotonic() - started
            )
            if remaining <= 0.0:
                return AIUnavailable(error=error_message)
            sleep_seconds = self.random_source.uniform(
                0.0,
                min(delay, remaining),
            )
            _LOGGER.warning(
                "ai.remote attempt=%d delay_ms=%.3f error=%s",
                attempt,
                sleep_seconds * 1000.0,
                error_message,
            )
            if sleep_seconds > 0.0:
                await asyncio.sleep(sleep_seconds)
            delay = min(
                max(delay * 2.0, self.retry.initial_delay_seconds),
                self.retry.maximum_delay_seconds,
            )
            attempt += 1


__all__ = (
    "LocalControllerFactory",
    "RemoteAIController",
    "RemoteCommand",
    "RemoteDecisionRequest",
    "RemoteDecisionResponse",
    "RemoteObservation",
    "RemoteRetryPolicy",
    "RemoteSessionRegistry",
    "RemoteTaskOwner",
)
