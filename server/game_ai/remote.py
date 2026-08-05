"""Whole-controller request protocol and accelerated remote boundary."""

from __future__ import annotations

import secrets
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

from .controller import AIControllerPort

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


@dataclass(slots=True)
class _RemoteSession:
    seat: Seat
    controller: AIControllerPort
    last_decision_seq: int | None = None
    last_command: RemoteCommand | None = None


def _session_map() -> dict[str, _RemoteSession]:
    return {}


@dataclass(slots=True)
class RemoteSessionRegistry:
    """Process-local idempotent controller sessions for remote games."""

    factory: LocalControllerFactory
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
        for observation in request.observations:
            observed = session.controller.observe(
                seq=observation.seq,
                snapshot=observation.snapshot,
                error=observation.error,
            )
            if isinstance(observed, Rejected):
                return RemoteDecisionResponse(error=observed.reason)
        if (
            session.last_decision_seq is not None
            and request.seq < session.last_decision_seq
        ):
            return RemoteDecisionResponse(
                error="AI decision sequence moved backwards"
            )
        decided = await session.controller.decide(
            seq=request.seq,
            snapshot=request.snapshot,
        )
        if isinstance(decided, Rejected):
            return RemoteDecisionResponse(error=decided.reason)
        command = RemoteCommand.from_domain(decided.value)
        session.last_decision_seq = request.seq
        session.last_command = command
        return RemoteDecisionResponse(command=command)

    def clear(self) -> None:
        """Forget all remote controller sessions."""
        self._sessions.clear()


def _observation_list() -> list[RemoteObservation]:
    return []


@dataclass(slots=True)
class RemoteAIController:
    """Buffer observations and query one whole remote controller."""

    seat: Seat
    client: httpx.AsyncClient
    session_id: str = field(
        default_factory=lambda: secrets.token_urlsafe(24)
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
    ) -> Ok[None] | Rejected:
        """Buffer one contiguous view without network round trips."""
        observation = RemoteObservation(
            seq=seq,
            snapshot=snapshot,
            error=error,
        )
        previous = self._last_observation
        if previous is not None and seq == previous.seq:
            if observation != previous:
                return Rejected(
                    reason=(
                        "remote AI received conflicting duplicate state"
                    )
                )
            return Ok(None)
        if previous is not None and seq != previous.seq + 1:
            return Rejected(
                reason="remote AI missed a game-state sequence"
            )
        self._pending.append(observation)
        self._last_observation = observation
        return Ok(None)

    async def decide(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[commands.Command] | Rejected:
        """Send pending views and await one remote policy decision."""
        previous = self._last_observation
        if (
            previous is None
            or previous.seq != seq
            or previous.snapshot != snapshot
        ):
            return Rejected(
                reason="remote AI decision does not match latest view"
            )
        request = RemoteDecisionRequest(
            session_id=self.session_id,
            seat=self.seat,
            observations=tuple(self._pending),
            seq=seq,
            snapshot=snapshot,
        )
        try:
            response = await self.client.post(
                "/api/ai/decision",
                content=request.model_dump_json(
                    exclude_computed_fields=True
                ),
                headers={"content-type": "application/json"},
            )
            _ = response.raise_for_status()
        except (httpx.RequestError, httpx.HTTPStatusError) as error:
            return Rejected(reason=f"remote AI request failed: {error}")
        try:
            decoded = RemoteDecisionResponse.model_validate_json(
                response.content
            )
        except ValidationError as error:
            return Rejected(
                reason=f"remote AI response is invalid: {error}"
            )
        self._pending.clear()
        if decoded.error is not None:
            return Rejected(reason=decoded.error)
        assert decoded.command is not None
        return Ok(decoded.command.to_domain())


__all__ = (
    "LocalControllerFactory",
    "RemoteAIController",
    "RemoteCommand",
    "RemoteDecisionRequest",
    "RemoteDecisionResponse",
    "RemoteObservation",
    "RemoteSessionRegistry",
)
