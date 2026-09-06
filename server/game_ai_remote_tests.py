"""Black-box tests for whole-controller remote inference."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import final, override

import httpx

from server.foundation.result import Ok, Rejected
from server.game import Seat, commands
from server.game.rules.cards import CardId
from server.game.snapshots import PlayerSnapshot
from server.game_ai.controller import AIControllerPort, AIUnavailable
from server.game_ai.remote import (
    RemoteAIController,
    RemoteCommand,
    RemoteDecisionRequest,
    RemoteDecisionResponse,
    RemoteObservation,
    RemoteRetryPolicy,
    RemoteSessionRegistry,
)
from tests.support import card
from tests.support import snapshot as make_snapshot


def _observations() -> list[int]:
    return []


@dataclass(slots=True)
class _Controller:
    observed: list[int] = field(default_factory=_observations)
    decision_count: int = 0

    def observe(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
        error: str | None,
    ) -> None:
        del snapshot, error
        self.observed.append(seq)

    async def decide(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[commands.Command] | AIUnavailable:
        del seq, snapshot
        self.decision_count += 1
        return Ok(commands.Play(card_ids=(CardId("D1-hearts-3"),)))


@dataclass(slots=True)
class _Factory:
    controller: _Controller

    def local_controller(
        self,
        seat: Seat,
    ) -> Ok[AIControllerPort] | Rejected:
        assert seat == Seat.A
        return Ok(self.controller)


@dataclass(slots=True)
class _TaskOwner:
    def create_task[ResultT](
        self,
        coroutine: Coroutine[object, object, ResultT],
        *,
        name: str,
    ) -> asyncio.Task[ResultT]:
        del name
        return asyncio.create_task(coroutine)


async def test_remote_registry_replays_idempotent_decision() -> None:
    controller = _Controller()
    registry = RemoteSessionRegistry(_Factory(controller), _TaskOwner())
    snapshot = _play_snapshot()
    request = RemoteDecisionRequest(
        session_id="session-0123456789",
        seat=Seat.A,
        observations=(
            RemoteObservation(
                seq=0,
                snapshot=snapshot,
            ),
        ),
        seq=0,
        snapshot=snapshot,
    )

    first = await registry.decide(request)
    second = await registry.decide(request)

    assert first == second
    assert first.command == RemoteCommand(
        kind="play",
        card_ids=("D1-hearts-3",),
    )
    assert controller.decision_count == 1
    assert controller.observed == [0]


@final
@dataclass(slots=True)
class _BlockingController(_Controller):
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    @override
    async def decide(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[commands.Command] | AIUnavailable:
        del seq, snapshot
        self.decision_count += 1
        _ = self.entered.set()
        _ = await self.release.wait()
        return Ok(commands.Play(card_ids=(CardId("D1-hearts-3"),)))


async def test_remote_registry_coalesces_concurrent_decision() -> None:
    controller = _BlockingController()
    registry = RemoteSessionRegistry(_Factory(controller), _TaskOwner())
    snapshot = _play_snapshot()
    request = RemoteDecisionRequest(
        session_id="session-0123456789",
        seat=Seat.A,
        observations=(RemoteObservation(seq=0, snapshot=snapshot),),
        seq=0,
        snapshot=snapshot,
    )

    first = asyncio.create_task(registry.decide(request))
    _ = await controller.entered.wait()
    second = asyncio.create_task(registry.decide(request))
    await asyncio.sleep(0)
    assert controller.decision_count == 1
    _ = controller.release.set()

    assert await first == await second
    assert controller.decision_count == 1


async def test_remote_controller_buffers_views_until_decision() -> None:
    requests: list[RemoteDecisionRequest] = []

    def handle(request: httpx.Request) -> httpx.Response:
        decoded = RemoteDecisionRequest.model_validate_json(
            request.content
        )
        requests.append(decoded)
        response = RemoteDecisionResponse(
            command=RemoteCommand(
                kind="play",
                card_ids=("D1-hearts-3",),
            )
        )
        return httpx.Response(
            status_code=200,
            content=response.model_dump_json(),
            headers={"content-type": "application/json"},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle),
        base_url="http://ai.example",
    ) as client:
        controller = RemoteAIController(
            seat=Seat.A,
            client=client,
            session_id="session-0123456789",
        )
        waiting = make_snapshot()
        playing = _play_snapshot()
        controller.observe(
            seq=0,
            snapshot=waiting,
            error=None,
        )
        controller.observe(
            seq=1,
            snapshot=playing,
            error=None,
        )
        assert requests == []

        decided = await controller.decide(
            seq=1,
            snapshot=playing,
        )

    assert isinstance(decided, Ok)
    assert decided.value == commands.Play(
        card_ids=(CardId("D1-hearts-3"),)
    )
    assert len(requests) == 1
    assert tuple(item.seq for item in requests[0].observations) == (
        0,
        1,
    )


async def test_remote_controller_retries_transient_status() -> None:
    attempts = 0

    def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(status_code=503)
        response = RemoteDecisionResponse(
            command=RemoteCommand(
                kind="play",
                card_ids=("D1-hearts-3",),
            )
        )
        return httpx.Response(
            status_code=200,
            content=response.model_dump_json(),
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle),
        base_url="http://ai.example",
    ) as client:
        controller = RemoteAIController(
            seat=Seat.A,
            client=client,
            session_id="session-0123456789",
            retry=RemoteRetryPolicy(
                deadline_seconds=1.0,
                attempt_timeout_seconds=1.0,
                initial_delay_seconds=0.0,
                maximum_delay_seconds=0.0,
            ),
        )
        playing = _play_snapshot()
        controller.observe(
            seq=0,
            snapshot=playing,
            error=None,
        )

        decided = await controller.decide(
            seq=0,
            snapshot=playing,
        )

    assert isinstance(decided, Ok)
    assert attempts == 2


async def test_remote_controller_returns_nonretryable_failure() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=400)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle),
        base_url="http://ai.example",
    ) as client:
        controller = RemoteAIController(
            seat=Seat.A,
            client=client,
            session_id="session-0123456789",
        )
        playing = _play_snapshot()
        controller.observe(
            seq=0,
            snapshot=playing,
            error=None,
        )

        decided = await controller.decide(
            seq=0,
            snapshot=playing,
        )

    assert isinstance(decided, AIUnavailable)
    assert decided.error == "remote AI request failed: HTTP 400"


def _play_snapshot() -> PlayerSnapshot:
    return make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        hand=[card("hearts", "3")],
    )
