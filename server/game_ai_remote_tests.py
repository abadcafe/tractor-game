"""Black-box tests for whole-controller remote inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from server.foundation.result import Ok, Rejected
from server.game import Seat, commands
from server.game.rules.cards import CardId
from server.game.snapshots import PlayerSnapshot
from server.game_ai.controller import AIControllerPort
from server.game_ai.remote import (
    RemoteAIController,
    RemoteCommand,
    RemoteDecisionRequest,
    RemoteDecisionResponse,
    RemoteObservation,
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
    ) -> Ok[None] | Rejected:
        del snapshot, error
        self.observed.append(seq)
        return Ok(None)

    async def decide(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[commands.Command] | Rejected:
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


async def test_remote_registry_replays_idempotent_decision() -> None:
    controller = _Controller()
    registry = RemoteSessionRegistry(_Factory(controller))
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
        first = controller.observe(
            seq=0,
            snapshot=waiting,
            error=None,
        )
        second = controller.observe(
            seq=1,
            snapshot=playing,
            error=None,
        )
        assert isinstance(first, Ok)
        assert isinstance(second, Ok)
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


def _play_snapshot() -> PlayerSnapshot:
    return make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        hand=[card("hearts", "3")],
    )
