"""Black-box tests for the shared BotPlayer runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from server.foundation.result import Ok, Rejected
from server.game import Seat, commands
from server.game.rules.cards import CardId
from server.game.snapshots import PlayerSnapshot
from server.game_ai import AIControllerPort, AIUnavailable
from server.game_bots import (
    BotPlayer,
    DecisionCommand,
    DecisionRequest,
    DecisionUnavailable,
    DefaultBotPlayerFactory,
)
from server.game_runtime.player import (
    BotPlayerDescription,
    CommandDecoder,
    PlayerFailure,
    PlayerView,
)
from server.game_runtime.registry import GameId
from tests.support import card, seat_values, snapshot


def _views() -> list[PlayerView]:
    return []


def _requests() -> list[DecisionRequest]:
    return []


def _decisions() -> list[Ok[DecisionCommand] | DecisionUnavailable]:
    return []


def _submissions() -> list[tuple[int, commands.Command]]:
    return []


def _failures() -> list[PlayerFailure]:
    return []


def _seats() -> list[Seat]:
    return []


@dataclass(slots=True)
class _AIController:
    decision: Ok[commands.Command] | AIUnavailable

    def observe(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
        error: str | None,
    ) -> None:
        del seq, snapshot, error

    async def decide(
        self,
        *,
        seq: int,
        snapshot: PlayerSnapshot,
    ) -> Ok[commands.Command] | AIUnavailable:
        del seq, snapshot
        return self.decision


@dataclass(slots=True)
class _AIControllerFactory:
    result: Ok[AIControllerPort] | Rejected
    requested_seats: list[Seat] = field(default_factory=_seats)

    def controller(
        self,
        seat: Seat,
    ) -> Ok[AIControllerPort] | Rejected:
        self.requested_seats.append(seat)
        return self.result


@dataclass(slots=True)
class _Policy:
    observation_error: str | None = None
    decisions: list[Ok[DecisionCommand] | DecisionUnavailable] = field(
        default_factory=_decisions
    )
    views: list[PlayerView] = field(default_factory=_views)
    requests: list[DecisionRequest] = field(default_factory=_requests)

    def observe(self, view: PlayerView) -> None:
        self.views.append(view)
        if self.observation_error is not None:
            raise AssertionError(self.observation_error)

    async def decide(
        self,
        request: DecisionRequest,
    ) -> Ok[DecisionCommand] | DecisionUnavailable:
        self.requests.append(request)
        assert self.decisions
        return self.decisions.pop(0)


@dataclass(slots=True)
class _Inbox:
    queue: asyncio.Queue[PlayerView | None] = field(
        default_factory=asyncio.Queue
    )

    async def receive(self) -> PlayerView | None:
        return await self.queue.get()


@dataclass(slots=True)
class _Channel:
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    state_requests: int = 0
    submissions: list[tuple[int, commands.Command]] = field(
        default_factory=_submissions
    )
    failures: list[PlayerFailure] = field(default_factory=_failures)

    @property
    def game_id(self) -> GameId:
        return GameId("4" * 32)

    async def player_ready(self) -> None:
        self.ready.set()

    async def player_initialized(self) -> None:
        return

    async def request_view(self) -> None:
        self.state_requests += 1

    async def submit(
        self,
        seq: int,
        decoder: CommandDecoder,
    ) -> None:
        decoded = decoder.decode()
        assert isinstance(decoded, Ok)
        self.submissions.append((seq, decoded.value))

    async def report_failure(self, failure: PlayerFailure) -> None:
        self.failures.append(failure)


def _view(
    *,
    seq: int,
    awaiting_action: None | str = None,
    error: str | None = None,
) -> PlayerView:
    if awaiting_action == "play":
        state = snapshot(
            awaiting_action="play",
            hand=(card("spades", "A"),),
            remaining_cards=seat_values(1, 1, 1, 1),
        )
    elif awaiting_action == "next_round":
        state = snapshot(
            phase="WAITING",
            awaiting_action="next_round",
        )
    else:
        assert awaiting_action is None
        state = snapshot(awaiting_action=None)
    return PlayerView(
        viewer=Seat.A,
        seq=seq,
        snapshot=state,
        status="running",
        error=error,
    )


async def _running_player(
    player: BotPlayer,
) -> tuple[asyncio.Task[None], _Channel, _Inbox]:
    channel = _Channel()
    inbox = _Inbox()
    task = asyncio.create_task(player.run(channel, inbox))
    _ = await channel.ready.wait()
    return task, channel, inbox


async def _finish(task: asyncio.Task[None], inbox: _Inbox) -> None:
    inbox.queue.put_nowait(None)
    await task


def test_factory_creates_ai_player_through_controller_boundary() -> (
    None
):
    controller: AIControllerPort = _AIController(
        decision=Ok(commands.PassBid()),
    )
    controllers = _AIControllerFactory(result=Ok(controller))
    factory = DefaultBotPlayerFactory(controllers)

    created = factory.create(Seat.C, "ai")

    assert isinstance(created, Ok)
    assert controllers.requested_seats == [Seat.C]
    assert created.value.lobby_status(None) == BotPlayerDescription(
        kind="bot",
        policy="ai",
    )


def test_factory_propagates_ai_checkpoint_rejection() -> None:
    controllers = _AIControllerFactory(
        result=Rejected("checkpoint unavailable")
    )
    factory = DefaultBotPlayerFactory(controllers)

    created = factory.create(Seat.D, "ai")

    assert isinstance(created, Rejected)
    assert created.reason == "checkpoint unavailable"


async def test_run_requests_state_and_observes_idle_view() -> None:
    policy = _Policy()
    player = BotPlayer(policy_name="auto", policy=policy)
    task, channel, inbox = await _running_player(player)

    inbox.queue.put_nowait(_view(seq=1, error="stale"))
    await asyncio.sleep(0)
    await _finish(task, inbox)

    assert channel.state_requests == 1
    assert [view.seq for view in policy.views] == [1]
    assert policy.views[0].error == "stale"
    assert policy.requests == []
    assert channel.submissions == []


async def test_run_confirms_round_without_policy_decision() -> None:
    policy = _Policy()
    player = BotPlayer(policy_name="auto", policy=policy)
    task, channel, inbox = await _running_player(player)

    inbox.queue.put_nowait(_view(seq=4, awaiting_action="next_round"))
    await asyncio.sleep(0)
    await _finish(task, inbox)

    assert policy.requests == []
    assert len(channel.submissions) == 1
    seq, command = channel.submissions[0]
    assert seq == 4
    assert isinstance(command, commands.ConfirmRound)


async def test_run_submits_exact_policy_command_and_sequence() -> None:
    command = commands.Play((CardId("d0-spades-A"),))
    policy = _Policy(decisions=[Ok(command)])
    player = BotPlayer(policy_name="ai", policy=policy)
    task, channel, inbox = await _running_player(player)

    inbox.queue.put_nowait(_view(seq=7, awaiting_action="play"))
    await asyncio.sleep(0)
    await _finish(task, inbox)

    assert len(policy.requests) == 1
    assert policy.requests[0].action == "play"
    assert channel.submissions == [(7, command)]


async def test_run_reports_decision_unavailable_without_fallback() -> (
    None
):
    policy = _Policy(
        decisions=[DecisionUnavailable(error="model unavailable")]
    )
    player = BotPlayer(policy_name="ai", policy=policy)
    task, channel, inbox = await _running_player(player)

    inbox.queue.put_nowait(_view(seq=8, awaiting_action="play"))
    await asyncio.sleep(0)
    await _finish(task, inbox)

    assert len(policy.requests) == 1
    assert channel.submissions == []
    assert channel.failures == [
        PlayerFailure(error="model unavailable")
    ]


async def test_run_propagates_observation_programming_error() -> None:
    policy = _Policy(observation_error="history discontinuity")
    player = BotPlayer(policy_name="ai", policy=policy)
    task, _channel, inbox = await _running_player(player)

    inbox.queue.put_nowait(_view(seq=9, awaiting_action="play"))
    await asyncio.sleep(0)

    assert task.done()
    failure = task.exception()
    assert isinstance(failure, AssertionError)
    assert str(failure) == "history discontinuity"
