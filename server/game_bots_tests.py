"""Black-box tests for the shared BotPlayer runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from server.foundation.result import Ok, Rejected
from server.game import Seat, commands
from server.game.rules.cards import CardId
from server.game_bots import (
    BotPlayer,
    DecisionCommand,
    DecisionRequest,
)
from server.game_runtime.player import (
    BotPlayerDescription,
    CommandDecoder,
    PlayerView,
)
from tests.support import card, seat_values, snapshot


def _views() -> list[PlayerView]:
    return []


def _requests() -> list[DecisionRequest]:
    return []


def _decisions() -> list[Ok[DecisionCommand] | Rejected]:
    return []


def _submissions() -> list[tuple[int, commands.Command]]:
    return []


@dataclass(slots=True)
class _Policy:
    observation_result: Ok[None] | Rejected = field(
        default_factory=lambda: Ok(None)
    )
    decisions: list[Ok[DecisionCommand] | Rejected] = field(
        default_factory=_decisions
    )
    views: list[PlayerView] = field(default_factory=_views)
    requests: list[DecisionRequest] = field(default_factory=_requests)

    def observe(self, view: PlayerView) -> Ok[None] | Rejected:
        self.views.append(view)
        return self.observation_result

    async def decide(
        self,
        request: DecisionRequest,
    ) -> Ok[DecisionCommand] | Rejected:
        self.requests.append(request)
        assert self.decisions
        return self.decisions.pop(0)


@dataclass(slots=True)
class _Channel:
    state_requests: int = 0
    submissions: list[tuple[int, commands.Command]] = field(
        default_factory=_submissions
    )

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


def _view(
    *,
    seq: int,
    awaiting_action: (None | str) = None,
    error: str | None = None,
) -> PlayerView:
    if awaiting_action == "play":
        hand = (card("spades", "A"),)
        state = snapshot(
            awaiting_action="play",
            hand=hand,
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
        error=error,
    )


async def _settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


async def test_player_requests_state_and_observes_idle_view() -> None:
    policy = _Policy()
    channel = _Channel()
    player = BotPlayer(
        policy_name="auto",
        policy=policy,
    )

    ready = asyncio.create_task(player.start(channel))
    await asyncio.sleep(0)
    await player.update(_view(seq=1, error="stale"))
    await ready
    await player.stop()

    assert channel.state_requests == 1
    assert [view.seq for view in policy.views] == [1]
    assert policy.views[0].error == "stale"
    assert policy.requests == []
    assert channel.submissions == []


async def test_player_confirms_round_without_policy_decision() -> None:
    policy = _Policy()
    channel = _Channel()
    player = BotPlayer(
        policy_name="auto",
        policy=policy,
    )
    start = asyncio.create_task(player.start(channel))
    await asyncio.sleep(0)

    await player.update(_view(seq=4, awaiting_action="next_round"))
    await start
    await _settle()
    await player.stop()

    assert policy.requests == []
    assert len(channel.submissions) == 1
    seq, command = channel.submissions[0]
    assert seq == 4
    assert isinstance(command, commands.ConfirmRound)


async def test_player_submits_exact_policy_command_and_sequence() -> (
    None
):
    command = commands.Play((CardId("d0-spades-A"),))
    policy = _Policy(decisions=[Ok(command)])
    channel = _Channel()
    player = BotPlayer(
        policy_name="llm",
        policy=policy,
    )
    start = asyncio.create_task(player.start(channel))
    await asyncio.sleep(0)

    await player.update(_view(seq=7, awaiting_action="play"))
    await start
    await _settle()
    await player.stop()

    assert len(policy.requests) == 1
    assert policy.requests[0].action == "play"
    assert channel.submissions == [(7, command)]
    description = player.lobby_status(None)
    assert description == BotPlayerDescription(
        kind="bot",
        policy="llm",
    )


async def test_policy_rejection_has_no_fallback() -> None:
    policy = _Policy(decisions=[Rejected("model unavailable")])
    channel = _Channel()
    player = BotPlayer(
        policy_name="llm",
        policy=policy,
    )
    start = asyncio.create_task(player.start(channel))
    await asyncio.sleep(0)

    await player.update(_view(seq=8, awaiting_action="play"))
    await start
    await _settle()
    await player.stop()

    assert len(policy.requests) == 1
    assert channel.submissions == []


async def test_observation_rejection_prevents_decision() -> None:
    policy = _Policy(
        observation_result=Rejected("history discontinuity"),
        decisions=[
            Ok(commands.Play((CardId("d0-spades-A"),))),
        ],
    )
    channel = _Channel()
    player = BotPlayer(
        policy_name="llm",
        policy=policy,
    )
    start = asyncio.create_task(player.start(channel))
    await asyncio.sleep(0)

    await player.update(_view(seq=9, awaiting_action="play"))
    await start
    await _settle()
    await player.stop()

    assert policy.requests == []
    assert channel.submissions == []


async def test_stop_before_start_and_after_start_is_idempotent() -> (
    None
):
    unstarted = BotPlayer(
        policy_name="auto",
        policy=_Policy(),
    )
    await unstarted.stop()
    await unstarted.stop()
    running = BotPlayer(
        policy_name="auto",
        policy=_Policy(),
    )
    channel = _Channel()
    start = asyncio.create_task(running.start(channel))
    await asyncio.sleep(0)
    await running.update(_view(seq=1))
    await start

    await running.stop()
    await running.stop()

    assert channel.state_requests == 1
