"""Black-box tests for the provider-neutral LLM game agent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from server.foundation.result import Ok, Rejected
from server.game import Seat, commands
from server.game_agents.llm import (
    Decision,
    DecisionClient,
    DecisionPrompt,
    JSONObject,
    JSONValue,
    LLMAgent,
    LLMConfig,
    ToolCall,
    ToolSpec,
    TranscriptRecordDict,
)
from server.game_runtime.session import CommandDecoder, Delivery
from tests.support import card, snapshot


def _config(*, decision_retries: int = 1) -> LLMConfig:
    return LLMConfig(
        provider="openai",
        base_url="https://example.invalid/v1",
        api_key=None,
        model="test",
        http_timeout_seconds=1.0,
        http_max_retries=0,
        http_retry_delay_seconds=0.0,
        decision_max_retries=decision_retries,
        max_output_tokens=256,
    )


def _submissions() -> list[tuple[int, commands.Command]]:
    return []


@dataclass(slots=True)
class _Target:
    submissions: list[tuple[int, commands.Command]] = field(
        default_factory=_submissions
    )

    async def submit(
        self,
        seq: int,
        decoder: CommandDecoder,
    ) -> None:
        result = decoder.decode()
        assert isinstance(result, Ok)
        self.submissions.append((seq, result.value))


type _DecisionResult = Ok[Decision] | Rejected


def _decision_results() -> list[_DecisionResult]:
    return []


def _prompts() -> list[DecisionPrompt]:
    return []


def _tools() -> list[list[ToolSpec]]:
    return []


@dataclass(slots=True)
class _Client:
    results: list[_DecisionResult] = field(
        default_factory=_decision_results
    )
    prompts: list[DecisionPrompt] = field(default_factory=_prompts)
    tools: list[list[ToolSpec]] = field(default_factory=_tools)

    async def decide(
        self,
        prompt: DecisionPrompt,
        tools: list[ToolSpec],
    ) -> Ok[Decision] | Rejected:
        self.prompts.append(prompt)
        self.tools.append(tools)
        assert self.results
        return self.results.pop(0)


def _tool_decision(
    name: str,
    card_ids: list[str],
) -> Ok[Decision]:
    values: list[JSONValue] = []
    values.extend(card_ids)
    arguments: JSONObject = {"card_ids": values}
    return Ok(
        Decision(
            assistant_content=None,
            tool_call=ToolCall(
                name=name,
                arguments=arguments,
            ),
        )
    )


async def _start(
    client: DecisionClient,
    target: _Target,
    *,
    decision_retries: int = 1,
) -> LLMAgent:
    agent = LLMAgent(
        Seat.NORTH,
        target,
        config=_config(decision_retries=decision_retries),
        client=client,
    )
    starting = asyncio.create_task(agent.start())
    await asyncio.sleep(0)
    await agent.offer(
        Delivery(
            viewer=Seat.NORTH,
            seq=1,
            snapshot=snapshot(awaiting_action=None),
            error=None,
        )
    )
    await starting
    return agent


async def test_agent_requests_initial_state_with_sequence_zero() -> (
    None
):
    client = _Client()
    target = _Target()
    agent = await _start(client, target)

    await agent.stop()

    assert target.submissions[0][0] == 0
    assert isinstance(target.submissions[0][1], commands.ConfirmRound)
    assert client.prompts == []


async def test_agent_handles_next_round_without_llm() -> None:
    client = _Client()
    target = _Target()
    agent = await _start(client, target)

    await agent.offer(
        Delivery(
            viewer=Seat.NORTH,
            seq=2,
            snapshot=snapshot(
                phase="WAITING",
                awaiting_action="next_round",
            ),
            error=None,
        )
    )
    await agent.stop()

    assert isinstance(target.submissions[-1][1], commands.ConfirmRound)
    assert target.submissions[-1][0] == 2
    assert client.prompts == []
    assert agent.transcript() == []


async def test_agent_passes_bid_locally_without_legal_reveal() -> None:
    client = _Client()
    target = _Target()
    agent = await _start(client, target)

    await agent.offer(
        Delivery(
            viewer=Seat.NORTH,
            seq=3,
            snapshot=snapshot(
                phase="DEAL_BID",
                awaiting_action="bid",
                player_hand=(card("spades", "3"),),
                player_hand_counts=(1, 0, 0, 0),
            ),
            error=None,
        )
    )
    await agent.stop()

    assert isinstance(target.submissions[-1][1], commands.PassBid)
    assert client.prompts == []


async def test_agent_asks_llm_at_bid_decision_count() -> None:
    hand = (
        card("spades", "2"),
        card("hearts", "3"),
        card("clubs", "4"),
        card("diamonds", "5"),
        card("spades", "6"),
    )
    client = _Client(
        results=[_tool_decision("bid_trump", [hand[0].id])]
    )
    target = _Target()
    agent = await _start(client, target)

    await agent.offer(
        Delivery(
            viewer=Seat.NORTH,
            seq=4,
            snapshot=snapshot(
                phase="DEAL_BID",
                awaiting_action="bid",
                player_hand=hand,
                player_hand_counts=(5, 4, 4, 4),
            ),
            error=None,
        )
    )
    await agent.stop()

    assert len(client.prompts) == 1
    assert len(client.tools) == 1
    assert client.tools[0][0].name == "bid_trump"
    assert client.tools[0][0].strict
    command = target.submissions[-1][1]
    assert isinstance(command, commands.RevealBid)
    assert tuple(command.card_ids) == (hand[0].id,)


async def test_agent_tool_schema_uses_only_current_hand_ids() -> None:
    hand = (card("spades", "A"), card("hearts", "K"))
    client = _Client(
        results=[_tool_decision("play_cards", [hand[0].id])]
    )
    target = _Target()
    agent = await _start(client, target)

    await agent.offer(
        Delivery(
            viewer=Seat.NORTH,
            seq=5,
            snapshot=snapshot(
                awaiting_action="play",
                player_hand=hand,
                player_hand_counts=(2, 2, 2, 2),
            ),
            error=None,
        )
    )
    await agent.stop()

    parameters = client.tools[0][0].parameters
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    card_ids = properties["card_ids"]
    assert isinstance(card_ids, dict)
    items = card_ids["items"]
    assert isinstance(items, dict)
    assert items["enum"] == [hand[0].id, hand[1].id]


async def test_agent_rejects_wrong_tool_and_records_result() -> None:
    hand = (card("spades", "A"),)
    client = _Client(
        results=[_tool_decision("bid_trump", [hand[0].id])]
    )
    target = _Target()
    agent = await _start(
        client,
        target,
        decision_retries=0,
    )

    await agent.offer(
        Delivery(
            viewer=Seat.NORTH,
            seq=6,
            snapshot=snapshot(
                awaiting_action="play",
                player_hand=hand,
                player_hand_counts=(1, 1, 1, 1),
            ),
            error=None,
        )
    )
    await agent.stop()

    assert len(target.submissions) == 1
    records = agent.transcript()
    assert len(records) == 1
    assert records[0]["tool_result"] is not None
    assert "rejected" in records[0]["tool_result"]


async def test_agent_repairs_invalid_card_id_once() -> None:
    hand = (card("spades", "A"),)
    client = _Client(
        results=[
            _tool_decision("play_cards", ["unknown"]),
            _tool_decision("play_cards", [hand[0].id]),
        ]
    )
    target = _Target()
    agent = await _start(client, target)

    await agent.offer(
        Delivery(
            viewer=Seat.NORTH,
            seq=7,
            snapshot=snapshot(
                awaiting_action="play",
                player_hand=hand,
                player_hand_counts=(1, 1, 1, 1),
            ),
            error=None,
        )
    )
    for _ in range(10):
        if len(client.prompts) == 2:
            break
        await asyncio.sleep(0)
    await agent.stop()

    assert len(client.prompts) == 2
    assert "上一次动作被拒绝" in client.prompts[1].user
    command = target.submissions[-1][1]
    assert isinstance(command, commands.Play)
    assert command.card_ids == (hand[0].id,)
    assert len(agent.transcript()) == 2


async def test_agent_records_client_rejection_without_submission() -> (
    None
):
    client = _Client(results=[Rejected("provider failed")])
    target = _Target()
    agent = await _start(
        client,
        target,
        decision_retries=0,
    )

    await agent.offer(
        Delivery(
            viewer=Seat.NORTH,
            seq=8,
            snapshot=snapshot(
                awaiting_action="play",
                player_hand=(card("spades", "A"),),
                player_hand_counts=(1, 1, 1, 1),
            ),
            error=None,
        )
    )
    await agent.stop()

    assert len(target.submissions) == 1
    record = agent.transcript()[0]
    assert record["api_error"] == "provider failed"
    assert record["tool_result"] is None


async def test_agent_marks_server_rejection_and_retries_same_seq() -> (
    None
):
    hand = (card("spades", "A"),)
    client = _Client(
        results=[
            _tool_decision("play_cards", [hand[0].id]),
            _tool_decision("play_cards", [hand[0].id]),
        ]
    )
    target = _Target()
    agent = await _start(client, target)
    live = snapshot(
        awaiting_action="play",
        player_hand=hand,
        player_hand_counts=(1, 1, 1, 1),
    )
    updates = agent.subscribe_transcript()

    await agent.offer(
        Delivery(
            viewer=Seat.NORTH,
            seq=9,
            snapshot=live,
            error=None,
        )
    )
    await asyncio.sleep(0)
    await agent.offer(
        Delivery(
            viewer=Seat.NORTH,
            seq=9,
            snapshot=live,
            error="follow rejected",
        )
    )
    await agent.stop()

    records = agent.transcript()
    assert len(records) == 2
    assert records[0]["tool_result"] is not None
    assert "follow rejected" in records[0]["tool_result"]
    assert records[1]["attempt"] == 2
    streamed: list[TranscriptRecordDict] = []
    while not updates.empty():
        streamed.append(updates.get_nowait())
    assert len(streamed) == 3
    agent.unsubscribe_transcript(updates)
