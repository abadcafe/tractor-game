"""Black-box tests for the provider-neutral LLM decision policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from server.foundation.result import Ok, Rejected
from server.game import Seat, commands
from server.game.snapshots import PlayerSnapshot
from server.game_bots import DecisionRequest, StrategicAction
from server.game_bots.llm import (
    Decision,
    DecisionClient,
    DecisionPrompt,
    JSONObject,
    JSONValue,
    LLMConfig,
    LLMPolicy,
    LLMTranscript,
    ToolCall,
    ToolSpec,
    TranscriptRecordDict,
)
from server.game_runtime.player import PlayerView
from tests.support import card, seat_values, snapshot


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


def _request(
    state: PlayerSnapshot,
    action: StrategicAction,
    *,
    seq: int = 1,
    error: str | None = None,
) -> DecisionRequest:
    return DecisionRequest(
        view=PlayerView(
            viewer=Seat.A,
            seq=seq,
            snapshot=state,
            error=error,
        ),
        action=action,
    )


def _policy(
    client: DecisionClient,
    transcript: LLMTranscript,
    *,
    decision_retries: int = 1,
) -> LLMPolicy:
    return LLMPolicy(
        config=_config(decision_retries=decision_retries),
        client=client,
        transcript=transcript,
    )


async def test_policy_passes_bid_locally_without_provider() -> None:
    client = _Client()
    transcript = LLMTranscript()
    policy = _policy(client, transcript)
    state = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=(card("spades", "3"),),
        remaining_cards=seat_values(1, 0, 0, 0),
    )

    result = await policy.decide(_request(state, "bid"))

    assert isinstance(result, Ok)
    assert isinstance(result.value, commands.PassBid)
    assert client.prompts == []
    assert transcript.to_dict() == []


async def test_policy_asks_provider_at_bid_decision_count() -> None:
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
    transcript = LLMTranscript()
    policy = _policy(client, transcript)
    state = snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        hand=hand,
        remaining_cards=seat_values(5, 4, 4, 4),
    )

    result = await policy.decide(_request(state, "bid", seq=4))

    assert isinstance(result, Ok)
    assert isinstance(result.value, commands.RevealBid)
    assert tuple(result.value.card_ids) == (hand[0].id,)
    assert len(client.prompts) == 1
    assert client.tools[0][0].name == "bid_trump"
    assert client.tools[0][0].strict
    assert len(transcript.to_dict()) == 1


async def test_policy_tool_schema_uses_only_current_hand_ids() -> None:
    hand = (card("spades", "A"), card("hearts", "K"))
    client = _Client(
        results=[_tool_decision("play_cards", [hand[0].id])]
    )
    transcript = LLMTranscript()
    policy = _policy(client, transcript)
    state = snapshot(
        awaiting_action="play",
        hand=hand,
        remaining_cards=seat_values(2, 2, 2, 2),
    )

    result = await policy.decide(_request(state, "play"))

    assert isinstance(result, Ok)
    parameters = client.tools[0][0].parameters
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    card_ids = properties["card_ids"]
    assert isinstance(card_ids, dict)
    items = card_ids["items"]
    assert isinstance(items, dict)
    assert items["enum"] == [hand[0].id, hand[1].id]


async def test_policy_rejects_wrong_tool_and_records_result() -> None:
    hand = (card("spades", "A"),)
    client = _Client(
        results=[_tool_decision("bid_trump", [hand[0].id])]
    )
    transcript = LLMTranscript()
    policy = _policy(client, transcript, decision_retries=0)
    state = snapshot(
        awaiting_action="play",
        hand=hand,
        remaining_cards=seat_values(1, 1, 1, 1),
    )

    result = await policy.decide(_request(state, "play"))

    assert isinstance(result, Rejected)
    records = transcript.to_dict()
    assert len(records) == 1
    assert records[0]["tool_result"] is not None
    assert "rejected" in records[0]["tool_result"]


async def test_policy_repairs_invalid_card_id_in_same_decision() -> (
    None
):
    hand = (card("spades", "A"),)
    client = _Client(
        results=[
            _tool_decision("play_cards", ["unknown"]),
            _tool_decision("play_cards", [hand[0].id]),
        ]
    )
    transcript = LLMTranscript()
    policy = _policy(client, transcript)
    state = snapshot(
        awaiting_action="play",
        hand=hand,
        remaining_cards=seat_values(1, 1, 1, 1),
    )

    result = await policy.decide(_request(state, "play"))

    assert isinstance(result, Ok)
    assert isinstance(result.value, commands.Play)
    assert result.value.card_ids == (hand[0].id,)
    assert len(client.prompts) == 2
    assert "上一次动作被拒绝" in client.prompts[1].user
    assert len(transcript.to_dict()) == 2


async def test_policy_records_provider_rejection() -> None:
    client = _Client(results=[Rejected("provider failed")])
    transcript = LLMTranscript()
    policy = _policy(client, transcript, decision_retries=0)
    state = snapshot(
        awaiting_action="play",
        hand=(card("spades", "A"),),
        remaining_cards=seat_values(1, 1, 1, 1),
    )

    result = await policy.decide(_request(state, "play"))

    assert isinstance(result, Rejected)
    record = transcript.to_dict()[0]
    assert record["api_error"] == "provider failed"
    assert record["tool_result"] is None


async def test_observe_marks_rejection_and_preserves_attempt() -> None:
    hand = (card("spades", "A"),)
    client = _Client(
        results=[
            _tool_decision("play_cards", [hand[0].id]),
            _tool_decision("play_cards", [hand[0].id]),
        ]
    )
    transcript = LLMTranscript()
    policy = _policy(client, transcript)
    state = snapshot(
        awaiting_action="play",
        hand=hand,
        remaining_cards=seat_values(1, 1, 1, 1),
    )
    request = _request(state, "play", seq=9)
    updates = transcript.subscribe()
    first = await policy.decide(request)
    assert isinstance(first, Ok)

    observed = policy.observe(
        PlayerView(
            viewer=Seat.A,
            seq=9,
            snapshot=state,
            error="follow rejected",
        )
    )
    second = await policy.decide(request)

    assert isinstance(observed, Ok)
    assert isinstance(second, Ok)
    records = transcript.to_dict()
    assert len(records) == 2
    assert records[0]["tool_result"] is not None
    assert "follow rejected" in records[0]["tool_result"]
    assert records[1]["attempt"] == 2
    streamed: list[TranscriptRecordDict] = []
    while not updates.empty():
        streamed.append(updates.get_nowait())
    assert len(streamed) == 3
    transcript.unsubscribe(updates)


async def test_observe_completed_sequence_releases_attempt_state() -> (
    None
):
    hand = (card("spades", "A"),)
    client = _Client(
        results=[
            _tool_decision("play_cards", [hand[0].id]),
            _tool_decision("play_cards", [hand[0].id]),
        ]
    )
    transcript = LLMTranscript()
    policy = _policy(client, transcript)
    state = snapshot(
        awaiting_action="play",
        hand=hand,
        remaining_cards=seat_values(1, 1, 1, 1),
    )
    first = await policy.decide(_request(state, "play", seq=3))
    assert isinstance(first, Ok)

    observed = policy.observe(
        PlayerView(
            viewer=Seat.A,
            seq=4,
            snapshot=state,
            error=None,
        )
    )
    second = await policy.decide(_request(state, "play", seq=4))

    assert isinstance(observed, Ok)
    assert isinstance(second, Ok)
    assert transcript.to_dict()[1]["attempt"] == 1
