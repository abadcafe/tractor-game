"""Tests for AIPlayer type boundaries."""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest

from server.protocol import TrickSlotSnapshot, TrickSnapshot
from server.result import Ok, Rejected

from . import ai, auto, base
from .ai import config as ai_config
from .ai.client import (
    AIClient,
    AIClientRejected,
    AIDecision,
    AIDecisionPrompt,
    AIToolCall,
    AIToolSpec,
    JSONObject,
    is_json_object,
)
from .ai.config import AIConfig
from .ai.memory import AIMemory
from .ai.openai_client import (
    OpenAIChatCompletionsClient,
    build_chat_completions_payload,
    chat_completion_message_log,
    extract_chat_completion_tool_call,
)
from .ai.prompt import RuleBook, build_decision_prompt
from .ai.rejections import AIToolRejected
from .ai.tools import allowed_tool_specs, tool_call_to_message
from .test_helpers import (
    card,
    make_game,
    make_snapshot,
    make_state_message,
)


def test_ai_player_is_player() -> None:
    player = ai.AIPlayer(index=0)
    assert isinstance(player, base.Player)


def test_ai_player_is_not_auto_player() -> None:
    player = ai.AIPlayer(index=0)
    assert isinstance(player, ai.AIPlayer)
    assert not isinstance(player, auto.AutoPlayer)
    assert type(player) is ai.AIPlayer


def test_ai_decision_prompt_uses_chinese_game_labels() -> None:
    test_card = card("hearts", "A")
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        trump_suit="spades",
        trump_rank="2",
        player_hand=[test_card],
        player_hand_counts=[10, 9, 8, 7],
        declarer_team=0,
        declarer_player=2,
        defender_points=35,
    )
    rules = RuleBook({"common": "- 测试规则"})

    prompt = build_decision_prompt(
        player_index=1,
        snapshot=snap,
        memory=AIMemory(),
        rules=rules,
    )

    assert "- 你是：玩家 1" in prompt.user
    assert "- 阶段：出牌阶段" in prompt.user
    assert "- 当前需要你：出牌" in prompt.user
    assert "- 主花色：黑桃" in prompt.user
    assert "当前墩：无" in prompt.user
    assert "合法动作约束（legal_action_groups）：无" in prompt.user
    assert "phase:" not in prompt.user
    assert "awaiting_action:" not in prompt.user
    assert "trump_suit:" not in prompt.user
    assert "not_played" not in prompt.user


def test_ai_rulebook_selects_common_play_lead_and_scoring() -> None:
    rules = RuleBook(
        {
            "common": "common rules",
            "play": "play rules",
            "play_lead": "lead rules",
            "play_follow": "follow rules",
            "scoring": "scoring rules",
        }
    )
    snap = make_snapshot(phase="PLAYING", awaiting_action="play")

    selected = rules.select(snap)

    assert "规则: common\ncommon rules" in selected
    assert "规则: play\nplay rules" in selected
    assert "规则: play_lead\nlead rules" in selected
    assert "规则: scoring\nscoring rules" in selected
    assert "follow rules" not in selected


def test_ai_rulebook_selects_common_play_follow_and_scoring() -> None:
    rules = RuleBook(
        {
            "common": "common rules",
            "play": "play rules",
            "play_lead": "lead rules",
            "play_follow": "follow rules",
            "scoring": "scoring rules",
        }
    )
    lead_card = card("hearts", "A")
    trick = TrickSnapshot(
        lead_player=0,
        current_player=1,
        slots=[
            TrickSlotSnapshot(player=0, cards=[lead_card]),
            TrickSlotSnapshot(player=1, cards=[]),
            TrickSlotSnapshot(player=2, cards=[]),
            TrickSlotSnapshot(player=3, cards=[]),
        ],
    )
    snap = make_snapshot(
        phase="PLAYING", awaiting_action="play", trick=trick
    )

    selected = rules.select(snap)

    assert "规则: common\ncommon rules" in selected
    assert "规则: play\nplay rules" in selected
    assert "规则: play_follow\nfollow rules" in selected
    assert "规则: scoring\nscoring rules" in selected
    assert "lead rules" not in selected


@pytest.mark.asyncio
async def test_ai_player_run_requests_state() -> None:
    game = make_game()
    player = ai.AIPlayer(index=0, config=_config())

    await player.run(game)

    game.receive.assert_awaited()
    assert game.receive.call_args[0][0] == 0
    assert game.receive.call_args[0][1].seq == 0


@pytest.mark.asyncio
async def test_ai_next_round_confirms_locally() -> None:
    snap = make_snapshot(phase="WAITING", awaiting_action="next_round")
    game = make_game(snap)
    client = StaticAIClient(
        AIToolCall(
            name="unused_tool",
            arguments={"reason": "should not be used"},
        )
    )
    player = ai.AIPlayer(index=0, config=_config(), client=client)

    await player.on_state(game, make_state_message(snap, seq=3))
    await asyncio.sleep(0.05)

    game.receive.assert_awaited()
    assert game.receive.call_args[0][0] == 0
    assert game.receive.call_args[0][1].seq == 3
    assert game.receive.call_args[0][1].raw == {"type": "next_round"}
    assert client.prompts == []


@pytest.mark.asyncio
async def test_ai_player_does_not_log_local_action_submission() -> None:
    snap = make_snapshot(phase="WAITING", awaiting_action="next_round")
    game = make_game(snap)
    client = StaticAIClient(
        AIToolCall(
            name="unused_tool",
            arguments={"reason": "should not be used"},
        )
    )
    player = ai.AIPlayer(index=0, config=_config(), client=client)

    logs = ListLogHandler()
    target_logger = logging.getLogger("server.player.ai.player")
    old_level = target_logger.level
    target_logger.setLevel(logging.INFO)
    target_logger.addHandler(logs)
    try:
        await player.on_state(game, make_state_message(snap, seq=3))
        await asyncio.sleep(0.05)
    finally:
        target_logger.removeHandler(logs)
        target_logger.setLevel(old_level)

    assert logs.text() == ""


@pytest.mark.asyncio
async def test_ai_bid_without_hint_passes_locally() -> None:
    snap = make_snapshot(phase="DEAL_BID", awaiting_action="bid")
    game = make_game(snap)
    client = StaticAIClient(
        AIToolCall(
            name="unused_tool",
            arguments={"reason": "should not be used"},
        )
    )
    player = ai.AIPlayer(index=0, config=_config(), client=client)

    await player.on_state(game, make_state_message(snap, seq=4))
    await asyncio.sleep(0.05)

    game.receive.assert_awaited()
    assert game.receive.call_args[0][0] == 0
    assert game.receive.call_args[0][1].seq == 4
    assert game.receive.call_args[0][1].raw == {
        "type": "bid",
        "pass": True,
    }
    assert client.prompts == []


@pytest.mark.asyncio
async def test_ai_bid_with_hints_uses_first_server_hint() -> None:
    spade_two = card("spades", "2")
    diamond_two = card("diamonds", "2")
    snap = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[spade_two, diamond_two],
        action_hints=[[spade_two], [diamond_two]],
        trump_rank="2",
    )
    game = make_game(snap)
    client = StaticAIClient(
        AIToolCall(
            name="unused_tool",
            arguments={"reason": "should not be used"},
        )
    )
    player = ai.AIPlayer(index=0, config=_config(), client=client)

    await player.on_state(game, make_state_message(snap, seq=5))
    await asyncio.sleep(0.05)

    game.receive.assert_awaited()
    assert game.receive.call_args[0][0] == 0
    assert game.receive.call_args[0][1].seq == 5
    assert game.receive.call_args[0][1].raw == {
        "type": "bid",
        "cards": [spade_two["id"]],
    }
    assert client.prompts == []


@pytest.mark.asyncio
async def test_ai_stir_without_hint_passes_locally() -> None:
    snap = make_snapshot(phase="STIRRING", awaiting_action="stir")
    game = make_game(snap)
    client = StaticAIClient(
        AIToolCall(
            name="unused_tool",
            arguments={"reason": "should not be used"},
        )
    )
    player = ai.AIPlayer(index=0, config=_config(), client=client)

    await player.on_state(game, make_state_message(snap, seq=6))
    await asyncio.sleep(0.05)

    game.receive.assert_awaited()
    assert game.receive.call_args[0][0] == 0
    assert game.receive.call_args[0][1].seq == 6
    assert game.receive.call_args[0][1].raw == {
        "type": "stir",
        "pass": True,
    }
    assert client.prompts == []


@pytest.mark.asyncio
async def test_ai_player_stir_with_hint_uses_llm() -> None:
    trump_pair = [
        card("hearts", "2", deck=1),
        card("hearts", "2", deck=2),
    ]
    snap = make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        player_hand=trump_pair,
        action_hints=[trump_pair],
        trump_rank="2",
    )
    game = make_game(snap)
    client = StaticAIClient(AIToolCall(name="stir_trump", arguments={}))
    player = ai.AIPlayer(index=0, config=_config(), client=client)

    await player.on_state(game, make_state_message(snap, seq=7))
    await asyncio.sleep(0.05)

    game.receive.assert_awaited()
    assert game.receive.call_args[0][0] == 0
    assert game.receive.call_args[0][1].seq == 7
    assert game.receive.call_args[0][1].raw == {
        "type": "stir",
        "pass": True,
    }
    assert len(client.prompts) == 1


@pytest.mark.asyncio
async def test_ai_player_llm_tool_call_submits_play() -> None:
    test_card = card("hearts", "A")
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[test_card],
    )
    game = make_game(snap)
    client = StaticAIClient(
        AIToolCall(
            name="play_cards",
            arguments={
                "card_ids": [test_card["id"]],
                "reason": "test play",
            },
        )
    )
    player = ai.AIPlayer(index=0, config=_config(), client=client)

    await player.on_state(game, make_state_message(snap, seq=5))
    await asyncio.sleep(0.05)

    game.receive.assert_awaited()
    assert game.receive.call_args[0][0] == 0
    assert game.receive.call_args[0][1].seq == 5
    assert game.receive.call_args[0][1].raw == {
        "type": "play",
        "cards": [test_card["id"]],
    }


@pytest.mark.asyncio
async def test_ai_player_does_not_log_llm_transcript_to_terminal() -> (
    None
):
    test_card = card("hearts", "A")
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[test_card],
    )
    game = make_game(snap)
    client = StaticAIClient(
        AIToolCall(
            name="play_cards",
            arguments={
                "card_ids": [test_card["id"]],
                "reason": "test play",
            },
        )
    )
    player = ai.AIPlayer(index=0, config=_config(), client=client)

    logs = ListLogHandler()
    target_logger = logging.getLogger("server.player.ai.player")
    old_level = target_logger.level
    target_logger.setLevel(logging.INFO)
    target_logger.addHandler(logs)
    try:
        await player.on_state(game, make_state_message(snap, seq=5))
        await asyncio.sleep(0.05)
    finally:
        target_logger.removeHandler(logs)
        target_logger.setLevel(old_level)

    assert logs.text() == ""


@pytest.mark.asyncio
async def test_ai_player_records_debug_transcript() -> None:
    test_card = card("hearts", "A")
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[test_card],
    )
    game = make_game(snap)
    client = StaticAIClient(
        AIToolCall(
            name="play_cards",
            arguments={
                "card_ids": [test_card["id"]],
                "reason": "test play",
            },
        )
    )
    player = ai.AIPlayer(index=0, config=_config(), client=client)

    await player.on_state(game, make_state_message(snap, seq=5))
    await asyncio.sleep(0.05)

    transcript = player.transcript()
    assert len(transcript) == 1
    decision = transcript[0]
    assert decision["player_index"] == 0
    assert decision["seq"] == 5
    assert decision["attempt"] == 1
    assert decision["api_request"] is None
    assert decision["api_response"] is None
    assert decision["api_error"] is None
    assert decision["tool_result"] is not None
    assert "accepted" in decision["tool_result"]
    assert test_card["id"] in decision["tool_result"]


@pytest.mark.asyncio
async def test_ai_player_records_debug_transcript_stream_messages() -> (
    None
):
    test_card = card("hearts", "A")
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[test_card],
    )
    game = make_game(snap)
    client = StaticAIClient(
        AIToolCall(
            name="play_cards",
            arguments={
                "card_ids": [test_card["id"]],
                "reason": "test play",
            },
        )
    )
    player = ai.AIPlayer(index=0, config=_config(), client=client)
    queue = player.subscribe_transcript()

    try:
        await player.on_state(game, make_state_message(snap, seq=5))
        first_message = await asyncio.wait_for(queue.get(), timeout=1.0)
        await asyncio.sleep(0.05)
    finally:
        player.unsubscribe_transcript(queue)

    stream = player.transcript_stream()
    assert len(stream) == 1
    assert first_message == stream[0]
    assert stream[0]["event_id"] == 1
    assert stream[0]["id"] == 1
    assert stream[0]["tool_result"] is not None
    assert "accepted" in stream[0]["tool_result"]


@pytest.mark.asyncio
async def test_ai_records_openai_response_without_tool_calls() -> None:
    test_card = card("hearts", "A")
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[test_card],
    )
    game = make_game(snap)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=json.dumps(
                {
                    "id": "chatcmpl-no-tool",
                    "model": "test-model",
                    "choices": [
                        {
                            "message": {
                                "content": "plain non-tool answer"
                            }
                        }
                    ],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    client = OpenAIChatCompletionsClient(
        _config(),
        transport=httpx.MockTransport(handler),
    )
    player = ai.AIPlayer(index=0, config=_config(), client=client)

    await player.on_state(game, make_state_message(snap, seq=15))
    await asyncio.sleep(0.05)

    game.receive.assert_not_awaited()
    transcript = player.transcript()
    assert len(transcript) == 1
    record = transcript[0]
    assert record["api_request"] is not None
    assert record["api_response"] is not None
    assert record["api_error"] is not None
    assert record["tool_result"] is None
    assert "plain non-tool answer" in record["api_response"]
    assert (
        "OpenAI-compatible message has no tool_calls list"
        in record["api_error"]
    )


@pytest.mark.asyncio
async def test_ai_player_records_openai_network_failure() -> None:
    test_card = card("hearts", "A")
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[test_card],
    )
    game = make_game(snap)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connect failed", request=request)

    config = _config(max_retries=1)
    client = OpenAIChatCompletionsClient(
        config,
        transport=httpx.MockTransport(handler),
    )
    player = ai.AIPlayer(index=0, config=config, client=client)

    await player.on_state(game, make_state_message(snap, seq=16))
    await asyncio.sleep(0.05)

    game.receive.assert_not_awaited()
    transcript = player.transcript()
    assert len(transcript) == 1
    record = transcript[0]
    assert record["api_request"] is not None
    assert record["api_response"] is None
    assert record["api_error"] is not None
    assert record["tool_result"] is None
    assert "connect failed" in record["api_error"]
    assert "OpenAI-compatible network error" in record["api_error"]


@pytest.mark.asyncio
async def test_ai_records_debug_without_tool_use_logging() -> None:
    test_card = card("hearts", "A")
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[test_card],
    )
    game = make_game(snap)
    client = StaticAIClient(
        AIToolCall(
            name="play_cards",
            arguments={
                "card_ids": [test_card["id"]],
                "reason": "test play",
            },
        )
    )
    player = ai.AIPlayer(
        index=0, config=_config(log_tool_use=False), client=client
    )

    await player.on_state(game, make_state_message(snap, seq=5))
    await asyncio.sleep(0.05)

    transcript = player.transcript()
    assert len(transcript) == 1
    assert transcript[0]["tool_result"] is not None
    assert "accepted" in transcript[0]["tool_result"]


@pytest.mark.asyncio
async def test_ai_player_llm_failure_does_not_submit_action() -> None:
    test_card = card("hearts", "A")
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[test_card],
    )
    game = make_game(snap)
    client = StaticAIClient(Rejected("llm unavailable"))
    player = ai.AIPlayer(index=0, config=_config(), client=client)

    await player.on_state(game, make_state_message(snap, seq=8))
    await asyncio.sleep(0.05)

    game.receive.assert_not_awaited()
    assert len(client.prompts) == 1


@pytest.mark.asyncio
async def test_ai_records_server_rejection_in_debug() -> None:
    snap = make_snapshot(phase="PLAYING", awaiting_action=None)
    game = make_game(snap)
    player = ai.AIPlayer(
        index=0,
        config=_config(),
        client=StaticAIClient(Rejected("unused")),
    )

    await player.on_state(
        game, make_state_message(snap, seq=11, error="illegal play")
    )

    transcript = player.transcript()
    assert len(transcript) == 1
    assert transcript[0]["tool_result"] is not None
    tool_result = _json_object(transcript[0]["tool_result"])
    assert tool_result["status"] == "rejected"
    assert tool_result["error_type"] == "rule"
    assert tool_result["reason"] == "illegal play"
    assert "repair" in tool_result
    assert "stage" not in tool_result
    game.receive.assert_not_awaited()


@pytest.mark.asyncio
async def test_ai_player_repairs_server_rejected_tool_call() -> None:
    first_card = card("hearts", "A")
    repaired_card = card("diamonds", "K")
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[first_card, repaired_card],
    )
    game = make_game(snap)
    client = SequenceAIClient(
        [
            AIToolCall(
                name="play_cards",
                arguments={
                    "card_ids": [first_card["id"]],
                    "reason": "first try",
                },
            ),
            AIToolCall(
                name="play_cards",
                arguments={
                    "card_ids": [repaired_card["id"]],
                    "reason": "server repair",
                },
            ),
        ]
    )
    player = ai.AIPlayer(index=0, config=_config(), client=client)

    await player.on_state(game, make_state_message(snap, seq=13))
    await asyncio.sleep(0.05)
    await player.on_state(
        game, make_state_message(snap, seq=13, error="illegal play")
    )
    await asyncio.sleep(0.05)

    assert game.receive.await_count == 2
    first_message = game.receive.await_args_list[0].args[1]
    second_message = game.receive.await_args_list[1].args[1]
    assert first_message.seq == 13
    assert first_message.raw == {
        "type": "play",
        "cards": [first_card["id"]],
    }
    assert second_message.seq == 13
    assert second_message.raw == {
        "type": "play",
        "cards": [repaired_card["id"]],
    }
    assert len(client.prompts) == 2
    assert "illegal play" in client.prompts[1].user
    assert "错误类型（error_type）：rule" in client.prompts[1].user

    transcript = player.transcript()
    assert len(transcript) == 2
    assert transcript[0]["attempt"] == 1
    assert transcript[1]["attempt"] == 2
    assert transcript[0]["tool_result"] is not None
    tool_result = _json_object(transcript[0]["tool_result"])
    assert tool_result["status"] == "rejected"
    assert tool_result["error_type"] == "rule"
    assert tool_result["reason"] == "illegal play"
    assert "stage" not in tool_result


@pytest.mark.asyncio
async def test_ai_player_repairs_play_not_matching_action_hint() -> (
    None
):
    hint_card_1 = card("diamonds", "2")
    hint_card_2 = card("hearts", "10")
    other_card = card("spades", "A")
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[hint_card_1, hint_card_2, other_card],
        action_hints=[[hint_card_1, hint_card_2]],
    )
    game = make_game(snap)
    client = SequenceAIClient(
        [
            AIToolCall(
                name="play_cards",
                arguments={
                    "card_ids": [hint_card_2["id"]],
                    "reason": "partial hint",
                },
            ),
            AIToolCall(
                name="play_cards",
                arguments={
                    "card_ids": [hint_card_1["id"], hint_card_2["id"]],
                    "reason": "full hint",
                },
            ),
        ]
    )
    player = ai.AIPlayer(index=0, config=_config(), client=client)

    await player.on_state(game, make_state_message(snap, seq=12))
    await asyncio.sleep(0.05)

    game.receive.assert_awaited()
    assert game.receive.call_args[0][1].seq == 12
    assert game.receive.call_args[0][1].raw == {
        "type": "play",
        "cards": [hint_card_1["id"], hint_card_2["id"]],
    }
    assert len(client.prompts) == 2
    assert "错误类型（error_type）：format" in client.prompts[1].user
    assert (
        "legal_action_groups 是合法动作约束；card_ids "
        "必须完整等于其中一组" in client.prompts[1].user
    )
    assert "legal_action_groups" in client.prompts[1].user


@pytest.mark.asyncio
async def test_ai_player_repairs_invalid_card_id_once() -> None:
    valid_card = card("hearts", "A")
    invalid_card_id = "D2-diamonds-A"
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[valid_card],
    )
    game = make_game(snap)
    client = SequenceAIClient(
        [
            AIToolCall(
                name="play_cards",
                arguments={
                    "card_ids": [invalid_card_id],
                    "reason": "bad id",
                },
            ),
            AIToolCall(
                name="play_cards",
                arguments={
                    "card_ids": [valid_card["id"]],
                    "reason": "repaired id",
                },
            ),
        ]
    )
    player = ai.AIPlayer(index=0, config=_config(), client=client)

    await player.on_state(game, make_state_message(snap, seq=9))
    await asyncio.sleep(0.05)

    game.receive.assert_awaited()
    assert game.receive.call_args[0][0] == 0
    assert game.receive.call_args[0][1].seq == 9
    assert game.receive.call_args[0][1].raw == {
        "type": "play",
        "cards": [valid_card["id"]],
    }
    assert len(client.prompts) == 2
    assert (
        f"牌 {invalid_card_id} 不在你的当前手牌里"
        in client.prompts[1].user
    )
    assert "错误类型（error_type）：format" in client.prompts[1].user
    assert valid_card["id"] in client.prompts[1].user


@pytest.mark.asyncio
async def test_ai_player_stops_after_invalid_card_id_repair_fails() -> (
    None
):
    valid_card = card("hearts", "A")
    invalid_card_id = "D2-diamonds-A"
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[valid_card],
    )
    game = make_game(snap)
    client = SequenceAIClient(
        [
            AIToolCall(
                name="play_cards",
                arguments={
                    "card_ids": [invalid_card_id],
                    "reason": "bad id",
                },
            ),
            AIToolCall(
                name="play_cards",
                arguments={
                    "card_ids": [invalid_card_id],
                    "reason": "still bad",
                },
            ),
        ]
    )
    player = ai.AIPlayer(index=0, config=_config(), client=client)

    await player.on_state(game, make_state_message(snap, seq=10))
    await asyncio.sleep(0.05)

    game.receive.assert_not_awaited()
    assert len(client.prompts) == 2


def test_openai_client_uses_chat_completions_tool_shape() -> None:
    parameters: JSONObject = {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
        "required": ["reason"],
        "additionalProperties": False,
    }
    payload = build_chat_completions_payload(
        _config(),
        AIDecisionPrompt(system="system text", user="user text"),
        [
            AIToolSpec(
                name="test_tool",
                description="confirm",
                parameters=parameters,
            )
        ],
    )

    assert payload["model"] == "test-model"
    assert payload["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]
    assert payload["tool_choice"] == "required"
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "confirm",
                "parameters": parameters,
                "strict": True,
            },
        }
    ]


def test_ai_config_defaults_to_larger_output_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRACTOR_AI_MAX_OUTPUT_TOKENS", raising=False)

    config = ai_config.AIConfig.from_env()

    assert config.max_output_tokens == 2400


def test_ai_tool_schema_hides_local_next_round_tool() -> None:
    snap = make_snapshot(phase="WAITING", awaiting_action="next_round")

    tools = allowed_tool_specs(snap)

    assert tools == []


def test_ai_tool_schema_hides_local_bid_tools() -> None:
    test_card = card("spades", "2")
    snap = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        player_hand=[test_card],
        action_hints=[[test_card]],
    )

    tools = allowed_tool_specs(snap)

    assert tools == []


def test_ai_tool_schema_stir_uses_one_optional_card_tool() -> None:
    card1 = card("hearts", "2", deck=1)
    card2 = card("hearts", "2", deck=2)
    snap = make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        player_hand=[card1, card2],
        action_hints=[[card1, card2]],
    )

    tools = allowed_tool_specs(snap)

    assert len(tools) == 1
    assert tools[0].name == "stir_trump"
    assert _card_id_enum(tools[0]) == [card1["id"], card2["id"]]
    assert _required_fields(tools[0]) == []


def test_ai_stir_tool_without_card_ids_passes() -> None:
    card1 = card("hearts", "2", deck=1)
    card2 = card("hearts", "2", deck=2)
    snap = make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        player_hand=[card1, card2],
        action_hints=[[card1, card2]],
    )

    result = tool_call_to_message(
        7,
        snap,
        AIToolCall(name="stir_trump", arguments={}),
    )

    assert isinstance(result, Ok)
    assert result.value.raw == {"type": "stir", "pass": True}


def test_ai_tool_schema_limits_play_card_ids_to_current_hand() -> None:
    card1 = card("hearts", "A")
    card2 = card("diamonds", "K")
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[card1, card2],
    )

    tools = allowed_tool_specs(snap)

    assert len(tools) == 1
    assert tools[0].name == "play_cards"
    assert _card_id_enum(tools[0]) == [card1["id"], card2["id"]]


def test_ai_play_schema_uses_action_hint_ids() -> None:
    card1 = card("hearts", "A")
    card2 = card("diamonds", "2")
    card3 = card("spades", "K")
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[card1, card2, card3],
        action_hints=[[card2, card1]],
    )

    tools = allowed_tool_specs(snap)

    assert len(tools) == 1
    assert tools[0].name == "play_cards"
    assert _card_id_enum(tools[0]) == [card2["id"], card1["id"]]


def test_ai_play_tool_rejects_partial_action_hint() -> None:
    card1 = card("diamonds", "2")
    card2 = card("hearts", "10")
    snap = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[card1, card2],
        action_hints=[[card1, card2]],
    )

    result = tool_call_to_message(
        7,
        snap,
        AIToolCall(
            name="play_cards",
            arguments={
                "card_ids": [card2["id"]],
                "reason": "partial hint",
            },
        ),
    )

    assert isinstance(result, Rejected)
    assert isinstance(result, AIToolRejected)
    assert result.feedback.error_type == "format"
    assert (
        result.reason == "legal_action_groups 是合法动作约束；"
        "card_ids 必须完整等于其中一组："
        "不能只选其中一部分，也不能混合多组。"
    )
    assert (
        result.feedback.repair
        == "从 legal_action_groups 中复制一整组 card_ids。"
    )


def test_openai_client_always_disables_thinking() -> None:
    config = AIConfig(
        provider="openai",
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        timeout_seconds=1.0,
        max_retries=2,
        retry_delay_seconds=0.0,
        decision_retries=1,
        max_output_tokens=200,
        log_payloads=False,
        log_tool_use=True,
    )
    payload = build_chat_completions_payload(
        config,
        AIDecisionPrompt(system="system text", user="user text"),
        [
            AIToolSpec(
                name="test_tool",
                description="confirm",
                parameters={
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            )
        ],
    )

    assert payload["thinking"] == {"type": "disabled"}


def test_openai_client_extracts_chat_completion_tool_call() -> None:
    arguments: JSONObject = {"card_ids": ["card-1"], "reason": "test"}

    result = extract_chat_completion_tool_call(
        _chat_completion_response(
            tool_name="play_cards", arguments=arguments
        )
    )

    assert isinstance(result, Ok)
    assert result.value == AIToolCall(
        name="play_cards", arguments=arguments
    )


def test_openai_client_reports_length_finish_before_tool_call() -> None:
    response: JSONObject = {
        "id": "chatcmpl-length",
        "model": "mimo-v2.5-pro",
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": "",
                    "tool_calls": None,
                    "reasoning_content": "long reasoning",
                },
            },
        ],
    }

    result = extract_chat_completion_tool_call(response)

    assert isinstance(result, Rejected)
    assert result.reason == (
        "OpenAI-compatible response hit max_tokens before tool call "
        "(finish_reason=length); increase TRACTOR_AI_MAX_OUTPUT_TOKENS "
        "or disable thinking"
    )


def test_openai_message_log_includes_content_and_tool_calls() -> None:
    arguments: JSONObject = {"card_ids": ["card-1"], "reason": "test"}
    response = _chat_completion_response(
        tool_name="play_cards", arguments=arguments
    )

    message_log = chat_completion_message_log(
        response, include_tool_calls=True
    )

    assert "play_cards" in message_log
    assert "card-1" in message_log
    assert '"content": ""' in message_log


def test_openai_client_message_log_can_hide_tool_calls() -> None:
    arguments: JSONObject = {"card_ids": ["card-1"], "reason": "test"}
    response = _chat_completion_response(
        tool_name="play_cards", arguments=arguments
    )

    message_log = chat_completion_message_log(
        response, include_tool_calls=False
    )

    assert "card-1" not in message_log
    assert '"tool_calls": "<hidden>"' in message_log


@pytest.mark.asyncio
async def test_openai_client_decide_uses_async_http_transport() -> None:
    captured_payloads: list[JSONObject] = []
    arguments: JSONObject = {"reason": "ready"}

    def handler(request: httpx.Request) -> httpx.Response:
        assert (
            str(request.url)
            == "https://example.test/v1/chat/completions"
        )
        assert request.headers["authorization"] == "Bearer test-key"
        parsed: object = json.loads(request.content.decode("utf-8"))
        assert is_json_object(parsed)
        captured_payloads.append(parsed)
        return httpx.Response(
            status_code=200,
            content=json.dumps(
                _chat_completion_response(
                    tool_name="test_tool",
                    arguments=arguments,
                )
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    client = OpenAIChatCompletionsClient(
        _config(),
        transport=httpx.MockTransport(handler),
    )

    result = await client.decide(
        AIDecisionPrompt(system="system", user="user"),
        [
            AIToolSpec(
                name="test_tool",
                description="confirm",
                parameters={
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            )
        ],
    )

    assert isinstance(result, Ok)
    assert result.value.tool_call == AIToolCall(
        name="test_tool", arguments=arguments
    )
    assert len(captured_payloads) == 1
    assert captured_payloads[0]["model"] == "test-model"
    assert result.value.api.request is not None
    assert result.value.api.response is not None
    assert result.value.api.error is None
    assert "test_tool" in result.value.api.response
    assert _duration_ms(_json_object(result.value.api.response)) >= 0


@pytest.mark.asyncio
async def test_openai_client_retries_timeout_then_succeeds() -> None:
    calls = 0
    arguments: JSONObject = {"reason": "ready after retry"}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("timeout", request=request)
        return httpx.Response(
            status_code=200,
            content=json.dumps(
                _chat_completion_response(
                    tool_name="test_tool",
                    arguments=arguments,
                )
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    client = OpenAIChatCompletionsClient(
        _config(),
        transport=httpx.MockTransport(handler),
    )

    result = await client.decide(
        AIDecisionPrompt(system="system", user="user"),
        [
            AIToolSpec(
                name="test_tool",
                description="confirm",
                parameters={
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            )
        ],
    )

    assert isinstance(result, Ok)
    assert calls == 2
    assert result.value.tool_call == AIToolCall(
        name="test_tool", arguments=arguments
    )
    assert result.value.api.request is not None
    assert result.value.api.response is not None
    assert result.value.api.error is not None
    assert "API TIMEOUT" in result.value.api.error
    assert _duration_ms(_json_object(result.value.api.response)) >= 0
    assert _duration_ms(_json_object(result.value.api.error)) >= 0


@pytest.mark.asyncio
async def test_openai_does_not_retry_non_retryable_http_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            status_code=400,
            content=b'{"error":"bad request"}',
            headers={"Content-Type": "application/json"},
        )

    client = OpenAIChatCompletionsClient(
        _config(),
        transport=httpx.MockTransport(handler),
    )

    result = await client.decide(
        AIDecisionPrompt(system="system", user="user"),
        [
            AIToolSpec(
                name="test_tool",
                description="confirm",
                parameters={
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            )
        ],
    )

    assert isinstance(result, AIClientRejected)
    assert calls == 1
    assert result.reason == "OpenAI-compatible HTTP error 400"
    assert result.api.request is not None
    assert result.api.response is None
    assert result.api.error is not None
    assert "API HTTP ERROR" in result.api.error
    assert "bad request" in result.api.error
    assert _duration_ms(_json_object(result.api.error)) >= 0


class StaticAIClient(AIClient):
    def __init__(self, result: AIToolCall | Rejected) -> None:
        self.result = result
        self.prompts: list[AIDecisionPrompt] = []
        self.tools: list[list[AIToolSpec]] = []

    async def decide(
        self,
        prompt: AIDecisionPrompt,
        tools: list[AIToolSpec],
    ) -> Ok[AIDecision] | Rejected:
        self.prompts.append(prompt)
        self.tools.append(tools)
        if isinstance(self.result, Rejected):
            return self.result
        return Ok(
            AIDecision(
                assistant_content="static test decision",
                tool_call=self.result,
            )
        )


class SequenceAIClient(AIClient):
    def __init__(self, results: list[AIToolCall | Rejected]) -> None:
        self.results = results
        self.prompts: list[AIDecisionPrompt] = []
        self.tools: list[list[AIToolSpec]] = []
        self.index = 0

    async def decide(
        self,
        prompt: AIDecisionPrompt,
        tools: list[AIToolSpec],
    ) -> Ok[AIDecision] | Rejected:
        self.prompts.append(prompt)
        self.tools.append(tools)
        result = self.results[self.index]
        if self.index + 1 < len(self.results):
            self.index += 1
        if isinstance(result, Rejected):
            return result
        return Ok(
            AIDecision(
                assistant_content="sequence test decision",
                tool_call=result,
            )
        )


class ListLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def text(self) -> str:
        return "\n".join(self.messages)


def _card_id_enum(tool: AIToolSpec) -> list[str]:
    properties = tool.parameters["properties"]
    assert isinstance(properties, dict)
    card_ids = properties["card_ids"]
    assert isinstance(card_ids, dict)
    items = card_ids["items"]
    assert isinstance(items, dict)
    enum_values = items["enum"]
    assert isinstance(enum_values, list)
    result: list[str] = []
    for item in enum_values:
        assert isinstance(item, str)
        result.append(item)
    return result


def _required_fields(tool: AIToolSpec) -> list[str]:
    required = tool.parameters["required"]
    assert isinstance(required, list)
    result: list[str] = []
    for item in required:
        assert isinstance(item, str)
        result.append(item)
    return result


def _json_object(raw: str) -> JSONObject:
    parsed: object = json.loads(raw)
    assert is_json_object(parsed)
    return parsed


def _duration_ms(payload: JSONObject) -> int:
    duration = payload.get("duration_ms")
    if type(duration) is int:
        return duration
    errors = payload.get("errors")
    if isinstance(errors, list) and len(errors) > 0:
        first_error = errors[0]
        assert is_json_object(first_error)
        return _duration_ms(first_error)
    raise AssertionError("duration_ms missing")


def _chat_completion_response(
    *, tool_name: str, arguments: JSONObject
) -> JSONObject:
    return {
        "id": "chatcmpl-test",
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            }
        ],
    }


def _config(
    *, log_tool_use: bool = True, max_retries: int = 2
) -> AIConfig:
    return AIConfig(
        provider="openai",
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        timeout_seconds=1.0,
        max_retries=max_retries,
        retry_delay_seconds=0.0,
        decision_retries=1,
        max_output_tokens=200,
        log_payloads=False,
        log_tool_use=log_tool_use,
    )
