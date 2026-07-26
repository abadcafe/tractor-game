"""Black-box tests for the OpenAI-compatible LLM adapter."""

from __future__ import annotations

import json

import httpx

from server.foundation.result import Ok, Rejected
from server.game_bots.llm import (
    DecisionPrompt,
    JSONObject,
    LLMConfig,
    ToolCall,
    ToolSpec,
)
from server.game_bots.llm.openai import (
    OpenAIChatCompletionsClient,
    build_chat_completions_payload,
    extract_chat_completion_tool_call,
)


def _config(
    *,
    api_key: str | None = "test-key",
    retries: int = 1,
) -> LLMConfig:
    return LLMConfig(
        provider="openai",
        base_url="https://example.test/v1",
        api_key=api_key,
        model="test-model",
        http_timeout_seconds=1.0,
        http_max_retries=retries,
        http_retry_delay_seconds=0.0,
        decision_max_retries=1,
        max_output_tokens=200,
    )


def _tool() -> ToolSpec:
    return ToolSpec(
        name="test_tool",
        description="confirm",
        parameters={
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    )


def _response(arguments: dict[str, str]) -> JSONObject:
    return {
        "id": "chatcmpl-test",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "test_tool",
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ],
    }


def test_build_payload_uses_strict_single_tool_completion() -> None:
    payload = build_chat_completions_payload(
        _config(),
        DecisionPrompt(system="system text", user="user text"),
        [_tool()],
    )

    assert payload["model"] == "test-model"
    assert payload["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]
    assert payload["tool_choice"] == "required"
    assert payload["parallel_tool_calls"] is False
    assert payload["temperature"] == 0
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["max_tokens"] == 200
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "confirm",
                "parameters": _tool().parameters,
                "strict": True,
            },
        }
    ]


def test_extract_returns_exact_function_tool_call() -> None:
    result = extract_chat_completion_tool_call(
        _response({"reason": "ready"})
    )

    assert isinstance(result, Ok)
    assert result.value == ToolCall(
        name="test_tool",
        arguments={"reason": "ready"},
    )


def test_extract_rejects_length_before_tool_call() -> None:
    response: JSONObject = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": "",
                    "tool_calls": None,
                },
            }
        ]
    }
    result = extract_chat_completion_tool_call(response)

    assert isinstance(result, Rejected)
    assert "finish_reason=length" in result.reason


def test_extract_rejects_multiple_or_malformed_tool_calls() -> None:
    call: JSONObject = {
        "type": "function",
        "function": {
            "name": "test_tool",
            "arguments": '{"reason": "ready"}',
        },
    }
    multiple: JSONObject = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [call, call],
                },
            }
        ]
    }
    malformed: JSONObject = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "test_tool",
                                "arguments": "{",
                            },
                        }
                    ],
                },
            }
        ]
    }

    assert isinstance(
        extract_chat_completion_tool_call(multiple), Rejected
    )
    assert isinstance(
        extract_chat_completion_tool_call(malformed), Rejected
    )


async def test_decide_uses_async_transport_and_authorization() -> None:
    payloads: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert (
            str(request.url)
            == "https://example.test/v1/chat/completions"
        )
        assert request.headers["authorization"] == "Bearer test-key"
        payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json=_response({"reason": "ready"}),
        )

    client = OpenAIChatCompletionsClient(
        _config(),
        transport=httpx.MockTransport(handler),
    )

    result = await client.decide(
        DecisionPrompt(system="system", user="user"),
        [_tool()],
    )

    assert isinstance(result, Ok)
    assert result.value.tool_call == ToolCall(
        name="test_tool",
        arguments={"reason": "ready"},
    )
    assert len(payloads) == 1
    assert result.value.api.request is not None
    assert result.value.api.response is not None
    assert result.value.api.error is None


async def test_decide_retries_timeout_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("timeout", request=request)
        return httpx.Response(
            200,
            json=_response({"reason": "ready"}),
        )

    client = OpenAIChatCompletionsClient(
        _config(),
        transport=httpx.MockTransport(handler),
    )

    result = await client.decide(
        DecisionPrompt(system="system", user="user"),
        [_tool()],
    )

    assert isinstance(result, Ok)
    assert calls == 2


async def test_decide_does_not_retry_nonretryable_http_status() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            400,
            json={"error": "bad request"},
        )

    client = OpenAIChatCompletionsClient(
        _config(),
        transport=httpx.MockTransport(handler),
    )

    result = await client.decide(
        DecisionPrompt(system="system", user="user"),
        [_tool()],
    )

    assert isinstance(result, Rejected)
    assert calls == 1
    assert result.reason == "OpenAI-compatible HTTP error 400"


async def test_decide_rejects_missing_api_key_without_transport() -> (
    None
):
    client = OpenAIChatCompletionsClient(
        _config(api_key=None),
    )

    result = await client.decide(
        DecisionPrompt(system="system", user="user"),
        [_tool()],
    )

    assert isinstance(result, Rejected)
    assert result.reason == "TRACTOR_LLM_API_KEY is not set"
