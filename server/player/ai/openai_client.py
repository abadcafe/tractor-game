"""OpenAI-compatible Chat Completions client for AIPlayer."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from time import perf_counter
from typing import TypeGuard

import httpx

from server.player.ai.client import (
    AIAPIInteraction,
    AIClient,
    AIClientRejected,
    AIDecision,
    AIDecisionPrompt,
    AIToolCall,
    AIToolSpec,
    JSONObject,
    JSONValue,
    OpenAIAPIKeyMissingRejected,
    OpenAIChoiceCountRejected,
    OpenAIChoiceNotObjectRejected,
    OpenAIChoicesMissingRejected,
    OpenAIFunctionCallArgumentsInvalidJSONRejected,
    OpenAIFunctionCallArgumentsNotJSONObjectRejected,
    OpenAIFunctionCallMissingNameOrArgumentsRejected,
    OpenAIHTTPStatusRejected,
    OpenAIInvalidJSONRejected,
    OpenAIMessageMissingRejected,
    OpenAINetworkRejected,
    OpenAIRequestNotAttemptedRejected,
    OpenAIRequestTimedOutRejected,
    OpenAIResponseLengthRejected,
    OpenAIResponseNotJSONObjectRejected,
    OpenAIToolCallCountRejected,
    OpenAIToolCallMissingFunctionRejected,
    OpenAIToolCallNotFunctionRejected,
    OpenAIToolCallNotObjectRejected,
    OpenAIToolCallsMissingRejected,
    is_json_object,
    is_json_value,
)
from server.player.ai.config import AIConfig
from server.result import Ok, Rejected

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _OpenAIHTTPResponse:
    body: JSONObject
    api: AIAPIInteraction


@dataclass(frozen=True, slots=True)
class OpenAIChatCompletionsClient(AIClient):
    """
    Minimal async wrapper around OpenAI-compatible chat completions.
    """

    config: AIConfig
    transport: httpx.AsyncBaseTransport | None = None

    async def decide(
        self,
        prompt: AIDecisionPrompt,
        tools: list[AIToolSpec],
    ) -> Ok[AIDecision] | Rejected:
        if self.config.api_key is None:
            rejection = OpenAIAPIKeyMissingRejected()
            return AIClientRejected(
                rejection.reason,
                api=AIAPIInteraction(
                    error=_api_error_content(
                        title="API CONFIG ERROR",
                        reason=rejection.reason,
                    ),
                ),
            )
        payload = build_chat_completions_payload(
            self.config, prompt, tools
        )
        endpoint = _chat_completions_endpoint(self.config.base_url)
        result = await self._post(endpoint, payload)
        if isinstance(result, Rejected):
            return result
        response = result.value.body
        call_result = extract_chat_completion_tool_call(response)
        if isinstance(call_result, Ok):
            return Ok(
                AIDecision(
                    assistant_content=assistant_content(response),
                    tool_call=call_result.value,
                    api=result.value.api,
                )
            )
        return AIClientRejected(
            call_result.reason,
            api=_api_with_error(
                result.value.api,
                _api_error_content(
                    title="API TOOL CALL ERROR",
                    reason=call_result.reason,
                ),
            ),
        )

    async def _post(
        self,
        endpoint: str,
        payload: JSONObject,
    ) -> Ok[_OpenAIHTTPResponse] | AIClientRejected:
        assert self.config.api_key is not None
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        max_attempts = max(self.config.http_max_retries, 0) + 1
        api_request = _api_request_content(endpoint, payload)
        api_errors: list[str] = []
        last_rejection: Rejected = OpenAIRequestNotAttemptedRejected()

        async with httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
            transport=self.transport,
        ) as client:
            for attempt in range(1, max_attempts + 1):
                (
                    result,
                    retryable,
                    api_response,
                    api_error,
                ) = await self._post_once(
                    client=client,
                    endpoint=endpoint,
                    payload=payload,
                    headers=headers,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
                if api_error is not None:
                    api_errors.append(api_error)
                if isinstance(result, Ok):
                    return Ok(
                        _OpenAIHTTPResponse(
                            body=result.value,
                            api=AIAPIInteraction(
                                request=api_request,
                                response=api_response,
                                error=_combine_api_errors(
                                    api_errors, final_reason=None
                                ),
                            ),
                        )
                    )
                last_rejection = result
                if retryable and attempt < max_attempts:
                    delay = self.config.http_retry_delay_seconds
                    await asyncio.sleep(delay)
                else:
                    break

        return AIClientRejected(
            last_rejection.reason,
            api=AIAPIInteraction(
                request=api_request,
                error=_combine_api_errors(
                    api_errors, final_reason=last_rejection.reason
                ),
            ),
        )

    async def _post_once(
        self,
        *,
        client: httpx.AsyncClient,
        endpoint: str,
        payload: JSONObject,
        headers: dict[str, str],
        attempt: int,
        max_attempts: int,
    ) -> tuple[Ok[JSONObject] | Rejected, bool, str | None, str | None]:
        started_at = perf_counter()
        try:
            response = await client.post(
                endpoint,
                json=payload,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            duration_ms = _elapsed_ms(started_at)
            error_content = _api_request_error_content(
                title="API TIMEOUT",
                attempt=attempt,
                max_attempts=max_attempts,
                duration_ms=duration_ms,
                endpoint=endpoint,
                error=exc,
            )
            logger.warning(
                "AI OpenAI-compatible request timed out: attempt=%d/%d",
                attempt,
                max_attempts,
            )
            return (
                OpenAIRequestTimedOutRejected(),
                True,
                None,
                error_content,
            )
        except httpx.RequestError as exc:
            duration_ms = _elapsed_ms(started_at)
            error_content = _api_request_error_content(
                title="API NETWORK ERROR",
                attempt=attempt,
                max_attempts=max_attempts,
                duration_ms=duration_ms,
                endpoint=endpoint,
                error=exc,
            )
            logger.warning(
                "AI OpenAI-compatible request error: attempt=%d/%d"
                "error=%s",
                attempt,
                max_attempts,
                exc,
            )
            return (OpenAINetworkRejected(), True, None, error_content)

        duration_ms = _elapsed_ms(started_at)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            error_content = _api_error_content(
                title="API HTTP ERROR",
                attempt=attempt,
                max_attempts=max_attempts,
                duration_ms=duration_ms,
                status_code=status_code,
                body=exc.response.text,
            )
            logger.warning(
                "AI OpenAI-compatible HTTP error: attempt=%d/%d"
                "status=%s body=%s",
                attempt,
                max_attempts,
                status_code,
                _truncate(exc.response.text),
            )
            rejection = OpenAIHTTPStatusRejected(status_code)
            return (
                rejection,
                _retryable_status(status_code),
                None,
                error_content,
            )

        response_body = response.text
        logger.debug(
            "AI OpenAI-compatible HTTP success: attempt=%d/%dstatus=%s",
            attempt,
            max_attempts,
            response.status_code,
        )
        try:
            parsed: object = json.loads(response_body)
        except JSONDecodeError:
            response_content = _api_response_content(
                title="API RESPONSE",
                attempt=attempt,
                max_attempts=max_attempts,
                duration_ms=duration_ms,
                status_code=response.status_code,
                body=response_body,
            )
            logger.warning(
                "AI OpenAI-compatible returned invalid JSON: %s",
                _truncate(response_body),
            )
            rejection = OpenAIInvalidJSONRejected()
            return (
                rejection,
                False,
                response_content,
                _api_error_content(
                    title="API JSON ERROR", reason=rejection.reason
                ),
            )

        if not is_json_object(parsed):
            response_content = _api_response_content(
                title="API RESPONSE",
                attempt=attempt,
                max_attempts=max_attempts,
                duration_ms=duration_ms,
                status_code=response.status_code,
                body=parsed if is_json_value(parsed) else response_body,
            )
            rejection = OpenAIResponseNotJSONObjectRejected()
            return (
                rejection,
                False,
                response_content,
                _api_error_content(
                    title="API SHAPE ERROR",
                    reason=rejection.reason,
                ),
            )
        response_content = _api_response_content(
            title="API RESPONSE",
            attempt=attempt,
            max_attempts=max_attempts,
            duration_ms=duration_ms,
            status_code=response.status_code,
            body=parsed,
        )
        return (Ok(parsed), False, response_content, None)


def build_chat_completions_payload(
    config: AIConfig,
    prompt: AIDecisionPrompt,
    tools: list[AIToolSpec],
) -> JSONObject:
    payload: JSONObject = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
        "tools": [_tool_to_json(tool) for tool in tools],
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "max_tokens": config.max_output_tokens,
        "temperature": 0,
        "thinking": {"type": "disabled"},
    }
    return payload


def _tool_to_json(tool: AIToolSpec) -> JSONObject:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": tool.strict,
        },
    }


def extract_chat_completion_tool_call(
    response: JSONObject,
) -> Ok[AIToolCall] | Rejected:
    message_result = _extract_message(response)
    if isinstance(message_result, Rejected):
        return message_result
    message = message_result.value
    tool_calls = message.get("tool_calls")
    if not _is_object_list(tool_calls):
        if _extract_finish_reason(response) == "length":
            return OpenAIResponseLengthRejected()
        return OpenAIToolCallsMissingRejected()
    calls: list[AIToolCall] = []
    for item in tool_calls:
        call_result = _extract_one_tool_call(item)
        if isinstance(call_result, Rejected):
            return call_result
        calls.append(call_result.value)
    if len(calls) != 1:
        return OpenAIToolCallCountRejected(len(calls))
    return Ok(calls[0])


def assistant_content(response: JSONObject) -> str | None:
    message_result = _extract_message(response)
    if isinstance(message_result, Rejected):
        return None
    content = message_result.value.get("content")
    if isinstance(content, str):
        return content
    return None


def _extract_message(
    response: JSONObject,
) -> Ok[dict[str, object]] | Rejected:
    choices = response.get("choices")
    if not _is_object_list(choices):
        return OpenAIChoicesMissingRejected()
    if len(choices) != 1:
        return OpenAIChoiceCountRejected(len(choices))
    choice = choices[0]
    if not _is_object_dict(choice):
        return OpenAIChoiceNotObjectRejected()
    message = choice.get("message")
    if not _is_object_dict(message):
        return OpenAIMessageMissingRejected()
    return Ok(message)


def _extract_finish_reason(response: JSONObject) -> str | None:
    choices = response.get("choices")
    if not _is_object_list(choices) or len(choices) != 1:
        return None
    choice = choices[0]
    if not _is_object_dict(choice):
        return None
    finish_reason = choice.get("finish_reason")
    if isinstance(finish_reason, str):
        return finish_reason
    return None


def _extract_one_tool_call(item: object) -> Ok[AIToolCall] | Rejected:
    if not _is_object_dict(item):
        return OpenAIToolCallNotObjectRejected()
    if item.get("type") != "function":
        return OpenAIToolCallNotFunctionRejected()
    function = item.get("function")
    if not _is_object_dict(function):
        return OpenAIToolCallMissingFunctionRejected()
    name = function.get("name")
    arguments_raw = function.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments_raw, str):
        return OpenAIFunctionCallMissingNameOrArgumentsRejected()
    try:
        arguments_obj: object = json.loads(arguments_raw)
    except JSONDecodeError:
        return OpenAIFunctionCallArgumentsInvalidJSONRejected()
    if not is_json_object(arguments_obj):
        return OpenAIFunctionCallArgumentsNotJSONObjectRejected()
    return Ok(AIToolCall(name=name, arguments=arguments_obj))


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code <= 599


def _chat_completions_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _elapsed_ms(started_at: float) -> int:
    return max(round((perf_counter() - started_at) * 1000), 0)


def _api_request_content(endpoint: str, payload: JSONObject) -> str:
    return json.dumps(
        {
            "method": "POST",
            "endpoint": endpoint,
            "json": payload,
        },
        ensure_ascii=False,
        indent=2,
    )


def _api_response_content(
    *,
    title: str,
    attempt: int,
    max_attempts: int,
    duration_ms: int,
    status_code: int,
    body: JSONValue,
) -> str:
    return json.dumps(
        {
            "title": title,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "duration_ms": duration_ms,
            "status_code": status_code,
            "body": body,
        },
        ensure_ascii=False,
        indent=2,
    )


def _api_request_error_content(
    *,
    title: str,
    attempt: int,
    max_attempts: int,
    duration_ms: int,
    endpoint: str,
    error: httpx.HTTPError,
) -> str:
    return json.dumps(
        {
            "title": title,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "duration_ms": duration_ms,
            "endpoint": endpoint,
            "error_type": type(error).__name__,
            "error": str(error),
        },
        ensure_ascii=False,
        indent=2,
    )


def _api_error_content(
    *,
    title: str,
    reason: str | None = None,
    attempt: int | None = None,
    max_attempts: int | None = None,
    duration_ms: int | None = None,
    status_code: int | None = None,
    body: JSONValue | None = None,
) -> str:
    payload: JSONObject = {"title": title}
    if reason is not None:
        payload["reason"] = reason
    if attempt is not None:
        payload["attempt"] = attempt
    if max_attempts is not None:
        payload["max_attempts"] = max_attempts
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    if status_code is not None:
        payload["status_code"] = status_code
    if body is not None:
        payload["body"] = body
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def _api_with_error(
    api: AIAPIInteraction, error: str
) -> AIAPIInteraction:
    return AIAPIInteraction(
        request=api.request,
        response=api.response,
        error=_combine_api_errors(
            [api.error, error], final_reason=None
        ),
    )


def _combine_api_errors(
    errors: Sequence[str | None], *, final_reason: str | None
) -> str | None:
    present = [error for error in errors if error is not None]
    if not present:
        if final_reason is None:
            return None
        return _api_error_content(
            title="API ERROR", reason=final_reason
        )
    if len(present) == 1 and final_reason is None:
        return present[0]
    parsed_errors = [_raw_json_value(error) for error in present]
    payload: JSONObject = {"errors": parsed_errors}
    if final_reason is not None:
        payload["reason"] = final_reason
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _raw_json_value(raw: str) -> JSONValue:
    try:
        parsed: object = json.loads(raw)
    except JSONDecodeError:
        return raw
    if is_json_value(parsed):
        return parsed
    return raw


def _truncate(value: str, limit: int = 1000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"
