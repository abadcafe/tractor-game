"""LLM client abstractions for AIPlayer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol, TypeGuard

from server.foundation.result import Ok, Rejected

type JSONValue = (
    str
    | int
    | float
    | bool
    | None
    | list[JSONValue]
    | dict[str, JSONValue]
)
type JSONObject = dict[str, JSONValue]


@dataclass(frozen=True, slots=True)
class AIDecisionPrompt:
    """Prompt payload for one AI decision."""

    system: str
    user: str


@dataclass(frozen=True, slots=True)
class AIToolSpec:
    """Function tool exposed to the LLM for one decision."""

    name: str
    description: str
    parameters: JSONObject
    strict: bool = True


@dataclass(frozen=True, slots=True)
class AIToolCall:
    """One function tool call returned by the LLM."""

    name: str
    arguments: JSONObject


@dataclass(frozen=True, slots=True)
class AIAPIInteraction:
    """Raw provider/API details for one LLM decision attempt."""

    request: str | None = None
    response: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AIClientRejected(Rejected):
    """LLM client rejection with raw API interaction details."""

    api: AIAPIInteraction = field(default_factory=AIAPIInteraction)


class AIClientDisabledRejected(Rejected):
    reason_text: ClassVar[str] = "AI LLM client is disabled"

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class OpenAIAPIKeyMissingRejected(Rejected):
    reason_text: ClassVar[str] = "TRACTOR_AI_API_KEY is not set"

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class OpenAIRequestNotAttemptedRejected(Rejected):
    reason_text: ClassVar[str] = (
        "OpenAI-compatible request was not attempted"
    )

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class OpenAIRequestTimedOutRejected(Rejected):
    reason_text: ClassVar[str] = "OpenAI-compatible request timed out"

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class OpenAINetworkRejected(Rejected):
    reason_text: ClassVar[str] = "OpenAI-compatible network error"

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class OpenAIHTTPStatusRejected(Rejected):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"OpenAI-compatible HTTP error {status_code}")


class OpenAIInvalidJSONRejected(Rejected):
    reason_text: ClassVar[str] = (
        "OpenAI-compatible returned invalid JSON"
    )

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class OpenAIResponseNotJSONObjectRejected(Rejected):
    reason_text: ClassVar[str] = (
        "OpenAI-compatible response is not a JSON object"
    )

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class OpenAIChoicesMissingRejected(Rejected):
    reason_text: ClassVar[str] = (
        "OpenAI-compatible response has no choices list"
    )

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class OpenAIChoiceCountRejected(Rejected):
    def __init__(self, choice_count: int) -> None:
        super().__init__(
            f"OpenAI-compatible returned {choice_count} choices;"
            f"expected exactly one"
        )


class OpenAIChoiceNotObjectRejected(Rejected):
    reason_text: ClassVar[str] = (
        "OpenAI-compatible choice is not an object"
    )

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class OpenAIMessageMissingRejected(Rejected):
    reason_text: ClassVar[str] = (
        "OpenAI-compatible choice has no message object"
    )

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class OpenAIResponseLengthRejected(Rejected):
    reason_text: ClassVar[str] = (
        "OpenAI-compatible response hit max_tokens before tool call "
        "(finish_reason=length); increase TRACTOR_AI_MAX_OUTPUT_TOKENS "
        "or disable thinking"
    )

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class OpenAIToolCallsMissingRejected(Rejected):
    reason_text: ClassVar[str] = (
        "OpenAI-compatible message has no tool_calls list"
    )

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class OpenAIToolCallCountRejected(Rejected):
    def __init__(self, tool_call_count: int) -> None:
        super().__init__(
            f"OpenAI-compatible returned {tool_call_count} tool calls;"
            f"expected exactly one"
        )


class OpenAIToolCallNotObjectRejected(Rejected):
    reason_text: ClassVar[str] = (
        "OpenAI-compatible tool_call is not an object"
    )

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class OpenAIToolCallNotFunctionRejected(Rejected):
    reason_text: ClassVar[str] = (
        "OpenAI-compatible tool_call is not a function call"
    )

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class OpenAIToolCallMissingFunctionRejected(Rejected):
    reason_text: ClassVar[str] = (
        "OpenAI-compatible tool_call missing function object"
    )

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class OpenAIFunctionCallMissingNameOrArgumentsRejected(Rejected):
    reason_text: ClassVar[str] = (
        "OpenAI-compatible function call missing name or arguments"
    )

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class OpenAIFunctionCallArgumentsInvalidJSONRejected(Rejected):
    reason_text: ClassVar[str] = (
        "OpenAI-compatible function call arguments are not valid JSON"
    )

    def __init__(self) -> None:
        super().__init__(self.reason_text)


class OpenAIFunctionCallArgumentsNotJSONObjectRejected(Rejected):
    reason_text: ClassVar[str] = (
        "OpenAI-compatible function call arguments are not a JSONobject"
    )

    def __init__(self) -> None:
        super().__init__(self.reason_text)


@dataclass(frozen=True, slots=True)
class AIDecision:
    """
    One assistant response plus the tool call selected for the game.
    """

    assistant_content: str | None
    tool_call: AIToolCall
    api: AIAPIInteraction = field(default_factory=AIAPIInteraction)


class AIClient(Protocol):
    """Provider-neutral interface used by AIPlayer."""

    async def decide(
        self,
        prompt: AIDecisionPrompt,
        tools: list[AIToolSpec],
    ) -> Ok[AIDecision] | Rejected: ...


class DisabledAIClient:
    """Client used when LLM calls are disabled or not configured."""

    async def decide(
        self,
        prompt: AIDecisionPrompt,
        tools: list[AIToolSpec],
    ) -> Ok[AIDecision] | Rejected:
        return AIClientDisabledRejected()


def is_json_value(value: object) -> TypeGuard[JSONValue]:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if _is_object_list(value):
        return all(is_json_value(item) for item in value)
    if _is_object_dict(value):
        return all(
            isinstance(key, str) and is_json_value(item)
            for key, item in value.items()
        )
    return False


def is_json_object(value: object) -> TypeGuard[JSONObject]:
    if not _is_object_dict(value):
        return False
    return all(
        isinstance(key, str) and is_json_value(item)
        for key, item in value.items()
    )


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)
