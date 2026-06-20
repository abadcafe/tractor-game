"""LLM client abstractions for AIPlayer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypeGuard

from server.sm.result import Ok, Rejected

type JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
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


@dataclass(frozen=True, slots=True)
class AIDecision:
    """One assistant response plus the tool call selected for the game."""

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
        return Rejected("AI LLM client is disabled")


def is_json_value(value: object) -> TypeGuard[JSONValue]:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if _is_object_list(value):
        return all(is_json_value(item) for item in value)
    if _is_object_dict(value):
        return all(isinstance(key, str) and is_json_value(item) for key, item in value.items())
    return False


def is_json_object(value: object) -> TypeGuard[JSONObject]:
    if not _is_object_dict(value):
        return False
    return all(isinstance(key, str) and is_json_value(item) for key, item in value.items())


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)
