"""Environment configuration for the LLM decision policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Self

type LLMProvider = Literal["openai"]


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Runtime configuration for one LLM decision policy."""

    provider: LLMProvider
    base_url: str
    api_key: str | None
    model: str
    http_timeout_seconds: float
    http_max_retries: int
    http_retry_delay_seconds: float
    decision_max_retries: int
    max_output_tokens: int

    @classmethod
    def from_env(cls) -> Self:
        return cls(
            provider="openai",
            base_url=os.environ.get(
                "TRACTOR_LLM_BASE_URL", "https://api.openai.com/v1"
            ),
            api_key=os.environ.get("TRACTOR_LLM_API_KEY"),
            model=os.environ.get("TRACTOR_LLM_MODEL", "gpt-5-mini"),
            http_timeout_seconds=_env_float(
                "TRACTOR_LLM_HTTP_TIMEOUT_SECONDS", default=8.0
            ),
            http_max_retries=_env_int(
                "TRACTOR_LLM_HTTP_MAX_RETRIES", default=2
            ),
            http_retry_delay_seconds=_env_float(
                "TRACTOR_LLM_HTTP_RETRY_DELAY_SECONDS", default=3.0
            ),
            decision_max_retries=_env_int(
                "TRACTOR_LLM_DECISION_MAX_RETRIES", default=2
            ),
            max_output_tokens=_env_int(
                "TRACTOR_LLM_MAX_OUTPUT_TOKENS", default=2400
            ),
        )


def _env_int(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, *, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)
