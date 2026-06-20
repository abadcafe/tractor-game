"""Environment configuration for AIPlayer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Self

type AIProvider = Literal["openai"]


@dataclass(frozen=True, slots=True)
class AIConfig:
    """Runtime configuration for AIPlayer."""

    provider: AIProvider
    base_url: str
    api_key: str | None
    model: str
    timeout_seconds: float
    max_retries: int
    retry_delay_seconds: float
    decision_retries: int
    max_output_tokens: int
    log_payloads: bool
    log_tool_use: bool

    @classmethod
    def from_env(cls) -> Self:
        return cls(
            provider="openai",
            base_url=os.environ.get("TRACTOR_AI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("TRACTOR_AI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("TRACTOR_AI_MODEL", "gpt-5-mini"),
            timeout_seconds=_env_float("TRACTOR_AI_TIMEOUT_SECONDS", default=8.0),
            max_retries=_env_int("TRACTOR_AI_MAX_RETRIES", default=2),
            retry_delay_seconds=_env_float("TRACTOR_AI_RETRY_DELAY_SECONDS", default=3.0),
            decision_retries=_env_int("TRACTOR_AI_DECISION_RETRIES", default=1),
            max_output_tokens=_env_int("TRACTOR_AI_MAX_OUTPUT_TOKENS", default=2400),
            log_payloads=_env_bool("TRACTOR_AI_LOG_PAYLOADS", default=False),
            log_tool_use=_env_bool("TRACTOR_AI_LOG_TOOL_USE", default=True),
        )


def _env_int(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, *, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)
