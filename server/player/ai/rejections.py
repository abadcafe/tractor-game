"""Structured rejection feedback for AI tool decisions.

These strings are part of the prompt loop. Keep them short, Chinese, and
actionable so the LLM can repair a rejected tool call on the next
attempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from server.result import Rejected

type AIRejectionErrorType = Literal["format", "rule"]


@dataclass(frozen=True, slots=True)
class AIRejectionFeedback:
    error_type: AIRejectionErrorType
    reason: str
    repair: str


@dataclass(frozen=True, slots=True)
class AIToolRejected(Rejected):
    feedback: AIRejectionFeedback


def format_rejected(reason: str, repair: str) -> AIToolRejected:
    return AIToolRejected(
        reason=reason,
        feedback=AIRejectionFeedback(
            error_type="format",
            reason=reason,
            repair=repair,
        ),
    )


def rule_feedback(reason: str) -> AIRejectionFeedback:
    return AIRejectionFeedback(
        error_type="rule",
        reason=reason,
        repair=(
            "请根据当前 state 重新选择合法动作；如果 action_hints "
            "非空，完整复制其中一组。"
        ),
    )


def feedback_from_rejected(rejection: Rejected) -> AIRejectionFeedback:
    if isinstance(rejection, AIToolRejected):
        return rejection.feedback
    return rule_feedback(rejection.reason)
