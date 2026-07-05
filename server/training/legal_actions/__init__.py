"""Rule-complete semantic action spaces for training policies."""

from __future__ import annotations

from server.training.legal_actions.contract import (
    LegalActionIndex,
)
from server.training.legal_actions.factory import (
    build_legal_action_index,
)

__all__ = (
    "LegalActionIndex",
    "build_legal_action_index",
)
