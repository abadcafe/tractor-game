"""Rule-complete semantic action spaces for policy decisions."""

from __future__ import annotations

from server.policy_model.actions.legality.contract import (
    LegalActionSpace,
)
from server.policy_model.actions.legality.factory import (
    build_legal_action_space,
)

__all__ = (
    "LegalActionSpace",
    "build_legal_action_space",
)
