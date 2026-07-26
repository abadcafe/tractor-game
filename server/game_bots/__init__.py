"""Bot players driven by injected strategic decision policies."""

from ._auto import AutoPolicy
from ._factory import DefaultBotPlayerFactory
from ._player import BotPlayer
from ._policy import (
    DecisionCommand,
    DecisionPolicy,
    DecisionRequest,
    StrategicAction,
)

__all__ = (
    "AutoPolicy",
    "BotPlayer",
    "DefaultBotPlayerFactory",
    "DecisionCommand",
    "DecisionPolicy",
    "DecisionRequest",
    "StrategicAction",
)
