"""Browser wire protocol for game sessions."""

from .client import ClientFrame, read_frame
from .projection import StateMessage, encode_state

__all__ = [
    "ClientFrame",
    "StateMessage",
    "encode_state",
    "read_frame",
]
