"""Human WebSocket player implementation."""

from __future__ import annotations

import logging
from typing import TypeGuard

from fastapi import WebSocket, WebSocketDisconnect

from server.game.players.base import GameView, Player
from server.game.protocol import PlayerMessage, StateMessage

logger = logging.getLogger(__name__)


class HumanPlayer(Player):
    """Human player that manages its own WebSocket lifecycle.

    Self-contained: receives WS messages and forwards PlayerMessage
    envelopes
    to Game.receive(). Server's WS endpoint delegates to
    handle_connection().
    """

    def __init__(self, index: int) -> None:
        super().__init__(index)
        self._ws: WebSocket | None = None

    async def on_state(
        self, game: GameView, message: StateMessage
    ) -> None:
        """Push state to the human player via WebSocket."""
        if self._ws is None:
            return
        try:
            await self._ws.send_json(message.model_dump(mode="json"))
        except WebSocketDisconnect, OSError:
            logger.debug(
                "Failed to push state to human player %d (WS likely"
                "disconnected)",
                self.index,
                exc_info=True,
            )

    async def handle_connection(
        self, websocket: WebSocket, game: GameView
    ) -> None:
        """Take over the full WS connection lifecycle."""
        if self._ws is not None:
            old_ws = self._ws
            self._ws = None
            try:
                await old_ws.close()
            except WebSocketDisconnect, OSError:
                logger.debug(
                    "Failed to close old WS for player %d during"
                    "takeover",
                    self.index,
                    exc_info=True,
                )

        await websocket.accept()
        self._ws = websocket

        try:
            while True:
                try:
                    raw = await websocket.receive_json()
                except WebSocketDisconnect, OSError:
                    logger.debug(
                        "WS receive loop ended (user disconnected)"
                    )
                    break

                if not _is_str_dict(raw):
                    continue
                await game.receive(
                    self.index,
                    PlayerMessage(seq=_message_seq(raw), raw=raw),
                )

        finally:
            self._clear_ws_if_current(websocket)

    def _clear_ws_if_current(self, ws: WebSocket) -> None:
        """
        Clear the WebSocket reference only if it still points to the
        given instance.
        """
        if self._ws is ws:
            self._ws = None

    def is_connected(self) -> bool:
        """
        Return True if this player has an active WebSocket connection.
        """
        return self._ws is not None

    async def close_ws(
        self, *, code: int = 1000, reason: str = ""
    ) -> None:
        """
        Close the WebSocket connection if active, then clear the
        reference.
        """
        if self._ws is not None:
            ws = self._ws
            self._ws = None
            try:
                await ws.close(code=code, reason=reason)
            except WebSocketDisconnect, OSError:
                logger.debug(
                    "Failed to close WS for player %d (already"
                    "disconnected)",
                    self.index,
                    exc_info=True,
                )


def _is_str_dict(val: object) -> TypeGuard[dict[str, object]]:
    """Narrow object to dict[str, object] with string keys."""
    return isinstance(val, dict)


def _message_seq(raw: dict[str, object]) -> int:
    seq_raw = raw.get("seq", 0)
    if isinstance(seq_raw, bool):
        return 0
    if isinstance(seq_raw, int):
        return seq_raw
    return 0
