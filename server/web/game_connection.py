"""Human WebSocket connection at the web/runtime boundary."""

from __future__ import annotations

import logging

from fastapi import WebSocket, WebSocketDisconnect

from server.foundation.result import Rejected
from server.game_runtime.room import GameRoom
from server.game_runtime.session import (
    Delivery,
    DeliverySink,
    DisconnectReason,
)

from .game_wire import encode_state, read_frame

logger = logging.getLogger(__name__)


class WebSocketSink(DeliverySink):
    """Encode runtime deliveries onto one WebSocket."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket

    async def offer(self, delivery: Delivery) -> None:
        message = encode_state(
            viewer=delivery.viewer,
            seq=delivery.seq,
            snapshot=delivery.snapshot,
            error=delivery.error,
        )
        try:
            await self._websocket.send_json(
                message.model_dump(mode="json")
            )
        except WebSocketDisconnect, OSError:
            logger.debug("game websocket send failed")

    async def disconnect(self, reason: DisconnectReason) -> None:
        try:
            await self._websocket.close(
                code=_disconnect_code(reason),
                reason=reason.value,
            )
        except WebSocketDisconnect, OSError:
            logger.debug("game websocket already disconnected")


async def handle_game_connection(
    websocket: WebSocket,
    room: GameRoom,
    *,
    player: int,
    user_id: str,
) -> None:
    """Own one human WebSocket until disconnect or takeover."""
    sink = WebSocketSink(websocket)
    await websocket.accept()
    connected = await room.connect_player(
        player=player,
        user_id=user_id,
        sink=sink,
    )
    if isinstance(connected, Rejected):
        await websocket.close(
            code=_rejection_code(connected.reason),
            reason=connected.reason,
        )
        return
    seat = connected.value
    try:
        while True:
            try:
                raw = await websocket.receive_json()
            except WebSocketDisconnect, OSError:
                return
            frame = read_frame(raw)
            await room.receive(
                seat,
                sink,
                frame.seq,
                frame.decoder,
            )
    finally:
        room.disconnect_player(seat, sink)


def _disconnect_code(reason: DisconnectReason) -> int:
    if reason == DisconnectReason.REPLACED:
        return 4000
    if reason == DisconnectReason.DETACHED:
        return 4408
    return 1000


def _rejection_code(reason: str) -> int:
    if reason in ("invalid player", "missing user id"):
        return 4410
    return 4409
