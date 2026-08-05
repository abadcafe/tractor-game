"""Human WebSocket transport at the web/runtime boundary."""

from __future__ import annotations

import logging
from typing import final, override

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter

from server.foundation.result import Rejected
from server.game import Seat
from server.game_runtime import GameRoom
from server.game_runtime.player import (
    ConnectionCloseReason,
    HumanTransport,
    PlayerView,
    UserId,
)

from .game_wire import encode_state, read_frame

logger = logging.getLogger(__name__)
_DYNAMIC_OBJECT = TypeAdapter(object)


@final
class WebSocketTransport(HumanTransport):
    """Encode runtime deliveries onto one WebSocket."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket

    @override
    async def send(self, view: PlayerView) -> None:
        message = encode_state(
            viewer=view.viewer,
            seq=view.seq,
            snapshot=view.snapshot,
            error=view.error,
        )
        try:
            await self._websocket.send_json(
                message.model_dump(mode="json")
            )
        except WebSocketDisconnect, OSError:
            logger.debug("game websocket send failed")

    @override
    async def close(self, reason: ConnectionCloseReason) -> None:
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
    seat: Seat,
    user_id: UserId,
) -> None:
    """Own one human WebSocket until disconnect or takeover."""
    transport = WebSocketTransport(websocket)
    await websocket.accept()
    connected = await room.connect_seat(
        seat=seat,
        user_id=user_id,
        transport=transport,
    )
    if isinstance(connected, Rejected):
        await websocket.close(
            code=_rejection_code(connected.reason),
            reason=connected.reason,
        )
        return
    try:
        while True:
            try:
                raw = _DYNAMIC_OBJECT.validate_python(
                    await websocket.receive_json()
                )
            except WebSocketDisconnect, OSError:
                return
            frame = read_frame(raw)
            await room.receive(
                seat=seat,
                user_id=user_id,
                transport=transport,
                seq=frame.seq,
                decoder=frame.decoder,
            )
    finally:
        room.disconnect_seat(
            seat=seat,
            user_id=user_id,
            transport=transport,
        )


def _disconnect_code(reason: ConnectionCloseReason) -> int:
    if reason == ConnectionCloseReason.REPLACED:
        return 4000
    return 1000


def _rejection_code(reason: str) -> int:
    if reason in ("invalid seat", "missing user id"):
        return 4410
    return 4409


__all__ = ("WebSocketTransport", "handle_game_connection")
