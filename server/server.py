"""Tractor game server -- REST + WebSocket API layer.

Wires Game, Player, and GameRegistry together. Handles HTTP/WS protocol
concerns only. Contains no game logic.
"""

import asyncio
import logging
import os
from collections.abc import Sequence
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from server.actions import (
    BidAction,
    DiscardAction,
    NextRoundAction,
    PlayAction,
    SkipBidAction,
    SkipStirAction,
    StirAction,
)
from server.game import Game
from server.game_registry import GameRegistry
from server.player import AutoPlayer, HumanPlayer, Player
from server.sm.result import Ok, Rejected, StateResult

logger = logging.getLogger(__name__)

_HUMAN_PLAYER_INDEX = 3
registry = GameRegistry()


async def _cleanup_loop():
    """Periodic cleanup of expired games."""
    while True:
        await asyncio.sleep(300)
        registry.cleanup_expired(max_age_seconds=3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)

# ---- REST Endpoints ----


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/game", status_code=201)
async def create_game():
    players: Sequence[Player] = [AutoPlayer(i) for i in range(3)]
    players = list(players)
    players.append(HumanPlayer(_HUMAN_PLAYER_INDEX, ws=None))
    game = Game(players=players)
    game_id = registry.create(game)

    def on_game_over(g: Game) -> None:
        registry.delete(game_id)

    game.set_on_game_over(on_game_over)

    asyncio.create_task(game.run())

    return {"game_id": game_id}


@app.get("/api/game")
async def list_games():
    return {"games": registry.list_games()}


@app.delete("/api/game/{game_id}")
async def delete_game(game_id: str):
    game = registry.get(game_id)
    if game is not None:
        human = game.get_player(_HUMAN_PLAYER_INDEX)
        if human.is_connected():
            await human.on_state(game, seq=game.current_seq)
            await human.close_ws()
        await game.cancel()
    registry.delete(game_id)
    return {"ok": True}


# ---- WebSocket Endpoint ----


@app.websocket("/game/{game_id}")
async def websocket_game(websocket: WebSocket, game_id: str):
    game = registry.get(game_id)
    if game is None:
        await websocket.close(code=4404, reason="game not found")
        return

    human_player = game.get_player(_HUMAN_PLAYER_INDEX)

    # Connection takeover: if an old connection exists, close it first.
    # The old connection's finally block uses clear_ws_if_current(old_ws),
    # so it won't clear the new connection we're about to set up.
    if human_player.is_connected():
        await human_player.close_ws()

    if game.is_over():
        await websocket.accept()
        snapshot = game.snapshot(_HUMAN_PLAYER_INDEX)
        try:
            await websocket.send_json({
                "type": "state",
                "seq": game.current_seq,
                "awaiting": snapshot.awaiting_action,
                "state": snapshot.to_dict(),
            })
        except (WebSocketDisconnect, OSError):
            logger.debug("Failed to send final state in game-over branch (WS disconnected)", exc_info=True)
            return
        try:
            await websocket.close()
        except (WebSocketDisconnect, OSError):
            logger.debug("Failed to close WS in game-over branch (already disconnected)", exc_info=True)
        return

    await websocket.accept()
    human_player.set_ws(websocket)

    while True:
        try:
            raw = await websocket.receive_json()
        except (WebSocketDisconnect, OSError):
            logger.debug("WS receive loop ended (client disconnected)")
            break
        action_type = raw.get("type")
        seq = raw.get("seq", 0)

        # Seq validation: if seq doesn't match current state, push state with error.
        # NOTE: There is an inherent TOCTOU race with the dealing loop — between
        # this seq check and the game.act() call below, the dealing loop may
        # advance _seq via _push_state_to_all(). This means a client with a seq
        # that was valid at validation time could still produce an action applied
        # after the state has already moved forward. The seq check protects
        # against obviously stale clients but does not guarantee atomicity.
        # This is an acceptable trade-off for an async state machine without locking.
        if seq != game.current_seq:
            snapshot = game.snapshot(_HUMAN_PLAYER_INDEX)
            try:
                await websocket.send_json({
                    "type": "state",
                    "seq": game.current_seq,
                    "awaiting": snapshot.awaiting_action,
                    "state": snapshot.to_dict(),
                    "error": f"seq mismatch: expected {game.current_seq}, got {seq}",
                })
            except (WebSocketDisconnect, OSError):
                break
            continue

        parse_result = _parse_action(game, action_type, raw)
        if isinstance(parse_result, Rejected):
            snapshot = game.snapshot(_HUMAN_PLAYER_INDEX)
            try:
                await websocket.send_json({
                    "type": "state",
                    "seq": game.current_seq,
                    "awaiting": snapshot.awaiting_action,
                    "state": snapshot.to_dict(),
                    "error": parse_result.reason,
                })
            except (WebSocketDisconnect, OSError):
                break
            continue

        await game.act(_HUMAN_PLAYER_INDEX, parse_result.value)

        if game.is_over():
            await human_player.on_state(game, seq=game.current_seq)
            break

    human_player.clear_ws_if_current(websocket)


def _parse_action(
    game: Game, action_type: str, raw: dict[str, str | int | bool | None | list[str] | list[dict[str, str]]]
) -> StateResult[BidAction | SkipBidAction | PlayAction | StirAction | SkipStirAction | DiscardAction | NextRoundAction]:
    """Parse a WebSocket JSON message into a PlayerAction."""
    if action_type == "bid":
        pass_val = raw.get("pass", False)
        if isinstance(pass_val, bool) and pass_val:
            return Ok(value=SkipBidAction())
        card_ids_result = _extract_card_ids(_get_cards_list(raw))
        if isinstance(card_ids_result, Rejected):
            return card_ids_result
        resolved_result = game.resolve_cards(_HUMAN_PLAYER_INDEX, card_ids_result.value)
        if isinstance(resolved_result, Rejected):
            return resolved_result
        return Ok(value=BidAction(cards=resolved_result.value, count=len(resolved_result.value)))
    elif action_type == "stir":
        pass_val = raw.get("pass", False)
        if isinstance(pass_val, bool) and pass_val:
            return Ok(value=SkipStirAction())
        card_ids_result = _extract_card_ids(_get_cards_list(raw))
        if isinstance(card_ids_result, Rejected):
            return card_ids_result
        resolved_result = game.resolve_cards(_HUMAN_PLAYER_INDEX, card_ids_result.value)
        if isinstance(resolved_result, Rejected):
            return resolved_result
        return Ok(value=StirAction(cards=resolved_result.value))
    elif action_type == "discard":
        card_ids_result = _extract_card_ids(_get_cards_list(raw))
        if isinstance(card_ids_result, Rejected):
            return card_ids_result
        resolved_result = game.resolve_cards(_HUMAN_PLAYER_INDEX, card_ids_result.value)
        if isinstance(resolved_result, Rejected):
            return resolved_result
        return Ok(value=DiscardAction(cards=resolved_result.value))
    elif action_type == "play":
        card_ids_result = _extract_card_ids(_get_cards_list(raw))
        if isinstance(card_ids_result, Rejected):
            return card_ids_result
        resolved_result = game.resolve_cards(_HUMAN_PLAYER_INDEX, card_ids_result.value)
        if isinstance(resolved_result, Rejected):
            return resolved_result
        return Ok(value=PlayAction(cards=resolved_result.value))
    elif action_type == "next_round":
        return Ok(value=NextRoundAction())
    else:
        return Rejected(reason=f"unknown action type: {action_type}")


def _get_cards_list(raw: dict[str, str | int | bool | None | list[str] | list[dict[str, str]]]) -> Sequence[str | dict[str, str]]:
    """Extract the 'cards' field from a WS message.

    Returns [] if the field is missing or not a list.
    """
    val = raw.get("cards")
    if isinstance(val, list):
        return val
    return []


def _extract_card_ids(cards: Sequence[str | dict[str, str]]) -> StateResult[list[str]]:
    """Extract card ID strings from WS message cards.

    Cards may be plain strings or dicts with an "id" key.
    Returns Rejected if any card dict is missing the 'id' key.
    """
    ids: list[str] = []
    for c in cards:
        if isinstance(c, str):
            ids.append(c)
        else:
            # c is dict[str, str] (the only other option in the union)
            id_raw = c.get("id")
            if id_raw is not None:
                ids.append(id_raw)
            else:
                return Rejected(reason=f"Invalid card format: missing 'id' in {c}")
    return Ok(value=ids)


# ---- Static files ----

static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))


@app.get("/")
async def index():
    html_path = os.path.join(static_dir, "index.html")
    if os.path.isfile(html_path):
        return FileResponse(html_path)
    return Response(status_code=404, content="Frontend not built. Run: deno task build")


@app.get("/{path:path}")
async def serve_static(path: str):
    """Serve static files from static/ directory. Falls back to index.html for unknown paths."""
    file_path = os.path.normpath(os.path.join(static_dir, path))
    if not file_path.startswith(static_dir + os.sep) and file_path != static_dir:
        return Response(status_code=403, content="Forbidden")
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    # Fallback to index.html for SPA routing
    html_path = os.path.join(static_dir, "index.html")
    if os.path.isfile(html_path):
        return FileResponse(html_path)
    return Response(status_code=404, content="Not found")
