"""Tractor game server -- REST + WebSocket API layer.

Wires Game, Player, and GameRegistry together. Handles HTTP/WS protocol
concerns only. Contains no game logic.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.game import Game
from server.game_registry import GameRegistry
from server.player import (
    AutoPlayer,
    BidAction,
    DiscardAction,
    HumanPlayer,
    NextRoundAction,
    PlayAction,
    SkipStirAction,
    StirAction,
)

logger = logging.getLogger(__name__)

_HUMAN_PLAYER_INDEX = 3
registry = GameRegistry()


async def _cleanup_loop():
    """Periodic cleanup of expired games."""
    while True:
        await asyncio.sleep(300)
        registry.cleanup_expired(max_age_seconds=3600)


@asynccontextmanager
async def lifespan(app):
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
    players = [AutoPlayer(i) for i in range(3)]
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
            try:
                await human.on_state(game)
            except Exception as e:
                logger.debug("Failed to push final state before delete: %s", e)
            human.set_ws(None)
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

    if human_player.is_connected():
        await websocket.close(code=4096, reason="game already connected")
        return

    if game.is_over():
        human_player.set_ws(websocket)
        try:
            await websocket.accept()
            await human_player.on_state(game)
        except Exception:
            pass
        finally:
            human_player.set_ws(None)
            try:
                await websocket.close()
            except Exception:
                pass
        return

    human_player.set_ws(websocket)
    try:
        await websocket.accept()
        await human_player.on_state(game)
    except Exception:
        human_player.set_ws(None)
        return

    try:
        while True:
            raw = await websocket.receive_json()
            action_type = raw.get("type")

            try:
                action = _parse_action(game, action_type, raw)
                await game.act(_HUMAN_PLAYER_INDEX, action)

                if game.is_over():
                    break
            except ValueError as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
            except Exception as e:
                logger.exception("Error processing WS action")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
    except WebSocketDisconnect:
        pass
    finally:
        human_player.set_ws(None)


def _parse_action(
    game: Game, action_type: str, raw: dict
) -> BidAction | PlayAction | StirAction | SkipStirAction | DiscardAction | NextRoundAction:
    """Parse a WebSocket JSON message into a PlayerAction."""
    if action_type == "bid":
        card_ids = _extract_card_ids(raw.get("cards", []))
        resolved = game.resolve_cards(_HUMAN_PLAYER_INDEX, card_ids)
        return BidAction(cards=resolved, count=len(resolved))
    elif action_type == "stir":
        if raw.get("pass", False):
            return SkipStirAction()
        card_ids = _extract_card_ids(raw.get("cards", []))
        resolved = game.resolve_cards(_HUMAN_PLAYER_INDEX, card_ids)
        return StirAction(cards=resolved)
    elif action_type == "discard":
        card_ids = _extract_card_ids(raw.get("cards", []))
        resolved = game.resolve_cards(_HUMAN_PLAYER_INDEX, card_ids)
        return DiscardAction(cards=resolved)
    elif action_type == "play":
        card_ids = _extract_card_ids(raw.get("cards", []))
        resolved = game.resolve_cards(_HUMAN_PLAYER_INDEX, card_ids)
        return PlayAction(cards=resolved)
    elif action_type == "next_round":
        return NextRoundAction()
    else:
        raise ValueError(f"unknown action type: {action_type}")


def _extract_card_ids(cards: list) -> list[str]:
    """Extract card ID strings from WS message cards.

    Cards may be plain strings or dicts with an "id" key.
    """
    ids = []
    for c in cards:
        if isinstance(c, str):
            ids.append(c)
        elif isinstance(c, dict):
            if "id" not in c:
                raise ValueError(f"Invalid card format: missing 'id' field in {c}")
            ids.append(c["id"])
        else:
            raise ValueError(f"Invalid card format: {c}")
    return ids


# ---- Static files ----

_static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/")
async def index():
    html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "index.html"))
    return FileResponse(html_path)
