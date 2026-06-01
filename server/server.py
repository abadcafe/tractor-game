"""
升级 (Shengji / Tractor) Game Server

Serves the static frontend and provides API endpoints for AI player decisions.
Uses LangGraph for multi-step AI reasoning with session management.
"""

import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure server/ is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

from ai.models import DecideRequest, DecideResponse, SessionInfo
from ai.session_manager import session_manager
from ai.graph import run_ai_decision
from ai.prompts import build_role_description

from server.api_types import (
    BidRequest,
    DiscardRequest,
    GameStateResponse,
    LegalPlayAction,
    PlayRequest,
    SetTrumpRequest,
    StirRequest,
)
from server.engine.card import Rank, Suit
from server.engine.constants import PLAYER_COUNT
from server.engine.game import Game
from server.engine.game_state import GameSettings
from server.engine.types import Phase, StirAction
from server.resilience import cleanup_expired_sessions, get_settings, update_settings
from server.storage.game_store import GameStore

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle: run periodic session cleanup."""
    async def _cleanup_loop() -> None:
        while True:
            await asyncio.sleep(300)  # every 5 minutes
            try:
                removed = cleanup_expired_sessions(store, max_age_seconds=3600)
                if removed > 0:
                    logger.info("Cleaned up %d expired session(s)", removed)
            except Exception:
                logger.exception("Session cleanup failed, will retry in 5 minutes")

    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()


app = FastAPI(title="Tractor Game Server", lifespan=lifespan)

# ---- Game Store ----

store = GameStore()

# ---- Configuration ----

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# ---- Serve static frontend ----

static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)

def _resolve_static(path: str) -> Path | None:
    """Resolve a static file path, trying .js extension if no extension given."""
    full = static_dir / path
    if full.is_file():
        return full
    # Try appending .js for extensionless ES module imports
    js = static_dir / f"{path}.js"
    if js.is_file():
        return js
    return None

@app.get("/static/{path:path}")
async def serve_static(path: str):
    """Serve static files, with .js fallback for extensionless imports."""
    file = _resolve_static(path)
    if file is None:
        raise StarletteHTTPException(status_code=404)
    return FileResponse(str(file))


# ---- API routes ----

@app.get("/api/health")
async def health():
    return {"status": "ok", "sessions": len(session_manager._sessions)}


@app.post("/api/ai/decide", response_model=DecideResponse)
async def ai_decide(req: DecideRequest):
    """
    Main AI decision endpoint.
    Runs the LangGraph pipeline for a single AI player decision.
    """
    gs = req.game_state

    # Determine role
    team_index = gs.get("my_team_index", 0)
    is_declarer = gs.get("my_is_declarer", False)
    role = build_role_description(req.player_index, team_index, is_declarer)

    # Get or create session
    session = session_manager.get_or_create(req.player_index, role)

    # Run decision pipeline
    result = await run_ai_decision(
        player_index=req.player_index,
        phase=req.phase,
        game_state=gs,
        hand=[h.model_dump() for h in req.hand],
        legal_actions=req.legal_actions,
        session=session,
        model=req.model or DEFAULT_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    # Save session after decision
    session_manager.save(req.player_index)

    return DecideResponse(
        action_type=result.get("action_type", "play"),
        card_ids=result.get("card_ids", []),
        reasoning=result.get("reasoning", ""),
    )


@app.get("/api/session/{player_index}", response_model=SessionInfo)
async def get_session(player_index: int):
    """Get session info for debugging."""
    session = session_manager.get(player_index)
    if session is None:
        return SessionInfo(player_index=player_index, role="unknown")
    return session.to_info()


@app.post("/api/session/{player_index}/reset")
async def reset_session(player_index: int):
    """Reset a player's AI session."""
    session_manager.reset(player_index)
    return {"status": "ok"}


@app.post("/api/config")
async def update_config(data: dict):
    """Update API configuration at runtime."""
    global API_KEY, BASE_URL, DEFAULT_MODEL
    if "api_key" in data:
        API_KEY = data["api_key"]
    if "base_url" in data:
        BASE_URL = data["base_url"]
    if "model" in data:
        DEFAULT_MODEL = data["model"]
    return {"status": "ok"}


@app.get("/api/settings")
async def read_settings():
    """Return current server-side game settings."""
    return get_settings()


@app.put("/api/settings")
async def write_settings(data: GameSettings):
    """Update server-side game settings."""
    update_settings(**data.model_dump(exclude_unset=True))
    return get_settings()


# ---- Game API helpers ----


def _game_response(game_id: str, game: Game) -> GameStateResponse:
    """Build the game state response for the client."""
    # Compute legal actions when in playing phase
    legal_actions = None
    if game.state.phase == Phase.PLAYING:
        legal_plays = game.get_legal_plays(game.state.current_player_index)
        legal_actions = [
            LegalPlayAction(type=play.type.value, cards=[c.id for c in play.cards])
            for play in legal_plays
        ]

    # Compute valid bid levels when in bidding phase
    valid_bid_levels = None
    if game.state.phase == Phase.BIDDING:
        bids = game.get_valid_bids()
        valid_bid_levels = [b.value for b in bids]

    # Compute scoring message/details when in scoring phase
    scoring_message = None
    scoring_details = None
    if game.state.phase == Phase.SCORING:
        result = game.get_round_score()
        declarer_team = game.state.declarer_team_index
        pts = result.total_defender_points
        change = result.declarer_level_change

        if change > 0:
            outcome = f"庄家升 {change} 级"
        elif change < 0:
            outcome = f"闲家升 {-change} 级"
        else:
            outcome = "换庄"

        scoring_message = f"防守方得分: {pts} — {outcome}"

        team0_cur = game.state.teams[0].current_level.value
        team1_cur = game.state.teams[1].current_level.value
        team0_new = result.team0_new_level.value
        team1_new = result.team1_new_level.value
        scoring_details = (
            f"队0 (庄家{'*' if declarer_team == 0 else ''}) {team0_cur} → {team0_new}  "
            f"队1 (庄家{'*' if declarer_team == 1 else ''}) {team1_cur} → {team1_new}  "
            f"扣底加成: {result.bottom_card_bonus}"
        )

    return GameStateResponse(
        game_id=game_id,
        state=game.state,
        awaiting_action=game.get_awaiting_action(),
        legal_actions=legal_actions,
        valid_bid_levels=valid_bid_levels,
        scoring_message=scoring_message,
        scoring_details=scoring_details,
        winning_team=game.get_winning_team(),
    )


def _get_game_or_404(game_id: str) -> Game:
    """Retrieve a game by ID or raise 404."""
    state = store.get(game_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return Game.from_state(state)


# ---- Game API endpoints ----

@app.post("/api/game")
async def create_game():
    """Create a new game in DEALING phase."""
    game = Game()
    game.start_new_game()
    game_id = store.create(game.state)
    return _game_response(game_id, game)


@app.get("/api/game/{game_id}")
async def get_game(game_id: str):
    """Get the current game state."""
    game = _get_game_or_404(game_id)
    return _game_response(game_id, game)


@app.post("/api/game/{game_id}/deal")
async def deal(game_id: str):
    """Deal cards and transition to BIDDING phase."""
    game = _get_game_or_404(game_id)
    game.start_round()
    # Auto-play AI turns so the game reaches the human player's turn
    game._ai_auto_play()
    store.update(game_id, game.state)
    return _game_response(game_id, game)


@app.post("/api/game/{game_id}/bid")
async def bid(game_id: str, req: BidRequest):
    """Submit a bid for the current player."""
    game = _get_game_or_404(game_id)
    level = Rank(req.level) if req.level else None
    success = game.submit_bid(req.player_index, level, pass_=req.pass_)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid bid")
    store.update(game_id, game.state)
    return _game_response(game_id, game)


@app.post("/api/game/{game_id}/set-trump")
async def set_trump(game_id: str, req: SetTrumpRequest):
    """Set the trump suit after winning the bid."""
    game = _get_game_or_404(game_id)
    trump_suit = Suit(req.trump_suit)
    success = game.set_trump(req.player_index, trump_suit)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid set-trump")
    store.update(game_id, game.state)
    return _game_response(game_id, game)


@app.post("/api/game/{game_id}/stir")
async def stir(game_id: str, req: StirRequest):
    """Submit a stir action or pass."""
    game = _get_game_or_404(game_id)
    if req.pass_:
        success = game.submit_stir(req.player_index, None)
    else:
        stir_action = StirAction(
            player_index=req.player_index,
            new_trump_suit=Suit(req.new_trump_suit) if req.new_trump_suit else None,
            level=Rank(req.level) if req.level else None,
        )
        success = game.submit_stir(req.player_index, stir_action)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid stir")
    store.update(game_id, game.state)
    return _game_response(game_id, game)


@app.post("/api/game/{game_id}/discard")
async def discard(game_id: str, req: DiscardRequest):
    """Discard bottom cards after picking them up."""
    game = _get_game_or_404(game_id)
    if not 0 <= req.player_index < PLAYER_COUNT:
        raise HTTPException(status_code=400, detail=f"player_index out of range: {req.player_index}")
    player = game.state.players[req.player_index]
    hand_by_id = {c.id: c for c in player.hand}
    invalid_ids = [cid for cid in req.card_ids if cid not in hand_by_id]
    if invalid_ids:
        raise HTTPException(status_code=400, detail=f"Card IDs not in hand: {invalid_ids}")
    cards = [hand_by_id[cid] for cid in req.card_ids]
    success = game.submit_discard(req.player_index, cards)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid discard")
    store.update(game_id, game.state)
    return _game_response(game_id, game)


@app.post("/api/game/{game_id}/play")
async def play(game_id: str, req: PlayRequest):
    """Play cards from a player's hand."""
    game = _get_game_or_404(game_id)
    if not 0 <= req.player_index < PLAYER_COUNT:
        raise HTTPException(status_code=400, detail=f"player_index out of range: {req.player_index}")
    player = game.state.players[req.player_index]
    hand_by_id = {c.id: c for c in player.hand}
    invalid_ids = [cid for cid in req.card_ids if cid not in hand_by_id]
    if invalid_ids:
        raise HTTPException(status_code=400, detail=f"Card IDs not in hand: {invalid_ids}")
    cards = [hand_by_id[cid] for cid in req.card_ids]
    success = game.submit_play(req.player_index, cards)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid play")
    store.update(game_id, game.state)
    return _game_response(game_id, game)


@app.post("/api/game/{game_id}/clear-trick")
async def clear_trick(game_id: str):
    """Clear the current trick for the next trick."""
    game = _get_game_or_404(game_id)
    if game.state.phase != Phase.PLAYING:
        raise HTTPException(status_code=400, detail="Not in playing phase")
    game.clear_trick()
    store.update(game_id, game.state)
    return _game_response(game_id, game)


@app.post("/api/game/{game_id}/next-round")
async def next_round(game_id: str):
    """Calculate scores and advance to the next round."""
    game = _get_game_or_404(game_id)
    if game.state.phase != Phase.SCORING:
        raise HTTPException(status_code=400, detail="Not in scoring phase")
    game.next_round()
    store.update(game_id, game.state)
    return _game_response(game_id, game)


# ---- Serve index.html for root ----

@app.get("/")
async def index():
    index_path = Path(__file__).parent.parent / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Tractor Game Server is running."}


# ---- Logger ----

logger = logging.getLogger(__name__)

# ---- Startup ----

if __name__ == "__main__":
    print(f"Starting Tractor Game Server on http://localhost:8787")
    print(f"Static files: {static_dir}")
    print(f"API Key configured: {'Yes' if API_KEY else 'No (set OPENAI_API_KEY env var)'}")
    uvicorn.run(app, host="0.0.0.0", port=8787)
