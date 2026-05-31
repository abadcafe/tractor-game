"""
升级 (Shengji / Tractor) Game Server

Serves the static frontend and provides API endpoints for AI player decisions.
Uses LangGraph for multi-step AI reasoning with session management.
"""

import os
import sys
from pathlib import Path

# Ensure server/ is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

from ai.models import DecideRequest, DecideResponse, SessionInfo
from ai.session_manager import session_manager
from ai.graph import run_ai_decision
from ai.prompts import build_role_description

app = FastAPI(title="Tractor Game Server")

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


# ---- Serve index.html for root ----

@app.get("/")
async def index():
    index_path = Path(__file__).parent.parent / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Tractor Game Server is running."}


# ---- Startup ----

if __name__ == "__main__":
    print(f"Starting Tractor Game Server on http://localhost:8787")
    print(f"Static files: {static_dir}")
    print(f"API Key configured: {'Yes' if API_KEY else 'No (set OPENAI_API_KEY env var)'}")
    uvicorn.run(app, host="0.0.0.0", port=8787)
