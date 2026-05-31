"""
Session manager for AI players.

Each AI player has an independent session with:
- Message history (for context management)
- Current strategy (persisted across tricks)
- Opponent models (updated based on observations)
- Key memories (important events)
- Token usage tracking
"""

import json
import time
from pathlib import Path
from typing import Optional

from .models import SessionInfo


class AISession:
    """Single AI player's session."""

    def __init__(self, player_index: int, role: str):
        self.player_index = player_index
        self.role = role
        self.strategy: Optional[dict] = None
        self.opponent_models: dict[str, dict] = {}
        self.key_memories: list[dict] = []
        self.message_history: list[dict] = []  # For token management
        self.total_tokens_used: int = 0
        self.created_at = time.time()
        self.last_updated = time.time()

    def update_strategy(self, strategy: dict) -> None:
        """Update the current strategy."""
        self.strategy = strategy
        self.last_updated = time.time()

    def add_memory(self, event: str, trick_number: int) -> None:
        """Record a key event."""
        self.key_memories.append({
            "trick": trick_number,
            "event": event,
            "time": time.time(),
        })
        # Keep only recent 20 memories
        if len(self.key_memories) > 20:
            self.key_memories = self.key_memories[-20:]
        self.last_updated = time.time()

    def update_opponent_model(self, player_id: str, observations: dict) -> None:
        """Update model of an opponent."""
        if player_id not in self.opponent_models:
            self.opponent_models[player_id] = {
                "observed_tendencies": [],
                "estimated_trump_left": "unknown",
                "likely_void_suits": [],
                "note": "",
            }

        model = self.opponent_models[player_id]
        if "note" in observations:
            model["note"] = observations["note"]
        if "estimated_trump" in observations:
            model["estimated_trump_left"] = observations["estimated_trump"]
        if "void_suit" in observations:
            suit = observations["void_suit"]
            if suit not in model["likely_void_suits"]:
                model["likely_void_suits"].append(suit)
        if "tendency" in observations:
            if observations["tendency"] not in model["observed_tendencies"]:
                model["observed_tendencies"].append(observations["tendency"])

        self.last_updated = time.time()

    def add_tokens(self, count: int) -> None:
        """Track token usage."""
        self.total_tokens_used += count

    def to_info(self) -> SessionInfo:
        """Convert to info model."""
        return SessionInfo(
            player_index=self.player_index,
            role=self.role,
            strategy=self.strategy,
            opponent_models=self.opponent_models,
            key_memories=self.key_memories,
            total_tokens_used=self.total_tokens_used,
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence."""
        return {
            "player_index": self.player_index,
            "role": self.role,
            "strategy": self.strategy,
            "opponent_models": self.opponent_models,
            "key_memories": self.key_memories,
            "total_tokens_used": self.total_tokens_used,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AISession":
        """Deserialize from dictionary."""
        session = cls(data["player_index"], data.get("role", "unknown"))
        session.strategy = data.get("strategy")
        session.opponent_models = data.get("opponent_models", {})
        session.key_memories = data.get("key_memories", [])
        session.total_tokens_used = data.get("total_tokens_used", 0)
        session.created_at = data.get("created_at", time.time())
        session.last_updated = data.get("last_updated", time.time())
        return session


class SessionManager:
    """Manages all AI player sessions."""

    def __init__(self, storage_dir: Path | None = None):
        self.storage_dir = storage_dir or Path(__file__).parent.parent / "sessions"
        self.storage_dir.mkdir(exist_ok=True)
        self._sessions: dict[int, AISession] = {}

    def get_or_create(self, player_index: int, role: str) -> AISession:
        """Get existing session or create a new one."""
        if player_index in self._sessions:
            return self._sessions[player_index]

        # Try loading from disk
        session = self._load(player_index)
        if session is None:
            session = AISession(player_index, role)

        self._sessions[player_index] = session
        return session

    def get(self, player_index: int) -> Optional[AISession]:
        """Get a session if it exists."""
        if player_index not in self._sessions:
            self._sessions[player_index] = self._load(player_index)  # type: ignore
        return self._sessions.get(player_index)

    def save(self, player_index: int) -> None:
        """Persist a session to disk."""
        session = self._sessions.get(player_index)
        if session is None:
            return

        path = self.storage_dir / f"player_{player_index}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

    def save_all(self) -> None:
        """Persist all sessions."""
        for idx in self._sessions:
            self.save(idx)

    def reset(self, player_index: int) -> None:
        """Reset a player's session (new game)."""
        if player_index in self._sessions:
            del self._sessions[player_index]
        path = self.storage_dir / f"player_{player_index}.json"
        if path.exists():
            path.unlink()

    def reset_all(self) -> None:
        """Reset all sessions."""
        self._sessions.clear()
        for path in self.storage_dir.glob("player_*.json"):
            path.unlink()

    def _load(self, player_index: int) -> Optional[AISession]:
        """Load a session from disk."""
        path = self.storage_dir / f"player_{player_index}.json"
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return AISession.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None


# Global session manager instance
session_manager = SessionManager()
