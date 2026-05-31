"""
Pydantic models for AI API requests and responses.
"""

from pydantic import BaseModel
from typing import Optional, Any


class CardInfo(BaseModel):
    id: str
    suit: str
    rank: str
    is_joker: bool = False
    is_big_joker: bool = False
    points: int = 0
    display: str = ""


class GameStateInfo(BaseModel):
    phase: str
    current_level: str
    trump_suit: Optional[str] = None
    trump_rank: str
    declarer_team: int
    defender_points: int
    current_trick: list[dict] = []
    lead_player: int
    lead_play_type: Optional[str] = None
    my_team_index: int
    my_is_declarer: bool
    trick_count: int


class DecideRequest(BaseModel):
    player_index: int
    phase: str  # "bidding", "stirring", "exchange", "playing"
    game_state: dict[str, Any]
    hand: list[CardInfo]
    legal_actions: list[str]
    model: str = "gpt-4o"


class DecideResponse(BaseModel):
    action_type: str  # "play", "bid", "stir", "discard", "pass"
    card_ids: list[str] = []
    reasoning: str = ""


class SessionInfo(BaseModel):
    player_index: int
    role: str
    strategy: Optional[dict] = None
    opponent_models: dict = {}
    key_memories: list[dict] = []
    total_tokens_used: int = 0
