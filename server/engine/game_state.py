"""Game state data models for 升级 (Shengji/Tractor).

Defines TrickSlot, CompletedTrick, PlayerState, TeamState, GameState,
and GameSettings.  Action/enum types are in the types module.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from server.engine.card import Card, Rank, Suit
from server.engine.types import Phase, PlayType


# ---- Trick Models ----


class TrickSlot(BaseModel):
    """One player's slot in the current trick."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    player_index: int
    cards: list[Card] | None = None


class CompletedTrick(BaseModel):
    """A finished trick with winner and points."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    lead_player_index: int
    lead_type: PlayType
    slots: list[TrickSlot]
    winner_index: int
    points: int


# ---- Player / Team State ----


class PlayerState(BaseModel):
    """Per-player state."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    index: int
    name: str
    hand: list[Card]
    team_index: int
    is_human: bool
    is_declarer: bool


class TeamState(BaseModel):
    """Per-team state."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    index: int
    tricks: list[CompletedTrick]
    current_level: Rank


# ---- Settings ----


class GameSettings(BaseModel):
    """Game configuration settings."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    api_key: str = ""
    model: str = "gpt-4o"
    target_level: Rank = Rank.ACE
    bottom_card_count: int = 8


# ---- Full Game State ----


class GameState(BaseModel):
    """Complete game state snapshot."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    phase: Phase
    current_level: Rank
    players: list[PlayerState]
    teams: list[TeamState]
    current_player_index: int
    trump_suit: Suit | None = None
    trump_rank: Rank
    declarer_team_index: int
    current_trick: list[TrickSlot]
    lead_player_index: int
    lead_play_type: PlayType | None = None
    bottom_cards: list[Card]
    trick_history: list[CompletedTrick]
    last_completed_trick: CompletedTrick | None = None
    bidding_history: list[dict]
    stir_history: list[dict]
    defender_points: int
    settings: GameSettings
