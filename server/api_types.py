"""API-layer data models for the Tractor game server.

Defines request/response types used by the REST API endpoints.
These are API concerns, not engine concerns -- they live here to
decouple the engine from the HTTP layer.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from server.engine.game_state import GameState


class LegalPlayAction(BaseModel):
    """A legal play option consisting of a play type and card IDs."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    type: str
    cards: list[str]


class GameStateResponse(BaseModel):
    """Full game state response returned to the client."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    game_id: str
    state: GameState
    awaiting_action: str | None = None
    legal_actions: list[LegalPlayAction] | None = None
    valid_bid_levels: list[str] | None = None
    scoring_message: str | None = None
    scoring_details: str | None = None
    winning_team: int | None = None


class CreateGameRequest(BaseModel):
    """Request to create a new game."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class BidRequest(BaseModel):
    """Request to submit a bid."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    player_index: int
    level: str | None = None
    pass_: bool = Field(alias="pass")


class SetTrumpRequest(BaseModel):
    """Request to set the trump suit after winning a bid."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    player_index: int
    trump_suit: str


class StirRequest(BaseModel):
    """Request to stir (change trump suit during stirring phase)."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    player_index: int
    new_trump_suit: str | None = None
    level: str | None = None
    pass_: bool = Field(alias="pass")


class DiscardRequest(BaseModel):
    """Request to discard bottom cards."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    player_index: int
    card_ids: list[str]


class PlayRequest(BaseModel):
    """Request to play cards."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    player_index: int
    card_ids: list[str]
