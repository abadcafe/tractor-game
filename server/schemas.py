"""API schema models for OpenAPI documentation."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = Field(examples=["ok"])


class CreateGameResponse(BaseModel):
    game_id: str = Field(examples=["abc123def456"], description="UUID hex string identifying the new game")


class GameInfo(BaseModel):
    game_id: str = Field(examples=["abc123def456"])
    phase: str = Field(examples=["IDLE"])


class ListGamesResponse(BaseModel):
    games: list[GameInfo] = Field(default_factory=list)


class DeleteGameResponse(BaseModel):
    ok: Literal[True] = Field(examples=[True])
