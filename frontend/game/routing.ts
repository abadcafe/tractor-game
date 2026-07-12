import { GAME_PLAYER_PATH, type PlayerIndex } from "./config.ts";

export interface GamePlayerRoute {
  gameId: string;
  playerIndex: PlayerIndex;
  userId: string;
}

export function gamePlayerHref(
  gameId: string,
  playerIndex: PlayerIndex,
  userId: string,
): string {
  return GAME_PLAYER_PATH(gameId, playerIndex, userId);
}

export function parseGamePlayerRoute(
  pathname: string,
  search: string,
): GamePlayerRoute | null {
  const match = /^\/game\/([^/]+)\/player\/([0-3])$/.exec(pathname);
  if (match === null) {
    return null;
  }
  const gameId = match[1];
  const playerIndex = playerIndexFromString(match[2]);
  const userId = new URLSearchParams(search).get("user_id");
  if (
    playerIndex === null || userId === null || userId.trim() === ""
  ) {
    return null;
  }
  return {
    gameId,
    playerIndex,
    userId,
  };
}

function playerIndexFromString(value: string): PlayerIndex | null {
  switch (value) {
    case "0":
      return 0;
    case "1":
      return 1;
    case "2":
      return 2;
    case "3":
      return 3;
    default:
      return null;
  }
}
