export type PlayerIndex = 0 | 1 | 2 | 3;
export type TeamIndex = 0 | 1;

export const PLAYER_INDEXES: readonly [
  PlayerIndex,
  PlayerIndex,
  PlayerIndex,
  PlayerIndex,
] = [0, 1, 2, 3];

export const DEFAULT_VIEWER_PLAYER: PlayerIndex = 2;

export function playerIndexFromNumber(
  value: number,
): PlayerIndex | null {
  switch (value) {
    case 0:
      return 0;
    case 1:
      return 1;
    case 2:
      return 2;
    case 3:
      return 3;
    default:
      return null;
  }
}

export function WS_PATH(
  gameId: string,
  playerIndex: PlayerIndex,
  userId: string,
): string {
  return GAME_PLAYER_PATH(gameId, playerIndex, userId);
}

export function GAME_PLAYER_PATH(
  gameId: string,
  playerIndex: PlayerIndex,
  userId: string,
): string {
  return `/game/${encodeURIComponent(gameId)}/player/${playerIndex}` +
    `?user_id=${encodeURIComponent(userId)}`;
}

export const API_BASE = "/api/game";
export const PLAYER_LEFT_WS_CLOSE_CODE = 4408;

export const TEAM_LABELS: Record<TeamIndex, string> = {
  0: "队伍 0",
  1: "队伍 1",
};
