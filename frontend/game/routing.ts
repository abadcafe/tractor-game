import {
  GAME_SEAT_PATH,
  type SeatId,
  seatIdFromString,
} from "./config.ts";

export interface GameSeatRoute {
  gameId: string;
  seatId: SeatId;
  userId: string;
}

export function gameSeatHref(
  gameId: string,
  seatId: SeatId,
  userId: string,
): string {
  return GAME_SEAT_PATH(gameId, seatId, userId);
}

export function parseGameSeatRoute(
  pathname: string,
  search: string,
): GameSeatRoute | null {
  const match = /^\/game\/([^/]+)\/seat\/([a-d])$/.exec(pathname);
  if (match === null) {
    return null;
  }
  const gameId = match[1];
  const seatId = seatIdFromString(match[2]);
  const userId = new URLSearchParams(search).get("user_id");
  if (seatId === null || userId === null || userId.trim() === "") {
    return null;
  }
  return { gameId, seatId, userId };
}
