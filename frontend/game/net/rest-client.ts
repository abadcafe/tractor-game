import { API_BASE, type SeatId } from "../config.ts";

export type { SeatId } from "../config.ts";
export type OccupantKind = "empty" | "user" | "ai" | "auto";
export type BotFillKind = "ai" | "auto";
export type BotFillMode = "none" | BotFillKind;

export interface ListedSeat {
  seat: SeatId;
  occupied: boolean;
  connected: boolean;
  kind?: OccupantKind;
  mine: boolean;
  ready: boolean;
}

export interface ListedGame {
  gameId: string;
  userCount: number;
  capacity: number;
  userSeats: SeatId[];
  seats: ListedSeat[];
}

type ListedSeatWire = {
  seat: SeatId;
  occupied: boolean;
  connected: boolean;
  kind: OccupantKind;
  mine: boolean;
  ready: boolean;
};

type ListedGameWire = {
  game_id: string;
  user_count: number;
  capacity: number;
  user_seats: SeatId[];
  seats: ListedSeatWire[];
};

type GameListResponseWire = {
  games: ListedGameWire[];
};

type CreateGameResponseWire = {
  game_id: string;
};

type SeatOperationResponseWire = {
  ok: boolean;
};

/**
 * Create a new game via the REST API.
 * @param baseUrl - optional base URL for testing; defaults to "" (relative paths)
 * @returns the game_id of the created game
 */
export async function createGame(
  baseUrl: string = "",
): Promise<string> {
  const resp = await fetch(`${baseUrl}${API_BASE}`, { method: "POST" });
  if (!resp.ok) {
    throw new Error(`Failed to create game: ${resp.status}`);
  }
  const data: unknown = await resp.json();
  if (!isCreateGameResponseWire(data)) {
    throw new Error("Invalid create game response");
  }
  return data.game_id;
}

export async function listGames(
  baseUrl: string = "",
  userId?: string,
): Promise<ListedGame[]> {
  const query = userId === undefined
    ? ""
    : `?user_id=${encodeURIComponent(userId)}`;
  const resp = await fetch(`${baseUrl}${API_BASE}${query}`, {
    method: "GET",
  });
  if (!resp.ok) {
    throw new Error(`Failed to list games: ${resp.status}`);
  }
  const data: unknown = await resp.json();
  if (!isGameListResponseWire(data)) {
    throw new Error("Invalid game list response");
  }
  return data.games.map(listedGameFromWire);
}

export async function occupySeat(
  gameId: string,
  seatId: SeatId,
  userId: string,
  baseUrl: string = "",
): Promise<boolean> {
  return await sendSeatOperation(
    "POST",
    gameId,
    seatId,
    userId,
    baseUrl,
  );
}

export async function vacateSeat(
  gameId: string,
  seatId: SeatId,
  userId: string,
  baseUrl: string = "",
): Promise<boolean> {
  return await sendSeatOperation(
    "DELETE",
    gameId,
    seatId,
    userId,
    baseUrl,
  );
}

export async function fillBotSeats(
  gameId: string,
  kind: BotFillKind,
  userId: string,
  baseUrl: string = "",
): Promise<boolean> {
  const resp = await fetch(
    `${baseUrl}${botFillApiPath(gameId, kind, userId)}`,
    { method: "POST" },
  );
  if (!resp.ok) {
    throw new Error(`Failed to fill bot seats: ${resp.status}`);
  }
  const data: unknown = await resp.json();
  if (!isSeatOperationResponseWire(data)) {
    throw new Error("Invalid bot fill response");
  }
  return data.ok;
}

export async function deleteGame(
  gameId: string,
  baseUrl: string = "",
): Promise<boolean> {
  const resp = await fetch(
    `${baseUrl}${API_BASE}/${encodeURIComponent(gameId)}`,
    { method: "DELETE" },
  );
  if (!resp.ok) {
    throw new Error(`Failed to delete game: ${resp.status}`);
  }
  const data: unknown = await resp.json();
  if (!isSeatOperationResponseWire(data)) {
    throw new Error("Invalid delete game response");
  }
  return data.ok;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isListedGameWire(value: unknown): value is ListedGameWire {
  if (!isRecord(value)) {
    return false;
  }
  const gameId = value["game_id"];
  const userCount = value["user_count"];
  const capacity = value["capacity"];
  const userSeats = value["user_seats"];
  const seats = value["seats"];
  return typeof gameId === "string" &&
    isNonNegativeInteger(userCount) &&
    isPositiveInteger(capacity) &&
    Array.isArray(userSeats) && userSeats.every(isSeatId) &&
    Array.isArray(seats) && seats.every(isListedSeatWire);
}

function isGameListResponseWire(
  value: unknown,
): value is GameListResponseWire {
  if (!isRecord(value)) {
    return false;
  }
  const games = value["games"];
  return Array.isArray(games) && games.every(isListedGameWire);
}

function isCreateGameResponseWire(
  value: unknown,
): value is CreateGameResponseWire {
  return isRecord(value) && typeof value["game_id"] === "string";
}

function isSeatOperationResponseWire(
  value: unknown,
): value is SeatOperationResponseWire {
  return isRecord(value) && typeof value["ok"] === "boolean";
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) &&
    value >= 0;
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) &&
    value > 0;
}

function isSeatId(value: unknown): value is SeatId {
  return value === "a" || value === "b" || value === "c" ||
    value === "d";
}

function isListedSeatWire(value: unknown): value is ListedSeatWire {
  if (!isRecord(value)) {
    return false;
  }
  const seat = value["seat"];
  const occupied = value["occupied"];
  const connected = value["connected"];
  const kind = value["kind"];
  const mine = value["mine"];
  const ready = value["ready"];
  return isSeatId(seat) &&
    typeof occupied === "boolean" &&
    typeof connected === "boolean" &&
    isOccupantKind(kind) &&
    typeof mine === "boolean" &&
    typeof ready === "boolean";
}

function isOccupantKind(value: unknown): value is OccupantKind {
  return value === "empty" || value === "user" || value === "ai" ||
    value === "auto";
}

function listedGameFromWire(game: ListedGameWire): ListedGame {
  return {
    gameId: game.game_id,
    userCount: game.user_count,
    capacity: game.capacity,
    userSeats: game.user_seats,
    seats: game.seats.map(listedSeatFromWire),
  };
}

function listedSeatFromWire(seat: ListedSeatWire): ListedSeat {
  return {
    seat: seat.seat,
    occupied: seat.occupied,
    connected: seat.connected,
    kind: seat.kind,
    mine: seat.mine,
    ready: seat.ready,
  };
}

async function sendSeatOperation(
  method: "POST" | "DELETE",
  gameId: string,
  seatId: SeatId,
  userId: string,
  baseUrl: string,
): Promise<boolean> {
  const resp = await fetch(
    `${baseUrl}${seatApiPath(gameId, seatId, userId)}`,
    { method },
  );
  if (!resp.ok) {
    throw new Error(`Failed to update seat: ${resp.status}`);
  }
  const data: unknown = await resp.json();
  if (!isSeatOperationResponseWire(data)) {
    throw new Error("Invalid seat operation response");
  }
  return data.ok;
}

function seatApiPath(
  gameId: string,
  seatId: SeatId,
  userId: string,
): string {
  return `${API_BASE}/${encodeURIComponent(gameId)}/seat/${seatId}` +
    `?user_id=${encodeURIComponent(userId)}`;
}

function botFillApiPath(
  gameId: string,
  kind: BotFillKind,
  userId: string,
): string {
  return `${API_BASE}/${encodeURIComponent(gameId)}/bots` +
    `?kind=${encodeURIComponent(kind)}` +
    `&user_id=${encodeURIComponent(userId)}`;
}
