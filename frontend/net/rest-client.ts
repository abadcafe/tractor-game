import { API_BASE, type PlayerIndex } from "../config.ts";

export type { PlayerIndex } from "../config.ts";
export type PlayerKind = "empty" | "user" | "ai" | "auto";
export type BotFillKind = "ai" | "auto";
export type BotFillMode = "none" | BotFillKind;

export interface ListedPlayer {
  index: PlayerIndex;
  occupied: boolean;
  connected: boolean;
  kind?: PlayerKind;
  mine: boolean;
  ready: boolean;
}

export interface ListedGame {
  gameId: string;
  userCount: number;
  capacity: number;
  userPlayers: PlayerIndex[];
  players: ListedPlayer[];
}

type ListedPlayerWire = {
  index: PlayerIndex;
  occupied: boolean;
  connected: boolean;
  kind: PlayerKind;
  mine: boolean;
  ready: boolean;
};

type ListedGameWire = {
  game_id: string;
  user_count: number;
  capacity: number;
  user_players: PlayerIndex[];
  players: ListedPlayerWire[];
};

type GameListResponseWire = {
  games: ListedGameWire[];
};

type CreateGameResponseWire = {
  game_id: string;
};

type PlayerOperationResponseWire = {
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

export async function joinPlayer(
  gameId: string,
  playerIndex: PlayerIndex,
  userId: string,
  baseUrl: string = "",
): Promise<boolean> {
  return await sendPlayerOperation(
    "POST",
    gameId,
    playerIndex,
    userId,
    baseUrl,
  );
}

export async function leavePlayer(
  gameId: string,
  playerIndex: PlayerIndex,
  userId: string,
  baseUrl: string = "",
): Promise<boolean> {
  return await sendPlayerOperation(
    "DELETE",
    gameId,
    playerIndex,
    userId,
    baseUrl,
  );
}

export async function fillBotPlayers(
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
    throw new Error(`Failed to fill bot players: ${resp.status}`);
  }
  const data: unknown = await resp.json();
  if (!isPlayerOperationResponseWire(data)) {
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
  if (!isPlayerOperationResponseWire(data)) {
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
  const userPlayers = value["user_players"];
  const players = value["players"];
  return typeof gameId === "string" &&
    isNonNegativeInteger(userCount) &&
    isPositiveInteger(capacity) &&
    Array.isArray(userPlayers) && userPlayers.every(isPlayerIndex) &&
    Array.isArray(players) && players.every(isListedPlayerWire);
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

function isPlayerOperationResponseWire(
  value: unknown,
): value is PlayerOperationResponseWire {
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

function isPlayerIndex(value: unknown): value is PlayerIndex {
  return value === 0 || value === 1 || value === 2 || value === 3;
}

function isListedPlayerWire(value: unknown): value is ListedPlayerWire {
  if (!isRecord(value)) {
    return false;
  }
  const index = value["index"];
  const occupied = value["occupied"];
  const connected = value["connected"];
  const kind = value["kind"];
  const mine = value["mine"];
  const ready = value["ready"];
  return isPlayerIndex(index) &&
    typeof occupied === "boolean" &&
    typeof connected === "boolean" &&
    isPlayerKind(kind) &&
    typeof mine === "boolean" &&
    typeof ready === "boolean";
}

function isPlayerKind(value: unknown): value is PlayerKind {
  return value === "empty" || value === "user" || value === "ai" ||
    value === "auto";
}

function listedGameFromWire(game: ListedGameWire): ListedGame {
  return {
    gameId: game.game_id,
    userCount: game.user_count,
    capacity: game.capacity,
    userPlayers: game.user_players,
    players: game.players.map(listedPlayerFromWire),
  };
}

function listedPlayerFromWire(player: ListedPlayerWire): ListedPlayer {
  return {
    index: player.index,
    occupied: player.occupied,
    connected: player.connected,
    kind: player.kind,
    mine: player.mine,
    ready: player.ready,
  };
}

async function sendPlayerOperation(
  method: "POST" | "DELETE",
  gameId: string,
  playerIndex: PlayerIndex,
  userId: string,
  baseUrl: string,
): Promise<boolean> {
  const resp = await fetch(
    `${baseUrl}${playerApiPath(gameId, playerIndex, userId)}`,
    { method },
  );
  if (!resp.ok) {
    throw new Error(`Failed to update player: ${resp.status}`);
  }
  const data: unknown = await resp.json();
  if (!isPlayerOperationResponseWire(data)) {
    throw new Error("Invalid player operation response");
  }
  return data.ok;
}

function playerApiPath(
  gameId: string,
  playerIndex: PlayerIndex,
  userId: string,
): string {
  return `${API_BASE}/${
    encodeURIComponent(gameId)
  }/player/${playerIndex}` +
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
