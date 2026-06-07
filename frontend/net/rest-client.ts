import { API_BASE } from "../config.ts";

/**
 * Create a new game via the REST API.
 * @param baseUrl - optional base URL for testing; defaults to "" (relative paths)
 * @returns the game_id of the created game
 */
export async function createGame(baseUrl: string = ""): Promise<string> {
  const resp = await fetch(`${baseUrl}${API_BASE}`, { method: "POST" });
  if (!resp.ok) {
    throw new Error(`Failed to create game: ${resp.status}`);
  }
  const data = await resp.json();
  return data.game_id;
}

/**
 * List all games via the REST API.
 * @param baseUrl - optional base URL for testing; defaults to "" (relative paths)
 * @returns array of game IDs
 */
export async function listGames(baseUrl: string = ""): Promise<string[]> {
  const resp = await fetch(`${baseUrl}${API_BASE}`, { method: "GET" });
  if (!resp.ok) {
    throw new Error(`Failed to list games: ${resp.status}`);
  }
  const data = await resp.json();
  return data.games;
}

/**
 * Delete a game via the REST API.
 * @param gameId - the game to delete
 * @param baseUrl - optional base URL for testing; defaults to "" (relative paths)
 */
export async function deleteGame(gameId: string, baseUrl: string = ""): Promise<void> {
  const resp = await fetch(`${baseUrl}${API_BASE}/${gameId}`, { method: "DELETE" });
  if (!resp.ok) {
    throw new Error(`Failed to delete game: ${resp.status}`);
  }
}
