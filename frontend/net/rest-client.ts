import { API_BASE } from "../config.ts";

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
  const data = await resp.json();
  return data.game_id;
}
