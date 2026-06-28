import type { ListedGame } from "./net/rest-client.ts";

/**
 * Resolve the selected lobby game after a game-list refresh.
 */
export function resolveLobbySelectedGameId(
  games: readonly ListedGame[],
  selectedGameId: string | null,
): string | null {
  if (games.length === 0) {
    return null;
  }
  if (
    selectedGameId !== null &&
    games.some((game) => game.gameId === selectedGameId)
  ) {
    return selectedGameId;
  }
  return games[0].gameId;
}

export function selectedGameHasEmptyPlayer(
  games: readonly ListedGame[],
  selectedGameId: string,
): boolean {
  const selectedGame = games.find((game) =>
    game.gameId === selectedGameId
  );
  return selectedGame?.players.some((player) => !player.occupied) ??
    false;
}
