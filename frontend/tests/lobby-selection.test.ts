import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import type { ListedGame } from "../net/rest-client.ts";
import {
  resolveLobbySelectedGameId,
  selectedGameHasEmptyPlayer,
} from "../lobby-selection.ts";

function makeGame(gameId: string): ListedGame {
  return {
    gameId,
    userCount: 0,
    capacity: 4,
    userPlayers: [],
    players: [
      {
        index: 0,
        occupied: false,
        connected: false,
        mine: false,
        ready: false,
      },
      {
        index: 1,
        occupied: false,
        connected: false,
        mine: false,
        ready: false,
      },
      {
        index: 2,
        occupied: false,
        connected: false,
        mine: false,
        ready: false,
      },
      {
        index: 3,
        occupied: false,
        connected: false,
        mine: false,
        ready: false,
      },
    ],
  };
}

function makeFilledGame(gameId: string): ListedGame {
  return {
    ...makeGame(gameId),
    players: [
      {
        index: 0,
        occupied: true,
        connected: false,
        kind: "auto",
        mine: false,
        ready: true,
      },
      {
        index: 1,
        occupied: true,
        connected: false,
        kind: "user",
        mine: true,
        ready: false,
      },
      {
        index: 2,
        occupied: true,
        connected: false,
        kind: "auto",
        mine: false,
        ready: true,
      },
      {
        index: 3,
        occupied: true,
        connected: false,
        kind: "auto",
        mine: false,
        ready: true,
      },
    ],
  };
}

Deno.test("resolveLobbySelectedGameId keeps existing selected game", () => {
  assertEquals(
    resolveLobbySelectedGameId(
      [makeGame("game-1"), makeGame("game-2")],
      "game-2",
    ),
    "game-2",
  );
});

Deno.test("resolveLobbySelectedGameId selects first game when none selected", () => {
  assertEquals(
    resolveLobbySelectedGameId(
      [makeGame("game-1"), makeGame("game-2")],
      null,
    ),
    "game-1",
  );
});

Deno.test("resolveLobbySelectedGameId selects first game when previous selection disappeared", () => {
  assertEquals(
    resolveLobbySelectedGameId(
      [makeGame("game-1"), makeGame("game-2")],
      "deleted-game",
    ),
    "game-1",
  );
});

Deno.test("resolveLobbySelectedGameId clears selection when no games remain", () => {
  assertEquals(resolveLobbySelectedGameId([], "game-1"), null);
});

Deno.test("selectedGameHasEmptyPlayer detects open players", () => {
  assertEquals(
    selectedGameHasEmptyPlayer([makeGame("game-1")], "game-1"),
    true,
  );
});

Deno.test("selectedGameHasEmptyPlayer is false when selected game is filled", () => {
  assertEquals(
    selectedGameHasEmptyPlayer([makeFilledGame("game-1")], "game-1"),
    false,
  );
});

Deno.test("selectedGameHasEmptyPlayer is false for missing game", () => {
  assertEquals(
    selectedGameHasEmptyPlayer([makeGame("game-1")], "missing-game"),
    false,
  );
});
