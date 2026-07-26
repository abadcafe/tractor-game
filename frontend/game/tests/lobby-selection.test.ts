import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import type { ListedGame } from "../net/rest-client.ts";
import {
  resolveLobbySelectedGameId,
  selectedGameHasEmptySeat,
} from "../lobby-selection.ts";

function makeGame(gameId: string): ListedGame {
  return {
    gameId,
    userCount: 0,
    capacity: 4,
    userSeats: [],
    seats: [
      {
        seat: "a",
        player: null,
        ready: false,
      },
      {
        seat: "b",
        player: null,
        ready: false,
      },
      {
        seat: "c",
        player: null,
        ready: false,
      },
      {
        seat: "d",
        player: null,
        ready: false,
      },
    ],
  };
}

function makeFilledGame(gameId: string): ListedGame {
  return {
    ...makeGame(gameId),
    seats: [
      {
        seat: "a",
        player: { kind: "bot", policy: "auto" },
        ready: true,
      },
      {
        seat: "b",
        player: {
          kind: "human",
          connected: false,
          mine: true,
        },
        ready: false,
      },
      {
        seat: "c",
        player: { kind: "bot", policy: "auto" },
        ready: true,
      },
      {
        seat: "d",
        player: { kind: "bot", policy: "auto" },
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

Deno.test("selectedGameHasEmptySeat detects open seats", () => {
  assertEquals(
    selectedGameHasEmptySeat([makeGame("game-1")], "game-1"),
    true,
  );
});

Deno.test("selectedGameHasEmptySeat is false when selected game is filled", () => {
  assertEquals(
    selectedGameHasEmptySeat([makeFilledGame("game-1")], "game-1"),
    false,
  );
});

Deno.test("selectedGameHasEmptySeat is false for missing game", () => {
  assertEquals(
    selectedGameHasEmptySeat([makeGame("game-1")], "missing-game"),
    false,
  );
});
