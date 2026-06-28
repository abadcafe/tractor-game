import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { gamePlayerHref, parseGamePlayerRoute } from "../routing.ts";

Deno.test("test_gamePlayerHref_builds_player_page_url", () => {
  assertEquals(
    gamePlayerHref("game-1", 0, "user-1"),
    "/game/game-1/player/0?user_id=user-1",
  );
});

Deno.test("test_parseGamePlayerRoute_valid_route", () => {
  assertEquals(
    parseGamePlayerRoute(
      "/game/game-1/player/3",
      "?user_id=user-3",
    ),
    {
      gameId: "game-1",
      playerIndex: 3,
      userId: "user-3",
    },
  );
});

Deno.test("test_parseGamePlayerRoute_rejects_missing_user", () => {
  assertEquals(parseGamePlayerRoute("/game/game-1/player/3", ""), null);
});

Deno.test("test_parseGamePlayerRoute_rejects_legacy_route", () => {
  assertEquals(parseGamePlayerRoute("/game/game-1", ""), null);
});
