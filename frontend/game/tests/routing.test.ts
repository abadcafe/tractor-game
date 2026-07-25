import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { gameSeatHref, parseGameSeatRoute } from "../routing.ts";

Deno.test("test_gameSeatHref_builds_seat_page_url", () => {
  assertEquals(
    gameSeatHref("game-1", "a", "user-1"),
    "/game/game-1/seat/a?user_id=user-1",
  );
});

Deno.test("test_parseGameSeatRoute_valid_route", () => {
  assertEquals(
    parseGameSeatRoute(
      "/game/game-1/seat/d",
      "?user_id=user-3",
    ),
    {
      gameId: "game-1",
      seatId: "d",
      userId: "user-3",
    },
  );
});

Deno.test("test_parseGameSeatRoute_rejects_missing_user", () => {
  assertEquals(parseGameSeatRoute("/game/game-1/seat/d", ""), null);
});
