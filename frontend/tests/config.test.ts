import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  API_BASE,
  GAME_PLAYER_PATH,
  playerIndexFromNumber,
  WS_PATH,
} from "../config.ts";

Deno.test("test_ws_path_format", () => {
  assertEquals(
    WS_PATH("abc123", 2, "user-1"),
    "/game/abc123/player/2?user_id=user-1",
  );
});

Deno.test("test_game_player_path_escapes_identity", () => {
  assertEquals(
    GAME_PLAYER_PATH("game/id", 3, "user id"),
    "/game/game%2Fid/player/3?user_id=user%20id",
  );
});

Deno.test("test_player_index_from_number", () => {
  assertEquals(playerIndexFromNumber(0), 0);
  assertEquals(playerIndexFromNumber(3), 3);
  assertEquals(playerIndexFromNumber(4), null);
});

Deno.test("test_api_base", () => {
  assertEquals(API_BASE, "/api/game");
});
