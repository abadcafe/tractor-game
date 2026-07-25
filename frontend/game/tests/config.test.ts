import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  API_BASE,
  GAME_SEAT_PATH,
  seatIdFromString,
  WS_PATH,
} from "../config.ts";

Deno.test("test_ws_path_format", () => {
  assertEquals(
    WS_PATH("abc123", "c", "user-1"),
    "/game/abc123/seat/c?user_id=user-1",
  );
});

Deno.test("test_game_seat_path_escapes_identity", () => {
  assertEquals(
    GAME_SEAT_PATH("game/id", "d", "user id"),
    "/game/game%2Fid/seat/d?user_id=user%20id",
  );
});

Deno.test("test_seat_id_from_string_is_strict", () => {
  assertEquals(seatIdFromString("a"), "a");
  assertEquals(seatIdFromString("d"), "d");
  assertEquals(seatIdFromString("0"), null);
  assertEquals(seatIdFromString("A"), null);
});

Deno.test("test_api_base", () => {
  assertEquals(API_BASE, "/api/game");
});
