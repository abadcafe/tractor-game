import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { SEAT_MAP, WS_PATH, API_BASE } from "../config.ts";

Deno.test("test_ws_path_format", () => {
  assertEquals(WS_PATH("abc123"), "/game/abc123");
});

Deno.test("test_ws_path_different_id", () => {
  assertEquals(WS_PATH("xyz789"), "/game/xyz789");
});

Deno.test("test_seat_map_has_four_entries", () => {
  assertEquals(Object.keys(SEAT_MAP).length, 4);
});

Deno.test("test_seat_map_south_is_human", () => {
  assertEquals(SEAT_MAP[3].label, "你");
  assertEquals(SEAT_MAP[3].position, "南");
  assertEquals(SEAT_MAP[3].team, 0);
});

Deno.test("test_seat_map_north_is_teammate", () => {
  assertEquals(SEAT_MAP[0].label, "同伴");
  assertEquals(SEAT_MAP[0].position, "北");
  assertEquals(SEAT_MAP[0].team, 0);
});

Deno.test("test_seat_map_west_is_opponent_a", () => {
  assertEquals(SEAT_MAP[1].label, "左家");
  assertEquals(SEAT_MAP[1].position, "西");
  assertEquals(SEAT_MAP[1].team, 1);
});

Deno.test("test_seat_map_east_is_opponent_b", () => {
  assertEquals(SEAT_MAP[2].label, "右家");
  assertEquals(SEAT_MAP[2].position, "东");
  assertEquals(SEAT_MAP[2].team, 1);
});

Deno.test("test_api_base", () => {
  assertEquals(API_BASE, "/api/game");
});
