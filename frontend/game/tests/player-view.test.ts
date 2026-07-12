import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  playerView,
  teamLabelForViewer,
  viewerTeam,
} from "../ui/player-view.ts";

Deno.test("test_playerView_places_viewer_at_south", () => {
  const player = playerView(1, 1);

  assertEquals(player.label, "玩家 1 / 你");
  assertEquals(player.direction, "south");
  assertEquals(player.position, "南");
  assertEquals(player.teamLabel, "我方");
});

Deno.test("test_playerView_rotates_other_players_around_viewer", () => {
  assertEquals(playerView(2, 1).direction, "east");
  assertEquals(playerView(3, 1).direction, "north");
  assertEquals(playerView(0, 1).direction, "west");
});

Deno.test("test_teamLabelForViewer_is_relative_to_viewer_team", () => {
  assertEquals(viewerTeam(3), 1);
  assertEquals(teamLabelForViewer(1, 3), "我方");
  assertEquals(teamLabelForViewer(0, 3), "对方");
});
