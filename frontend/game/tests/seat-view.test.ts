import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  partnershipLabelForViewer,
  seatView,
  viewerPartnership,
} from "../ui/seat-view.ts";

Deno.test("test_seatView_places_viewer_at_bottom", () => {
  const seat = seatView("b", "b");

  assertEquals(seat.label, "座位 b / 你");
  assertEquals(seat.slot, "bottom");
  assertEquals(seat.position, "本人");
  assertEquals(seat.partnershipLabel, "我方");
});

Deno.test("test_seatView_projects_topology_relative_to_viewer", () => {
  assertEquals(seatView("c", "b").slot, "right");
  assertEquals(seatView("d", "b").slot, "top");
  assertEquals(seatView("a", "b").slot, "left");
});

Deno.test("test_partnershipLabelForViewer_is_relative_to_viewer", () => {
  assertEquals(viewerPartnership("d"), "second");
  assertEquals(partnershipLabelForViewer("second", "d"), "我方");
  assertEquals(partnershipLabelForViewer("first", "d"), "对方");
});
