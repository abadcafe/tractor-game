import type { StateSnapshot } from "../../core/types.ts";
import { el } from "../dom.ts";

/**
 * Render a scoreboard sidebar showing team levels and defender points.
 */
export function renderScoreboard(snapshot: StateSnapshot): HTMLElement {
  const scoreboard = el("div", { class: "scoreboard" },
    el("div", { class: "scoreboard__team0" }, `Team 0 Level: ${snapshot.team0_level}`),
    el("div", { class: "scoreboard__team1" }, `Team 1 Level: ${snapshot.team1_level}`),
    el("div", { class: "scoreboard__defender-points" }, `Defender Points: ${snapshot.defender_points}`),
  );
  return scoreboard;
}
