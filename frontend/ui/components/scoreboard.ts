import type { StateSnapshot } from "../../core/types.ts";
import { el } from "../dom.ts";

/**
 * Render a scoreboard sidebar showing team levels and defender points.
 */
export function renderScoreboard(snapshot: StateSnapshot): HTMLElement {
  const scoreboard = el("div", { class: "scoreboard" },
    el("div", { class: "scoreboard__team0" }, `队伍0 当前级: ${snapshot.team0_level}`),
    el("div", { class: "scoreboard__team1" }, `队伍1 当前级: ${snapshot.team1_level}`),
    el("div", { class: "scoreboard__defender-points" }, `防守方得分: ${snapshot.defender_points}`),
  );
  return scoreboard;
}
