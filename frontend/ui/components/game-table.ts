import type { StateSnapshot } from "../../core/types.ts";
import { el } from "../dom.ts";
import { SEAT_MAP } from "../../config.ts";

/**
 * Render the game table with four player areas.
 * Each area shows: player label, card count, declarer marker, current player highlight.
 */
export function renderGameTable(snapshot: StateSnapshot): HTMLElement {
  const table = el("div", { class: "game-table" });

  for (let i = 0; i < 4; i++) {
    const seat = SEAT_MAP[i];
    const attrs: Record<string, string> = {
      class: "player-area",
      "data-position": seat.position,
    };

    if (snapshot.current_player === i) {
      attrs.class += " current";
    }

    const area = el("div", attrs);

    // Player label
    const label = el("span", { class: "player-label" }, seat.label);
    area.appendChild(label);

    // Card count placeholder (AI players show card backs / count)
    const countText = i === 3
      ? `${snapshot.player_hand.length} 张`
      : snapshot.player_hand.length > 0
        ? ""
        : "0 张";
    if (i !== 3) {
      // AI players: just show count placeholder
      area.appendChild(el("span", { class: "card-count" }, `${i === 3 ? snapshot.player_hand.length : 0} 张`));
    } else {
      // Human player: show hand count
      area.appendChild(el("span", { class: "card-count" }, `${snapshot.player_hand.length} 张`));
    }

    // Declarer marker
    if (snapshot.declarer_player === i) {
      const marker = el("span", { class: "declarer-marker" }, "庄");
      area.appendChild(marker);
    }

    table.appendChild(area);
  }

  return table;
}
