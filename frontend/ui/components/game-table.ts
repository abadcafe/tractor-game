import type { StateSnapshot } from "../../core/types.ts";
import { el } from "../dom.ts";
import { SEAT_MAP } from "../../config.ts";
import { renderTrickView } from "./trick-view.ts";

/**
 * Determine which player is currently active based on awaiting_action
 * and phase-specific state (replaces removed top-level current_player).
 */
function getCurrentPlayer(snapshot: StateSnapshot): number | null {
  if (snapshot.awaiting_action === "play" && snapshot.trick) {
    return snapshot.trick.current_player;
  }
  if (snapshot.awaiting_action === "stir" && snapshot.stirring_state) {
    return snapshot.stirring_state.current_player;
  }
  if (snapshot.awaiting_action === "discard" && snapshot.stirring_state) {
    return snapshot.stirring_state.exchanging_player;
  }
  // For bid/next_round, multiple players may be active; no single current player
  return null;
}

/**
 * Render the game table with four player areas and trick view in center.
 * Each area shows: player label, card count, declarer marker, current player highlight.
 */
export function renderGameTable(snapshot: StateSnapshot): HTMLElement {
  const table = el("div", { class: "game-table" });
  const currentPlayer = getCurrentPlayer(snapshot);

  for (let i = 0; i < 4; i++) {
    const seat = SEAT_MAP[i];
    const attrs: Record<string, string> = {
      class: "player-area",
      "data-position": seat.position,
    };

    if (currentPlayer === i) {
      attrs.class += " current";
    }

    const area = el("div", attrs);

    // Player label
    const label = el("span", { class: "player-label" }, seat.label);
    area.appendChild(label);

    // Card count
    const count = snapshot.player_hand_counts?.[i] ?? 0;
    area.appendChild(el("span", { class: "card-count" }, `${count} 张`));

    // Declarer marker
    if (snapshot.declarer_player === i) {
      const marker = el("span", { class: "declarer-marker" }, "庄");
      area.appendChild(marker);
    }

    table.appendChild(area);
  }

  // Render trick view in center area
  if (snapshot.trick) {
    table.appendChild(renderTrickView(snapshot));
  }

  return table;
}
