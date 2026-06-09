import type { StateSnapshot } from "../../core/types.ts";
import { el } from "../dom.ts";
import { cardDisplay } from "../../core/card.ts";
import { SEAT_MAP } from "../../config.ts";

/**
 * Render the current trick area showing played cards with player labels.
 * This is a display-only component with no interaction.
 *
 * @param snapshot - current game state snapshot
 * @returns HTMLElement containing the trick view
 */
export function renderTrickView(snapshot: StateSnapshot): HTMLElement {
  const trickView = el("div", { class: "trick-view" });

  if (!snapshot.trick) {
    return trickView;
  }

  for (const slot of snapshot.trick.slots) {
    const isCurrentPlayer = slot.player === snapshot.trick!.current_player;
    const slotEl = el("div", { class: isCurrentPlayer ? "trick-slot current" : "trick-slot" });

    // Add player label from SEAT_MAP
    const seatInfo = SEAT_MAP[slot.player];
    if (seatInfo) {
      slotEl.appendChild(el("span", { class: "player-label" }, seatInfo.label));
    }

    // Add each card in this slot
    for (const card of slot.cards) {
      slotEl.appendChild(el("span", { class: `trick-card suit-${card.suit}` }, cardDisplay(card)));
    }

    trickView.appendChild(slotEl);
  }

  return trickView;
}
