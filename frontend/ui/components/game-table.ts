import type { StateSnapshot } from "../../core/types.ts";
import { el } from "../dom.ts";
import { SEAT_MAP } from "../../config.ts";
import { cardDisplay } from "../../core/card.ts";

/**
 * Render the game table with four player areas and trick view in center.
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
    const trickView = el("div", { class: "trick-view" });

    // Info bar: lead player + current player
    const leadSeat = SEAT_MAP[snapshot.trick.lead_player];
    const curSeat = SEAT_MAP[snapshot.trick.current_player];
    const infoBar = el(
      "div",
      { class: "trick-info-bar" },
      `领出: ${leadSeat.label} | 当前: ${curSeat.label}`,
    );
    trickView.appendChild(infoBar);

    // Grid of four slots
    const grid = el("div", { class: "trick-grid" });

    for (let i = 0; i < 4; i++) {
      const slot = snapshot.trick.slots[i];
      const seat = SEAT_MAP[i];

      const slotClasses = ["trick-slot"];
      if (i === snapshot.trick.lead_player) slotClasses.push("lead");
      if (i === snapshot.trick.current_player) slotClasses.push("current");

      const positionClass =
        seat.position === "北"
          ? "trick-slot-north"
          : seat.position === "西"
            ? "trick-slot-west"
            : seat.position === "东"
              ? "trick-slot-east"
              : "trick-slot-south";
      slotClasses.push(positionClass);

      const slotEl = el("div", { class: slotClasses.join(" ") });

      // Player label above cards
      slotEl.appendChild(el("span", { class: "trick-player-label" }, seat.label));

      // Cards container
      const cardsEl = el("div", { class: "trick-cards" });
      for (const card of slot.cards) {
        cardsEl.appendChild(
          el("span", { class: `trick-card suit-${card.suit}` }, cardDisplay(card)),
        );
      }
      slotEl.appendChild(cardsEl);

      grid.appendChild(slotEl);
    }

    trickView.appendChild(grid);
    table.appendChild(trickView);
  }

  return table;
}
