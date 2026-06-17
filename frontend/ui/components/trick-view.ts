import type { StateSnapshot } from "../../core/types.ts";
import { el } from "../dom.ts";
import { cardDisplay, suitSymbol } from "../../core/card.ts";
import { SEAT_MAP } from "../../config.ts";

/** Map player index to compass direction for trick grid positioning. */
const PLAYER_TO_DIRECTION: Record<number, string> = {
  0: "north",
  1: "west",
  2: "east",
  3: "south",
};

/**
 * Render the current trick area showing played cards with player labels,
 * plus trump/phase info bar.
 */
export function renderTrickView(snapshot: StateSnapshot): HTMLElement {
  const trickView = el("div", { class: "trick-view" });

  // Always show trump info
  const infoBar = el("div", { class: "info-bar" });

  // Trump display
  const trumpDiv = el("div", { class: "info-bar__trump" });
  trumpDiv.appendChild(el("span", {}, "主:"));

  if (snapshot.trump_suit) {
    const suitSpan = el("span", { class: `trump-suit suit-${snapshot.trump_suit}` });
    suitSpan.textContent = suitSymbol(snapshot.trump_suit);
    trumpDiv.appendChild(suitSpan);
  } else if (snapshot.phase === "DEAL_BID" || snapshot.phase === "STIRRING") {
    trumpDiv.appendChild(el("span", {}, "待定"));
  } else {
    trumpDiv.appendChild(el("span", {}, "无主"));
  }

  trumpDiv.appendChild(el("span", {}, `级牌 ${snapshot.trump_rank}`));
  infoBar.appendChild(trumpDiv);

  // Phase display
  const PHASE_LABELS: Record<string, string> = {
    DEAL_BID: "叫牌阶段",
    STIRRING: "反主阶段",
    PLAYING: "出牌阶段",
    WAITING: "结算中",
    GAME_OVER: "游戏结束",
  };
  const phaseLabel = PHASE_LABELS[snapshot.phase] ?? snapshot.phase;
  infoBar.appendChild(el("div", { class: "info-bar__phase" }, phaseLabel));
  trickView.appendChild(infoBar);

  // Render trick grid if there's a trick
  if (snapshot.trick) {
    const grid = el("div", { class: "trick-grid" });
    const slotsByPlayer = new Map(snapshot.trick.slots.map((slot) => [slot.player, slot]));

    for (const player of [0, 1, 2, 3]) {
      const direction = PLAYER_TO_DIRECTION[player] ?? "north";
      const isLead = player === snapshot.trick!.lead_player;
      const isCurrent = player === snapshot.trick!.current_player;
      const slot = slotsByPlayer.get(player);

      let slotClass = `${slot ? "trick-slot" : "trick-placeholder-slot"} trick-slot-${direction}`;
      if (isLead) slotClass += " lead";
      if (isCurrent && !isLead) slotClass += " current";
      if (!slot) slotClass += " empty";

      const slotEl = el("div", { class: slotClass });

      // Player label
      const seatInfo = SEAT_MAP[player];
      if (seatInfo && slot) {
        slotEl.appendChild(el("span", { class: "trick-player-label" }, seatInfo.label));
      } else if (seatInfo) {
        slotEl.appendChild(el("span", { class: "trick-placeholder-label" }, seatInfo.label));
      }

      // Cards
      const cardsDiv = el("div", { class: "trick-cards" });
      for (const card of slot?.cards ?? []) {
        cardsDiv.appendChild(el("span", { class: `trick-card suit-${card.suit}` }, cardDisplay(card)));
      }
      if (!slot) {
        cardsDiv.appendChild(el("span", { class: "trick-card-placeholder" }, "等待"));
      }
      slotEl.appendChild(cardsDiv);

      grid.appendChild(slotEl);
    }

    trickView.appendChild(grid);
  }

  return trickView;
}
