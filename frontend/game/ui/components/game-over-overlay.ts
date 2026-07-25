import type { StateSnapshot } from "../../core/types.ts";
import { el } from "../dom.ts";
import type { SeatId } from "../../config.ts";
import {
  partnershipLabelForViewer,
  viewerPartnership,
} from "../seat-view.ts";

/**
 * Render a game-over overlay showing the winning team, final levels, and a "新游戏" button.
 */
export function renderGameOverOverlay(
  snapshot: StateSnapshot,
  viewerSeat: SeatId,
  onNewGame?: () => void,
): HTMLElement {
  const overlay = el("div", { class: "game-over-overlay" });

  const viewerWon = snapshot.winning_partnership ===
    viewerPartnership(viewerSeat);

  const winnerText = viewerWon
    ? "我们赢了！"
    : snapshot.winning_partnership !== null
    ? `${
      partnershipLabelForViewer(
        snapshot.winning_partnership,
        viewerSeat,
      )
    }获胜`
    : "游戏结束";

  overlay.appendChild(
    el("div", { class: "winner-text" }, winnerText),
  );

  // Final levels
  overlay.appendChild(
    el(
      "div",
      { class: "game-over-overlay__levels" },
      `${
        partnershipLabelForViewer("first", viewerSeat)
      }: ${snapshot.partnership_levels.first}` +
        `    ${
          partnershipLabelForViewer("second", viewerSeat)
        }: ${snapshot.partnership_levels.second}`,
    ),
  );

  if (onNewGame) {
    const button = el("button", {
      class: "btn-primary game-over-overlay__new-game",
    }, "新游戏");
    button.addEventListener("click", () => onNewGame());
    overlay.appendChild(button);
  }

  return overlay;
}
