import type { StateSnapshot } from "../../core/types.ts";
import { el } from "../dom.ts";

/**
 * Render a game-over overlay showing the winning team and a "新游戏" button.
 *
 * @param snapshot - Current game state snapshot (should be in GAME_OVER phase)
 * @param onNewGame - Optional callback invoked when the "新游戏" button is clicked
 * @returns An HTMLElement containing the game-over overlay
 */
export function renderGameOverOverlay(
  snapshot: StateSnapshot,
  onNewGame?: () => void,
): HTMLElement {
  const overlay = el("div", { class: "game-over-overlay" });

  const winnerText = snapshot.winning_team === 0
    ? "队伍0获胜!"
    : snapshot.winning_team === 1
    ? "队伍1获胜!"
    : "游戏结束";

  overlay.appendChild(
    el("div", { class: "winner-text" }, winnerText),
  );

  if (onNewGame) {
    const button = el("button", { class: "game-over-overlay__new-game" }, "新游戏");
    button.addEventListener("click", () => onNewGame());
    overlay.appendChild(button);
  }

  return overlay;
}
