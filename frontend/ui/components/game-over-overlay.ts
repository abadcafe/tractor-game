import type { StateSnapshot } from "../../core/types.ts";
import { el } from "../dom.ts";

/**
 * Render a game-over overlay showing the winning team, final levels, and a "新游戏" button.
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

  const humanTeam = 0; // Player 3 (South) is always Team 0
  const humanWon = snapshot.winning_team === humanTeam;

  const winnerText = humanWon
    ? "🏆 我们赢了!"
    : snapshot.winning_team === 0
    ? "队伍0获胜!"
    : snapshot.winning_team === 1
    ? "队伍1获胜!"
    : "游戏结束";

  overlay.appendChild(
    el("div", { class: "winner-text" }, winnerText),
  );

  overlay.appendChild(
    el("div", { class: "game-over-overlay__levels" },
      `最终等级 — 队伍0: ${snapshot.team0_level}  队伍1: ${snapshot.team1_level}`),
  );

  if (onNewGame) {
    const button = el("button", { class: "game-over-overlay__new-game" }, "新游戏");
    button.addEventListener("click", () => onNewGame());
    overlay.appendChild(button);
  }

  return overlay;
}
