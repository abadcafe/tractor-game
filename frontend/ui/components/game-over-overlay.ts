import type { StateSnapshot } from "../../core/types.ts";
import { el } from "../dom.ts";
import { HUMAN_TEAM, TEAM_LABELS } from "../../config.ts";

/**
 * Render a game-over overlay showing the winning team, final levels, and a "新游戏" button.
 */
export function renderGameOverOverlay(
  snapshot: StateSnapshot,
  onNewGame?: () => void,
): HTMLElement {
  const overlay = el("div", { class: "game-over-overlay" });

  const humanWon = snapshot.winning_team === HUMAN_TEAM;

  const winnerText = humanWon
    ? "🏆 我们赢了！"
    : snapshot.winning_team !== null
    ? `${TEAM_LABELS[snapshot.winning_team] ?? "队伍" + snapshot.winning_team}获胜`
    : "游戏结束";

  overlay.appendChild(
    el("div", { class: "winner-text" }, winnerText),
  );

  // Final levels
  overlay.appendChild(
    el("div", { class: "game-over-overlay__levels" },
      `${TEAM_LABELS[0]}: ${snapshot.team0_level}    ${TEAM_LABELS[1]}: ${snapshot.team1_level}`),
  );

  if (onNewGame) {
    const button = el("button", { class: "btn-primary game-over-overlay__new-game" }, "新游戏");
    button.addEventListener("click", () => onNewGame());
    overlay.appendChild(button);
  }

  return overlay;
}
