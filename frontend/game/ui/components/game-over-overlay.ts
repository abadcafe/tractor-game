import type { StateSnapshot } from "../../core/types.ts";
import { el } from "../dom.ts";
import type { PlayerIndex } from "../../config.ts";
import { teamLabelForViewer, viewerTeam } from "../player-view.ts";

/**
 * Render a game-over overlay showing the winning team, final levels, and a "新游戏" button.
 */
export function renderGameOverOverlay(
  snapshot: StateSnapshot,
  viewerPlayer?: PlayerIndex | null,
  onNewGame?: () => void,
): HTMLElement {
  const overlay = el("div", { class: "game-over-overlay" });

  const viewerWon = snapshot.winning_team === viewerTeam(viewerPlayer);

  const winnerText = viewerWon
    ? "我们赢了！"
    : snapshot.winning_team !== null
    ? `${teamLabelForViewer(snapshot.winning_team, viewerPlayer)}获胜`
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
        teamLabelForViewer(0, viewerPlayer)
      }: ${snapshot.team0_level}` +
        `    ${
          teamLabelForViewer(1, viewerPlayer)
        }: ${snapshot.team1_level}`,
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
