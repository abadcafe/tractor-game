import type { StateSnapshot } from "../core/types.ts";
import type { InteractionMode } from "../engine/types.ts";
import type { RenderContext } from "./types.ts";
import { renderGameTable } from "./components/game-table.ts";
import { renderHandView } from "./components/hand-view.ts";
import { renderScoreboard } from "./components/scoreboard.ts";
import { renderBiddingDialog } from "./components/bidding-dialog.ts";
import { renderScoringOverlay } from "./components/scoring-overlay.ts";
import { renderGameOverOverlay } from "./components/game-over-overlay.ts";

/**
 * Orchestrate rendering of all UI components from a state snapshot.
 *
 * Clears the container and appends components based on the current phase
 * and interaction mode.
 */
export function render(
  snapshot: StateSnapshot,
  container: Element,
  interactionMode: InteractionMode,
  ctx?: RenderContext,
): void {
  container.innerHTML = "";

  const shell = document.createElement("div");
  shell.className = "game-shell";

  const playRegion = document.createElement("main");
  playRegion.className = "play-region";

  const tableRegion = document.createElement("section");
  tableRegion.className = "table-region";
  tableRegion.appendChild(renderGameTable(snapshot));
  playRegion.appendChild(tableRegion);

  const tableControls = document.createElement("section");
  tableControls.className = "table-controls";

  const isBiddingPhase = snapshot.phase === "DEAL_BID" ||
    snapshot.phase === "STIRRING";
  if (isBiddingPhase) {
    tableControls.appendChild(
      renderBiddingDialog(
        snapshot,
        interactionMode,
        ctx?.callbacks?.onBid,
        ctx?.callbacks?.onStir,
        ctx?.callbacks?.onPass,
        ctx?.selectedCardIds,
        ctx?.bidButtonState,
        ctx?.stirButtonState,
      ),
    );
  }

  tableControls.appendChild(
    renderHandView(
      snapshot,
      interactionMode,
      ctx?.selectedCardIds,
      ctx?.legalCardIds,
      ctx?.callbacks?.onCardClick,
      ctx?.callbacks?.onAction,
    ),
  );
  playRegion.appendChild(tableControls);
  shell.appendChild(playRegion);

  const sidebar = document.createElement("aside");
  sidebar.className = "side-panel";
  sidebar.appendChild(renderScoreboard(snapshot));
  shell.appendChild(sidebar);

  container.appendChild(shell);

  // Scoring overlay for WAITING phase
  if (snapshot.phase === "WAITING") {
    container.appendChild(
      renderScoringOverlay(
        snapshot,
        interactionMode,
        interactionMode === "next_round" && ctx?.callbacks?.onAction
          ? () => ctx!.callbacks!.onAction("next_round")
          : undefined,
        ctx?.levelChange,
      ),
    );
  }

  // Game over overlay
  if (snapshot.phase === "GAME_OVER") {
    container.appendChild(
      renderGameOverOverlay(snapshot, ctx?.callbacks?.onNewGame),
    );
  }
}
