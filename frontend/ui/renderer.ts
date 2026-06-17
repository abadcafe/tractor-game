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
 */
export function render(
  snapshot: StateSnapshot,
  container: Element,
  interactionMode: InteractionMode,
  ctx?: RenderContext,
): void {
  // Clear the container
  container.innerHTML = "";

  // Always render: game table (includes trick view), hand view, scoreboard
  container.appendChild(renderGameTable(snapshot));

  container.appendChild(
    renderHandView(
      snapshot,
      interactionMode,
      ctx?.selectedCardIds,
      ctx?.legalCardIds,
      ctx?.callbacks?.onCardClick,
      ctx?.callbacks?.onAction,
    ),
  );

  container.appendChild(renderScoreboard(snapshot));

  // Bidding panel: always show during DEAL_BID or STIRRING
  const isBiddingPhase = snapshot.phase === "DEAL_BID" || snapshot.phase === "STIRRING";
  if (isBiddingPhase) {
    container.appendChild(
      renderBiddingDialog(
        snapshot,
        interactionMode,
        undefined, // onBid — no longer used
        ctx?.callbacks?.onStir,
        ctx?.callbacks?.onPass,
        ctx?.selectedCardIds,
        undefined, // bidButtonState — no longer used
        ctx?.stirButtonState,
        ctx?.bidOptions,
        ctx?.pendingBidIntent,
        ctx?.callbacks?.onBidOptionSelect,
      ),
    );
  }

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
