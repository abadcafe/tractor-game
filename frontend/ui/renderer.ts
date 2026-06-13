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
 *
 * @param snapshot - current game state snapshot
 * @param container - DOM element to render into
 * @param interactionMode - current interaction mode from GameLoop
 * @param ctx - optional render context (callbacks + selected card IDs)
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

  // Conditionally render bidding dialog for bid or stir interaction modes
  if (interactionMode === "bid" || interactionMode === "stir") {
    container.appendChild(
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

  // Conditionally render scoring overlay for COMPLETE phase (not GAME_OVER — that has its own overlay)
  if (snapshot.phase === "COMPLETE") {
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

  // Conditionally render game over overlay
  if (snapshot.phase === "GAME_OVER") {
    container.appendChild(
      renderGameOverOverlay(snapshot, ctx?.callbacks?.onNewGame),
    );
  }
}
