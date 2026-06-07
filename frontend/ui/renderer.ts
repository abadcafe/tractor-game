import type { StateSnapshot, InteractionMode, ActionCallbacks } from "../core/types.ts";
import { renderGameTable } from "./components/game-table.ts";
import { renderHandView } from "./components/hand-view.ts";
import { renderTrickView } from "./components/trick-view.ts";
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
 * @param callbacks - optional action callbacks for user interactions
 * @param selectedCardIds - optional set of currently selected card IDs
 */
export function render(
  snapshot: StateSnapshot,
  container: Element,
  interactionMode: InteractionMode,
  callbacks?: ActionCallbacks,
  selectedCardIds?: Set<string>,
): void {
  // Clear the container
  container.innerHTML = "";

  // Always render: game table, hand view, trick view, scoreboard
  container.appendChild(renderGameTable(snapshot));

  container.appendChild(
    renderHandView(
      snapshot,
      interactionMode,
      selectedCardIds,
      callbacks?.onCardClick,
      callbacks?.onAction,
    ),
  );

  container.appendChild(renderTrickView(snapshot));
  container.appendChild(renderScoreboard(snapshot));

  // Conditionally render bidding dialog for bid or stir interaction modes
  if (interactionMode === "bid" || interactionMode === "stir") {
    container.appendChild(
      renderBiddingDialog(
        snapshot,
        interactionMode,
        callbacks?.onBid,
        callbacks?.onStir,
        callbacks?.onPass,
      ),
    );
  }

  // Conditionally render scoring overlay for next_round or COMPLETE phase
  if (interactionMode === "next_round" || snapshot.phase === "COMPLETE") {
    container.appendChild(
      renderScoringOverlay(
        snapshot,
        interactionMode,
        callbacks?.onAction
          ? () => callbacks.onAction("next_round")
          : undefined,
      ),
    );
  }

  // Conditionally render game over overlay
  if (snapshot.phase === "GAME_OVER") {
    container.appendChild(
      renderGameOverOverlay(snapshot, callbacks?.onNewGame),
    );
  }
}
