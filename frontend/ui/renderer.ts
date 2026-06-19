import type { StateSnapshot } from "../core/types.ts";
import type { InteractionMode } from "../engine/types.ts";
import type { RenderContext } from "./types.ts";
import { renderGameTable } from "./components/game-table.ts";
import { renderHandView } from "./components/hand-view.ts";
import { renderScoreboard } from "./components/scoreboard.ts";
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
  container.classList.add("game-shell");
  container.classList.toggle(
    "game-shell--scoring",
    snapshot.phase === "WAITING",
  );

  // Always render: game table (includes trick view), hand view, scoreboard
  container.appendChild(
    renderGameTable(
      snapshot,
      ctx?.previousTrickPreview,
      ctx?.failedThrowPreview,
    ),
  );

  container.appendChild(
    renderHandView(
      snapshot,
      interactionMode,
      ctx?.selectedCardIds,
      ctx?.legalCardIds,
      ctx?.callbacks?.onCardClick,
      ctx?.callbacks?.onAction,
      ctx?.callbacks?.onClearSelection,
      ctx?.callbacks?.onUseHint,
      ctx?.callbacks?.onToggleHandCompact,
      ctx?.compactHand,
      ctx?.callbacks?.onStir,
      ctx?.callbacks?.onPass,
      ctx?.stirButtonState,
      ctx?.callbacks?.onShowPreviousTrick,
      ctx?.bidOptions,
      ctx?.pendingBidIntent,
      ctx?.callbacks?.onBidOptionSelect,
    ),
  );

  container.appendChild(renderScoreboard(snapshot));

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
