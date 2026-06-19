import type { StateSnapshot } from "../../core/types.ts";
import type { BidOption, InteractionMode } from "../../engine/types.ts";
import { suitDisplayName } from "../../core/card.ts";
import { SEAT_MAP } from "../../config.ts";

/**
 * Render the bidding info panel for DEAL_BID phase.
 *
 * - Shows available bid options as clickable pills
 * - Highlights the currently selected pending intent
 * - No action buttons — auto-bid handles everything
 *
 * For STIRRING phase, action buttons are rendered above the hand. This panel
 * only keeps passive waiting information when it is not the human player's turn.
 */
export function renderBiddingDialog(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
  _onBid?: (cardIds: string[]) => void,
  onStir?: (cardIds: string[]) => void,
  onPass?: () => void,
  _selectedCardIds?: Set<string>,
  _bidButtonState?: unknown,
  stirButtonState?: { disabled: boolean; title?: string },
  bidOptions?: BidOption[],
  pendingBidIntent?: BidOption | null,
  onBidOptionSelect?: (option: BidOption) => void,
): HTMLElement {
  if (snapshot.phase === "DEAL_BID") {
    return renderBidOptionsPanel(
      snapshot,
      bidOptions ?? [],
      pendingBidIntent ?? null,
      onBidOptionSelect,
    );
  }

  if (snapshot.phase === "STIRRING") {
    return renderStirDialog(
      snapshot,
      interactionMode,
    );
  }

  // Shouldn't be called for other phases
  return document.createElement("div");
}

/** Render the bid options panel for DEAL_BID phase. */
function renderBidOptionsPanel(
  snapshot: StateSnapshot,
  bidOptions: BidOption[],
  pendingBidIntent: BidOption | null,
  onBidOptionSelect?: (option: BidOption) => void,
): HTMLElement {
  const container = document.createElement("div");
  container.classList.add("bidding-dialog");

  // Title
  const title = document.createElement("div");
  title.classList.add("bidding-dialog-title");
  title.textContent = "叫牌";
  container.appendChild(title);

  // Bid options
  if (bidOptions.length > 0) {
    const optionsWrap = document.createElement("div");
    optionsWrap.classList.add("bid-options");

    for (const option of bidOptions) {
      const pill = document.createElement("span");
      pill.classList.add("bid-option");

      // Highlight if this is the pending intent
      if (
        pendingBidIntent &&
        pendingBidIntent.cardIds.join(",") === option.cardIds.join(",")
      ) {
        pill.classList.add("selected");
      }

      // Show what trump suit this would create
      const trumpLabel = option.trumpSuit
        ? ` (${suitDisplayName(option.trumpSuit)}主)`
        : "";
      pill.textContent = option.label + trumpLabel;

      if (onBidOptionSelect) {
        pill.addEventListener("click", () => onBidOptionSelect(option));
      }

      optionsWrap.appendChild(pill);
    }

    container.appendChild(optionsWrap);
  } else {
    const noOptions = document.createElement("div");
    noOptions.classList.add("bidding-dialog-hint");
    noOptions.textContent = "当前无可用叫牌选项";
    container.appendChild(noOptions);
  }

  // Pending intent indicator
  if (pendingBidIntent) {
    const intentHint = document.createElement("div");
    intentHint.classList.add("bid-intent-hint");
    const trumpLabel = pendingBidIntent.trumpSuit
      ? ` (${suitDisplayName(pendingBidIntent.trumpSuit)}主)`
      : "";
    intentHint.textContent =
      `待叫: ${pendingBidIntent.label}${trumpLabel}`;
    container.appendChild(intentHint);
  }

  return container;
}

/** Render the stirring dialog for STIRRING phase. */
function renderStirDialog(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
): HTMLElement {
  const container = document.createElement("div");

  if (interactionMode === "stir") {
    return container;
  } else {
    // Not our turn — passive info bar
    container.classList.add("bid-info-bar");

    const hint = document.createElement("div");
    hint.classList.add("waiting-hint");
    const seat = snapshot.stirring_state
      ? SEAT_MAP[snapshot.stirring_state.current_player]
      : null;
    hint.textContent = seat
      ? `等待${seat.label}反主...`
      : "等待反主...";
    container.appendChild(hint);
  }

  return container;
}
