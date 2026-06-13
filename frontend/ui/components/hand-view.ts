import type { StateSnapshot } from "../../core/types.ts";
import type { InteractionMode, GameAction } from "../../engine/types.ts";
import { cardDisplay, sortHand } from "../../core/card.ts";

/**
 * Render the human player's hand with card display, click selection,
 * legal action highlighting, and action buttons.
 *
 * @param snapshot - current game state snapshot
 * @param interactionMode - "play", "discard", or null (spectator)
 * @param selectedCardIds - set of currently selected card IDs (managed by parent)
 * @param legalCardIds - pre-computed set of legal card IDs for highlighting
 * @param onCardClick - callback when a card is clicked
 * @param onAction - callback when an action button is clicked
 */
export function renderHandView(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
  selectedCardIds?: Set<string>,
  legalCardIds?: Set<string>,
  onCardClick?: (cardId: string) => void,
  onAction?: (action: GameAction) => void,
): HTMLElement {
  const handView = document.createElement("div");
  handView.classList.add("hand-view");

  // Sort hand per spec
  const sortedHand = sortHand(
    snapshot.player_hand,
    snapshot.trump_suit,
    snapshot.trump_rank,
  );

  // Render each card
  for (const card of sortedHand) {
    const cardSpan = document.createElement("span");
    cardSpan.classList.add("card", `suit-${card.suit}`);
    cardSpan.textContent = cardDisplay(card);

    if (legalCardIds?.has(card.id)) {
      cardSpan.classList.add("legal");
    }

    if (selectedCardIds?.has(card.id)) {
      cardSpan.classList.add("selected");
    }

    if (onCardClick) {
      cardSpan.addEventListener("click", () => onCardClick(card.id));
    }

    handView.appendChild(cardSpan);
  }

  // Render action buttons based on interaction mode
  if (interactionMode === "play") {
    const button = document.createElement("button");
    button.textContent = "出牌";
    if (onAction) {
      button.addEventListener("click", () => onAction("play"));
    }
    handView.appendChild(button);
  } else if (interactionMode === "discard") {
    const button = document.createElement("button");
    button.textContent = "弃牌";
    if (onAction) {
      button.addEventListener("click", () => onAction("discard"));
    }
    handView.appendChild(button);
  }
  // bid/stir: cards are clickable in hand view; action buttons are in bidding-dialog
  // null interactionMode: no buttons (spectator)

  return handView;
}
