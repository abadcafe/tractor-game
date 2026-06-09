import type { StateSnapshot, InteractionMode, Card } from "../../core/types.ts";
import { cardDisplay, isTrump } from "../../core/card.ts";

/**
 * Render the human player's hand with card display, click selection,
 * legal action highlighting, and action buttons.
 *
 * @param snapshot - current game state snapshot
 * @param interactionMode - "play", "discard", or null (spectator)
 * @param selectedCardIds - set of currently selected card IDs (managed by parent)
 * @param onCardClick - callback when a card is clicked
 * @param onAction - callback when an action button is clicked
 */
export function renderHandView(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
  selectedCardIds?: Set<string>,
  onCardClick?: (cardId: string) => void,
  onAction?: (action: string) => void,
): HTMLElement {
  const handView = document.createElement("div");
  handView.classList.add("hand-view");

  // Build a set of legal card IDs for highlighting
  const legalCardIds = new Set<string>();
  for (const cards of snapshot.legal_actions) {
    for (const card of cards) {
      legalCardIds.add(card.id);
    }
  }

  // Sort cards: trump first, then by suit and rank
  const sortedHand = [...snapshot.player_hand].sort((a, b) => {
    const aTrump = isTrump(a, snapshot.trump_suit, snapshot.trump_rank);
    const bTrump = isTrump(b, snapshot.trump_suit, snapshot.trump_rank);
    if (aTrump && !bTrump) return -1;
    if (!aTrump && bTrump) return 1;
    const suitCompare = a.suit.localeCompare(b.suit);
    if (suitCompare !== 0) return suitCompare;
    return a.rank.localeCompare(b.rank);
  });

  // Render each card
  for (const card of sortedHand) {
    const cardSpan = document.createElement("span");
    cardSpan.classList.add("card", `suit-${card.suit}`);
    cardSpan.textContent = cardDisplay(card);

    if (legalCardIds.has(card.id)) {
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
  // null interactionMode: no buttons (spectator)

  return handView;
}
