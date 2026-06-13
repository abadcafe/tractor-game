import type { StateSnapshot, BidEvent } from "../../core/types.ts";
import type { InteractionMode, BidButtonState } from "../../engine/types.ts";
import { suitSymbol, suitDisplayName } from "../../core/card.ts";

/**
 * Render the bidding/stirring dialog.
 *
 * - DEAL_BID + "bid" interaction: shows "叫牌中" title and "叫牌" button
 * - STIRRING + "stir" interaction: shows "反主中" title with "反主" and "不反" buttons
 * - null interaction: shows dialog but no action buttons (spectator / not human turn)
 *
 * Always shows current bid events as `.bid-event` items.
 *
 * @param snapshot - current game state
 * @param interactionMode - "bid", "stir", or null
 * @param onBid - callback when bid button clicked, receives trump rank card IDs
 * @param onStir - callback when stir button clicked, receives trump rank card IDs
 * @param onPass - callback when pass button clicked
 * @param selectedCardIds - set of currently selected card IDs
 * @param bidButtonState - pre-computed button state (disabled + title) from engine layer
 * @param stirButtonState - pre-computed button state (disabled + title) from engine layer
 */
export function renderBiddingDialog(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
  onBid?: (cardIds: string[]) => void,
  onStir?: (cardIds: string[]) => void,
  onPass?: () => void,
  selectedCardIds?: Set<string>,
  bidButtonState?: BidButtonState,
  stirButtonState?: BidButtonState,
): HTMLElement {
  const container = document.createElement("div");
  container.classList.add("bidding-dialog");

  // Get user-selected card IDs
  const selectedIds = selectedCardIds ? [...selectedCardIds] : [];

  if (snapshot.phase === "DEAL_BID") {
    // Title
    const title = document.createElement("div");
    title.classList.add("bidding-dialog-title");
    title.textContent = "叫牌中";
    container.appendChild(title);

    // Hint
    const hint = document.createElement("div");
    hint.classList.add("bidding-dialog-hint");
    hint.textContent = "点击手牌选择叫牌牌张";
    container.appendChild(hint);

    // Bid button (only when interactionMode is "bid")
    if (interactionMode === "bid") {
      const bidButton = document.createElement("button");
      bidButton.textContent = "叫牌";

      // Use pre-computed button state from engine layer
      if (bidButtonState) {
        bidButton.disabled = bidButtonState.disabled;
        if (bidButtonState.title) bidButton.title = bidButtonState.title;
      }

      bidButton.addEventListener("click", () => {
        if (onBid && !bidButton.disabled && selectedIds.length > 0) {
          onBid(selectedIds);
        }
      });
      container.appendChild(bidButton);
    }
  } else if (snapshot.phase === "STIRRING") {
    // Title
    const title = document.createElement("div");
    title.classList.add("bidding-dialog-title");
    title.textContent = "反主中";
    container.appendChild(title);

    // Hint
    const hint = document.createElement("div");
    hint.classList.add("bidding-dialog-hint");
    hint.textContent = "点击手牌选择对子反主";
    container.appendChild(hint);

    // Action buttons (only when interactionMode is "stir")
    if (interactionMode === "stir") {
      // Stir button
      const stirButton = document.createElement("button");
      stirButton.textContent = "反主";

      // Use pre-computed button state from engine layer
      if (stirButtonState) {
        stirButton.disabled = stirButtonState.disabled;
        if (stirButtonState.title) stirButton.title = stirButtonState.title;
      }

      stirButton.addEventListener("click", () => {
        if (onStir && !stirButton.disabled && selectedIds.length > 0) {
          onStir(selectedIds);
        }
      });
      container.appendChild(stirButton);

      // Pass button
      const passButton = document.createElement("button");
      passButton.textContent = "不反";
      passButton.addEventListener("click", () => {
        if (onPass) {
          onPass();
        }
      });
      container.appendChild(passButton);
    }
  }

  // Render bid events
  if (snapshot.bid_events.length > 0) {
    const eventsContainer = document.createElement("div");
    eventsContainer.classList.add("bid-events");
    for (const event of snapshot.bid_events) {
      const eventEl = document.createElement("div");
      eventEl.classList.add("bid-event");
      eventEl.textContent = formatBidEvent(event);
      eventsContainer.appendChild(eventEl);
    }
    container.appendChild(eventsContainer);
  }

  return container;
}

/** Compact display for a card in bid events (horizontal, no newline). */
function _compactCard(c: { suit: string; rank: string }): string {
  if (c.suit === "joker") {
    return c.rank === "BJ" ? "大王" : "小王";
  }
  return suitSymbol(c.suit) + c.rank;
}

/** Format a bid event for display. */
function formatBidEvent(event: BidEvent): string {
  const cardsStr = event.cards.map(_compactCard).join(" ");
  if (event.kind === "trump_rank" && event.suit) {
    return `玩家${event.player}: ${cardsStr} (${suitDisplayName(event.suit)}主)`;
  }
  if (event.kind === "joker" && event.joker_type) {
    return `玩家${event.player}: ${cardsStr} (${event.joker_type === "big" ? "大" : "小"}王)`;
  }
  return `玩家${event.player}: ${cardsStr}`;
}
