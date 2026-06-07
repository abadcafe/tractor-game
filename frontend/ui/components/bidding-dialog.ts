import type { StateSnapshot, InteractionMode, BidEvent } from "../../core/types.ts";
import { cardDisplay, isTrumpRank } from "../../core/card.ts";

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
 */
export function renderBiddingDialog(
  snapshot: StateSnapshot,
  interactionMode: InteractionMode,
  onBid?: (cardIds: string[]) => void,
  onStir?: (cardIds: string[]) => void,
  onPass?: () => void,
): HTMLElement {
  const container = document.createElement("div");
  container.classList.add("bidding-dialog");

  if (snapshot.phase === "DEAL_BID") {
    // Title
    const title = document.createElement("div");
    title.classList.add("bidding-dialog-title");
    title.textContent = "叫牌中";
    container.appendChild(title);

    // Bid button (only when interactionMode is "bid")
    if (interactionMode === "bid") {
      const bidButton = document.createElement("button");
      bidButton.textContent = "叫牌";
      bidButton.addEventListener("click", () => {
        if (onBid) {
          // Collect all trump rank cards in the player's hand
          const trumpRankCardIds = snapshot.player_hand
            .filter((c) => isTrumpRank(c, snapshot.trump_rank))
            .map((c) => c.id);
          onBid(trumpRankCardIds);
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

    // Action buttons (only when interactionMode is "stir")
    if (interactionMode === "stir") {
      // Stir button
      const stirButton = document.createElement("button");
      stirButton.textContent = "反主";
      stirButton.addEventListener("click", () => {
        if (onStir) {
          // Collect all trump rank cards in the player's hand
          const trumpRankCardIds = snapshot.player_hand
            .filter((c) => isTrumpRank(c, snapshot.trump_rank))
            .map((c) => c.id);
          onStir(trumpRankCardIds);
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

/** Format a bid event for display. */
function formatBidEvent(event: BidEvent): string {
  const cardsStr = event.cards.map(cardDisplay).join("");
  if (event.kind === "trump_rank" && event.suit) {
    return `Player ${event.player}: ${cardsStr} (suit: ${event.suit})`;
  }
  if (event.kind === "joker" && event.joker_type) {
    return `Player ${event.player}: ${cardsStr} (${event.joker_type} joker)`;
  }
  return `Player ${event.player}: ${cardsStr}`;
}
