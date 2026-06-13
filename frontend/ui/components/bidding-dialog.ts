import type { StateSnapshot, InteractionMode, BidEvent, Card } from "../../core/types.ts";
import { cardDisplay, suitSymbol, isTrumpRank, isJoker } from "../../core/card.ts";

const SUIT_PRIORITY = ["spades", "hearts", "clubs", "diamonds"];

/** Suit rank values matching server-side bid_value ordering. */
const SUIT_RANK: Record<string, number> = {
  diamonds: 0,
  clubs: 1,
  hearts: 2,
  spades: 3,
};

const JOKER_RANK: Record<string, number> = {
  SJ: 4,
  BJ: 5,
};

/** Compute bid priority matching server-side bid_value logic.
 *  Value = count * 100 + card_rank (higher = stronger bid).
 *  Returns 0 for invalid bids (single joker, mixed, etc.).
 */
function computeBidPriority(cards: Card[], trumpRank: string): number {
  if (cards.length === 0) return 0;

  // All cards must be trump rank or jokers
  const allValid = cards.every(
    (c) => c.rank === trumpRank || c.rank === "SJ" || c.rank === "BJ",
  );
  if (!allValid) return 0;

  const count = cards.length;

  // Single joker is invalid
  if (count === 1 && (cards[0].rank === "SJ" || cards[0].rank === "BJ")) {
    return 0;
  }

  // Pair of jokers
  if (count === 2 && cards.every((c) => c.rank === "SJ" || c.rank === "BJ")) {
    if (cards[0].rank !== cards[1].rank) return 0;
    return count * 100 + (JOKER_RANK[cards[0].rank] ?? 0);
  }

  // All must be same suit (no mixed joker + rank)
  if (cards.some((c) => c.rank === "SJ" || c.rank === "BJ")) return 0;
  const suits = new Set(cards.map((c) => c.suit));
  if (suits.size !== 1) return 0;

  const suit = cards[0].suit;
  return count * 100 + (SUIT_RANK[suit] ?? 0);
}

const SUIT_CN: Record<string, string> = {
  spades: "黑桃",
  hearts: "红桃",
  clubs: "梅花",
  diamonds: "方块",
};

/** Pick the strongest valid bid (single or pair) from the hand.
 *  Returns null if no valid bid exists.
 */
function selectBidCards(hand: StateSnapshot["player_hand"], trumpRank: string): string[] | null {
  const jokers = hand.filter(isJoker);
  const bigJokers = jokers.filter((c) => c.rank === "BJ");
  const smallJokers = jokers.filter((c) => c.rank === "SJ");

  // Joker pair is the strongest bid
  if (bigJokers.length >= 2) return bigJokers.slice(0, 2).map((c) => c.id);
  if (smallJokers.length >= 2) return smallJokers.slice(0, 2).map((c) => c.id);

  const trumpCards = hand.filter((c) => isTrumpRank(c, trumpRank));
  const bySuit: Record<string, typeof trumpCards> = {};
  for (const c of trumpCards) {
    bySuit[c.suit] = bySuit[c.suit] || [];
    bySuit[c.suit].push(c);
  }

  // Prefer a pair over a single
  for (const suit of SUIT_PRIORITY) {
    if (bySuit[suit]?.length >= 2) {
      return bySuit[suit].slice(0, 2).map((c) => c.id);
    }
  }

  // Fallback to a single card
  for (const suit of SUIT_PRIORITY) {
    if (bySuit[suit]?.length >= 1) {
      return [bySuit[suit][0].id];
    }
  }

  return null;
}

/** Pick the strongest valid stir (must be a pair) from the hand.
 *  Returns null if no pair exists.
 */
function selectStirCards(hand: StateSnapshot["player_hand"], trumpRank: string): string[] | null {
  const jokers = hand.filter(isJoker);
  const bigJokers = jokers.filter((c) => c.rank === "BJ");
  const smallJokers = jokers.filter((c) => c.rank === "SJ");

  if (bigJokers.length >= 2) return bigJokers.slice(0, 2).map((c) => c.id);
  if (smallJokers.length >= 2) return smallJokers.slice(0, 2).map((c) => c.id);

  const trumpCards = hand.filter((c) => isTrumpRank(c, trumpRank));
  const bySuit: Record<string, typeof trumpCards> = {};
  for (const c of trumpCards) {
    bySuit[c.suit] = bySuit[c.suit] || [];
    bySuit[c.suit].push(c);
  }

  for (const suit of SUIT_PRIORITY) {
    if (bySuit[suit]?.length >= 2) {
      return bySuit[suit].slice(0, 2).map((c) => c.id);
    }
  }

  return null;
}

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
  selectedCardIds?: Set<string>,
): HTMLElement {
  const container = document.createElement("div");
  container.classList.add("bidding-dialog");

  // Get user-selected cards (from hand click)
  const selectedIds = selectedCardIds ? [...selectedCardIds] : [];
  const selectedCards = selectedIds
    .map((id) => snapshot.player_hand.find((c) => c.id === id))
    .filter((c): c is Card => c !== undefined);

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

      // Validate user selection
      if (selectedCards.length === 0) {
        bidButton.disabled = true;
        bidButton.title = "请先选择要叫的牌";
      } else {
        const priority = computeBidPriority(selectedCards, snapshot.trump_rank);
        if (priority === 0) {
          bidButton.disabled = true;
          bidButton.title = "选择的牌不构成有效叫牌";
        } else if (snapshot.bid_winner) {
          const winnerPriority = computeBidPriority(
            snapshot.bid_winner.cards,
            snapshot.trump_rank,
          );
          if (priority <= winnerPriority) {
            bidButton.disabled = true;
            bidButton.title = "优先级不足，无法超过当前叫牌";
          }
        }
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

      // Validate user selection: must be a valid pair
      if (selectedCards.length === 0) {
        stirButton.disabled = true;
        stirButton.title = "请先选择要反主的对子";
      } else if (selectedCards.length !== 2) {
        stirButton.disabled = true;
        stirButton.title = "反主必须选择2张牌";
      } else {
        const priority = computeBidPriority(selectedCards, snapshot.trump_rank);
        if (priority < 200) {
          // priority < 200 means not a pair (pair = count*100 + suit_rank, count=2 so >= 200)
          stirButton.disabled = true;
          stirButton.title = "反主必须用对子";
        }
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
    return `玩家${event.player}: ${cardsStr} (${SUIT_CN[event.suit] ?? event.suit}主)`;
  }
  if (event.kind === "joker" && event.joker_type) {
    return `玩家${event.player}: ${cardsStr} (${event.joker_type === "big" ? "大" : "小"}王)`;
  }
  return `玩家${event.player}: ${cardsStr}`;
}
