import type { Card } from "../core/types.ts";
import type { ClientAction } from "../core/protocol.ts";
import type { BidOption } from "./types.ts";
import { suitSymbol } from "../core/card.ts";

/** Suit rank values matching server-side bid_value ordering. */
const SUIT_RANK: Record<string, number> = {
  diamonds: 0,
  clubs: 1,
  hearts: 2,
  spades: 3,
};

/** Joker rank values matching server-side bid_value ordering. */
const JOKER_RANK: Record<string, number> = {
  SJ: 4,
  BJ: 5,
};

export interface DealBidActionDecision {
  action: ClientAction;
  matchedPending: boolean;
  stalePending: boolean;
}

/** Compute bid priority matching server-side bid_value logic.
 *  Value = count * 100 + card_rank (higher = stronger bid).
 *  Returns 0 for invalid bids (single joker, mixed, non-trump-rank, etc.).
 */
export function computeBidPriority(
  cards: Card[],
  trumpRank: string,
): number {
  if (cards.length === 0) return 0;

  // All cards must be trump rank or jokers
  const allValid = cards.every(
    (c) => c.rank === trumpRank || c.rank === "SJ" || c.rank === "BJ",
  );
  if (!allValid) return 0;

  const count = cards.length;

  if (
    count === 1 && (cards[0].rank === "SJ" || cards[0].rank === "BJ")
  ) {
    return 0;
  }

  // Pair of jokers
  if (
    count === 2 &&
    cards.every((c) => c.rank === "SJ" || c.rank === "BJ")
  ) {
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

/** Convert complete backend action hints into bid options for display. */
export function computeBidOptionsFromHints(
  hints: Card[][],
  trumpRank: string,
): BidOption[] {
  return hints
    .map((cards) => {
      const priority = computeBidPriority(cards, trumpRank);
      if (priority <= 0) return null;
      return {
        cardIds: cards.map((c) => c.id),
        label: formatBidOptionLabel(cards, trumpRank),
        trumpSuit: cards[0]?.suit === "joker"
          ? null
          : cards[0]?.suit ?? null,
        priority,
      } satisfies BidOption;
    })
    .filter((option): option is BidOption => option !== null);
}

export function computeDealBidAction(
  _hints: Card[][],
  pendingBidIntent: BidOption | null,
  seq: number,
): DealBidActionDecision {
  if (pendingBidIntent !== null) {
    return {
      action: { type: "bid", seq, cards: pendingBidIntent.cardIds },
      matchedPending: true,
      stalePending: false,
    };
  }

  return {
    action: { type: "bid", seq, pass: true },
    matchedPending: false,
    stalePending: false,
  };
}

function formatBidOptionLabel(
  cards: Card[],
  trumpRank: string,
): string {
  if (cards.length === 2 && cards.every((c) => c.rank === "BJ")) {
    return "大王对";
  }
  if (cards.length === 2 && cards.every((c) => c.rank === "SJ")) {
    return "小王对";
  }
  const first = cards[0];
  if (!first) return "";
  if (first.suit === "joker") {
    return first.rank === "BJ" ? "大王" : "小王";
  }
  const suitSymbols = suitSymbol(first.suit).repeat(cards.length);
  return `${suitSymbols}${trumpRank}`;
}
