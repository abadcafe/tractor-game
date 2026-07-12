import type { Card, Rank, Suit } from "../core/types.ts";
import type { ClientAction } from "../core/protocol.ts";
import type { BidOption } from "./types.ts";
import { suitSymbol } from "../core/card.ts";

type JokerRank = Extract<Rank, "SJ" | "BJ">;
type NonJokerSuit = Exclude<Suit, "joker">;

/** Suit rank values matching server-side bid_value ordering. */
const SUIT_RANK: Record<NonJokerSuit, number> = {
  diamonds: 0,
  clubs: 1,
  hearts: 2,
  spades: 3,
};

/** Joker rank values matching server-side bid_value ordering. */
const JOKER_RANK: Record<JokerRank, number> = {
  SJ: 4,
  BJ: 5,
};

function isJokerRank(rank: Rank): rank is JokerRank {
  return rank === "SJ" || rank === "BJ";
}

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
  trumpRank: Rank,
): number {
  if (cards.length === 0) return 0;

  // All cards must be trump rank or jokers
  const allValid = cards.every(
    (c) => c.rank === trumpRank || isJokerRank(c.rank),
  );
  if (!allValid) return 0;

  const count = cards.length;

  if (
    count === 1 && isJokerRank(cards[0].rank)
  ) {
    return 0;
  }

  // Pair of jokers
  if (
    count === 2 &&
    cards.every((c) => isJokerRank(c.rank))
  ) {
    if (cards[0].rank !== cards[1].rank) return 0;
    const rank = cards[0].rank;
    if (!isJokerRank(rank)) return 0;
    return count * 100 + JOKER_RANK[rank];
  }

  // All must be same suit (no mixed joker + rank)
  if (cards.some((c) => isJokerRank(c.rank))) return 0;
  const suits = new Set(cards.map((c) => c.suit));
  if (suits.size !== 1) return 0;

  const suit = cards[0].suit;
  if (suit === "joker") return 0;
  return count * 100 + SUIT_RANK[suit];
}

/** Convert complete backend action hints into bid options for display. */
export function computeBidOptionsFromHints(
  hints: Card[][],
  trumpRank: Rank,
): BidOption[] {
  const options: BidOption[] = [];
  for (const cards of hints) {
    const first = cards[0];
    if (!first) continue;
    const priority = computeBidPriority(cards, trumpRank);
    if (priority <= 0) continue;
    options.push({
      cardIds: cards.map((c) => c.id),
      label: formatBidOptionLabel(cards, trumpRank),
      trumpSuit: first.suit === "joker" ? null : first.suit,
      priority,
    });
  }
  return options;
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
  trumpRank: Rank,
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
