import type { Card } from "../core/types.ts";

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

/** Compute bid priority matching server-side bid_value logic.
 *  Value = count * 100 + card_rank (higher = stronger bid).
 *  Returns 0 for invalid bids (single joker, mixed, etc.).
 */
export function computeBidPriority(cards: Card[], trumpRank: string): number {
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
