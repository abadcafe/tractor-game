import type { Card, BidEvent } from "../core/types.ts";
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

/** Compute all available bid options from the player's hand.
 *  Returns options sorted by priority (highest first).
 *  Only includes options that would beat the current bid_winner. */
export function computeBidOptions(
  hand: Card[],
  trumpRank: string,
  bidWinner: BidEvent | null,
): BidOption[] {
  const options: BidOption[] = [];
  const winnerPriority = bidWinner
    ? computeBidPriority(bidWinner.cards, trumpRank)
    : 0;

  // Group trump-rank cards by suit, collect jokers
  const suitGroups: Record<string, Card[]> = {};
  const sjCards: Card[] = [];
  const bjCards: Card[] = [];

  for (const card of hand) {
    if (card.rank === "SJ") {
      sjCards.push(card);
    } else if (card.rank === "BJ") {
      bjCards.push(card);
    } else if (card.rank === trumpRank) {
      const suit = card.suit;
      if (!suitGroups[suit]) suitGroups[suit] = [];
      suitGroups[suit].push(card);
    }
  }

  // Big joker pair
  if (bjCards.length >= 2) {
    const cards = bjCards.slice(0, 2);
    const priority = computeBidPriority(cards, trumpRank);
    if (priority > winnerPriority) {
      options.push({
        cardIds: cards.map((c) => c.id),
        label: "大王对",
        trumpSuit: null,
        priority,
      });
    }
  }

  // Small joker pair
  if (sjCards.length >= 2) {
    const cards = sjCards.slice(0, 2);
    const priority = computeBidPriority(cards, trumpRank);
    if (priority > winnerPriority) {
      options.push({
        cardIds: cards.map((c) => c.id),
        label: "小王对",
        trumpSuit: null,
        priority,
      });
    }
  }

  // Trump-rank pairs by suit (highest suit first)
  const sortedSuits = Object.entries(suitGroups).sort(
    (a, b) => (SUIT_RANK[b[0]] ?? 0) - (SUIT_RANK[a[0]] ?? 0),
  );
  for (const [suit, cards] of sortedSuits) {
    if (cards.length >= 2) {
      const bidCards = cards.slice(0, 2);
      const priority = computeBidPriority(bidCards, trumpRank);
      if (priority > winnerPriority) {
        options.push({
          cardIds: bidCards.map((c) => c.id),
          label: `${suitSymbol(suit)}${trumpRank}对`,
          trumpSuit: suit,
          priority,
        });
      }
    }
  }

  // Single trump-rank cards by suit (highest suit first)
  for (const [suit, cards] of sortedSuits) {
    const bidCards = [cards[0]];
    const priority = computeBidPriority(bidCards, trumpRank);
    if (priority > winnerPriority) {
      options.push({
        cardIds: bidCards.map((c) => c.id),
        label: `${suitSymbol(suit)}${trumpRank}`,
        trumpSuit: suit,
        priority,
      });
    }
  }

  return options;
}
