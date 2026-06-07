import type { Card } from "./types.ts";

/** Map of suit name to display symbol. */
const SUIT_SYMBOLS: Record<string, string> = {
  hearts: "♥",
  spades: "♠",
  diamonds: "♦",
  clubs: "♣",
  joker: "🃏",
};

/** Returns the Unicode symbol for a given suit name. */
export function suitSymbol(suit: string): string {
  return SUIT_SYMBOLS[suit] ?? suit;
}

/** Returns a display string for a card (symbol + rank, or joker label). */
export function cardDisplay(c: Card): string {
  if (c.suit === "joker") {
    return c.rank === "SJ" ? "🃏小王" : "🃏大王";
  }
  return suitSymbol(c.suit) + c.rank;
}

/** Returns true if the card is a joker (small or big). */
export function isJoker(c: Card): boolean {
  return c.rank === "SJ" || c.rank === "BJ";
}

/** Returns true only if the card is the big joker. */
export function isBigJoker(c: Card): boolean {
  return c.rank === "BJ";
}

/** Returns true if the card matches the given trump rank (jokers never match). */
export function isTrumpRank(c: Card, rank: string): boolean {
  return !isJoker(c) && c.rank === rank;
}

/** Returns true if the card is a point card (5, 10, or K). */
export function isPointCard(c: Card): boolean {
  return c.rank === "5" || c.rank === "10" || c.rank === "K";
}

/** Returns the point value of a card: 5->5, 10->10, K->10, others->0. */
export function cardPoints(c: Card): number {
  if (c.rank === "5") return 5;
  if (c.rank === "10") return 10;
  if (c.rank === "K") return 10;
  return 0;
}
