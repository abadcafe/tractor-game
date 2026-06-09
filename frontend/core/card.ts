import type { Card } from "./types.ts";

/** Map of suit name to display symbol. */
const SUIT_SYMBOLS: Record<string, string> = {
  hearts: "♥",
  spades: "♠",
  diamonds: "♦",
  clubs: "♣",
  joker: "🃏",
};

/** Map of point card rank to point value. */
const POINT_VALUES: Record<string, number> = {
  "5": 5,
  "10": 10,
  K: 10,
};

/** Returns the Unicode symbol for a given suit name. */
export function suitSymbol(suit: string): string {
  return SUIT_SYMBOLS[suit] ?? suit;
}

/** Returns a display string for a card (symbol + rank, or joker label).
 *  Uses newline to stack suit above rank for vertical card layout.
 */
export function cardDisplay(c: Card): string {
  if (c.suit === "joker") {
    if (c.rank === "SJ") return "小\n王";
    if (c.rank === "BJ") return "大\n王";
  }
  return suitSymbol(c.suit) + "\n" + c.rank;
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

/** Returns true if the card is a trump card. */
export function isTrump(c: Card, trumpSuit: string | null, trumpRank: string): boolean {
  if (isJoker(c)) return true;
  if (isTrumpRank(c, trumpRank)) return true;
  if (trumpSuit !== null && c.suit === trumpSuit) return true;
  return false;
}

/** Returns true if the card is a point card (5, 10, or K). */
export function isPointCard(c: Card): boolean {
  return c.rank in POINT_VALUES;
}

/** Returns the point value of a card: 5->5, 10->10, K->10, others->0. */
export function cardPoints(c: Card): number {
  return POINT_VALUES[c.rank] ?? 0;
}
