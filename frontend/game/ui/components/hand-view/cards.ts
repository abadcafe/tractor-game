import { isJoker, isTrumpRank } from "../../../core/card.ts";
import type { Card, Rank, Suit } from "../../../core/types.ts";

export function handCardClass(
  card: Card,
  trumpSuit: Suit | null,
  trumpRank: Rank,
): string {
  let className = `card suit-${card.suit}`;
  if (isPointCard(card)) className += " point-card";
  if (isTrumpCard(card, trumpSuit, trumpRank)) {
    className += " trump-card";
  }
  return className;
}

/** Check if a card is a trump card for visual highlighting. */
export function isTrumpCard(
  c: Card,
  trumpSuit: Suit | null,
  trumpRank: Rank,
): boolean {
  if (isJoker(c)) return true;
  if (isTrumpRank(c, trumpRank)) return true;
  if (trumpSuit !== null && c.suit === trumpSuit) return true;
  return false;
}

export function isPointCard(card: Card): boolean {
  return card.rank === "5" || card.rank === "10" || card.rank === "K";
}
