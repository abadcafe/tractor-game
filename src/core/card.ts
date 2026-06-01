/**
 * Card type definitions for the thin TS client.
 *
 * These types match the JSON shape returned by the Python backend.
 * All game logic (creation, sorting, comparison) lives server-side.
 */

// ---- Enums ----

export enum Suit {
  HEARTS = 'hearts',
  SPADES = 'spades',
  DIAMONDS = 'diamonds',
  CLUBS = 'clubs',
  JOKER = 'joker',
}

export enum Rank {
  TWO = '2', THREE = '3', FOUR = '4', FIVE = '5',
  SIX = '6', SEVEN = '7', EIGHT = '8', NINE = '9',
  TEN = '10', JACK = 'J', QUEEN = 'Q', KING = 'K',
  ACE = 'A',
  SMALL_JOKER = 'SJ',
  BIG_JOKER = 'BJ',
}

// ---- Card Interface ----

export interface Card {
  readonly id: string;
  readonly suit: Suit;
  readonly rank: Rank;
  readonly isJoker: boolean;
  readonly isBigJoker: boolean;
  readonly points: number;
  readonly deck: 1 | 2;
}

// ---- Helpers ----

/**
 * Get a human-readable display string for a card.
 * e.g. "♥A", "♠10", "🃏大", "🃏小"
 */
export function cardDisplay(card: Card): string {
  if (card.isJoker) {
    return card.isBigJoker ? '🃏大' : '🃏小';
  }
  const suitSymbol: Record<Suit, string> = {
    [Suit.HEARTS]: '♥',
    [Suit.SPADES]: '♠',
    [Suit.DIAMONDS]: '♦',
    [Suit.CLUBS]: '♣',
    [Suit.JOKER]: '🃏',
  };
  const rankDisplay: Record<Rank, string> = {
    [Rank.TWO]: '2', [Rank.THREE]: '3', [Rank.FOUR]: '4', [Rank.FIVE]: '5',
    [Rank.SIX]: '6', [Rank.SEVEN]: '7', [Rank.EIGHT]: '8', [Rank.NINE]: '9',
    [Rank.TEN]: '10', [Rank.JACK]: 'J', [Rank.QUEEN]: 'Q', [Rank.KING]: 'K',
    [Rank.ACE]: 'A', [Rank.SMALL_JOKER]: '小', [Rank.BIG_JOKER]: '大',
  };
  return `${suitSymbol[card.suit]}${rankDisplay[card.rank]}`;
}
