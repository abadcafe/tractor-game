/**
 * Card model for 升级 (Shengji/Tractor) card game.
 *
 * Uses 2 standard 54-card decks = 108 cards total:
 *   2 × (4 suits × 13 ranks + 2 jokers) = 104 suited + 4 jokers
 *
 * Point cards: 5 (5pts), 10 (10pts), K (10pts)
 * Total points in game: 200
 *
 * Natural rank order (non-trump context): 2 < 3 < ... < 10 < J < Q < K < A
 * Trump order: BigJoker > SmallJoker > trumpRank+trumpSuit > trumpRank+otherSuit > trumpSuit(A,K,...) > ...
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

// ---- Constants ----

/** Natural rank order for non-trump comparison. Higher = stronger. */
export const RANK_ORDER: Record<Rank, number> = {
  [Rank.TWO]: 2, [Rank.THREE]: 3, [Rank.FOUR]: 4, [Rank.FIVE]: 5,
  [Rank.SIX]: 6, [Rank.SEVEN]: 7, [Rank.EIGHT]: 8, [Rank.NINE]: 9,
  [Rank.TEN]: 10, [Rank.JACK]: 11, [Rank.QUEEN]: 12, [Rank.KING]: 13,
  [Rank.ACE]: 14,
  [Rank.SMALL_JOKER]: 15,
  [Rank.BIG_JOKER]: 16,
};

/** All suited ranks (excluding jokers). */
export const SUITED_RANKS: Rank[] = [
  Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE,
  Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE,
  Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE,
];

/** All four suits (excluding joker). */
export const SUITS: Suit[] = [Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS, Suit.CLUBS];

/** Points on each card rank. */
export const POINTS_MAP: Record<Rank, number> = {
  [Rank.TWO]: 0, [Rank.THREE]: 0, [Rank.FOUR]: 0, [Rank.FIVE]: 5,
  [Rank.SIX]: 0, [Rank.SEVEN]: 0, [Rank.EIGHT]: 0, [Rank.NINE]: 0,
  [Rank.TEN]: 10, [Rank.JACK]: 0, [Rank.QUEEN]: 0, [Rank.KING]: 10,
  [Rank.ACE]: 0, [Rank.SMALL_JOKER]: 0, [Rank.BIG_JOKER]: 0,
};

/** Total points in one deck: 5×4 + 10×4 + 10×4 = 100. Two decks = 200. */
export const TOTAL_POINTS = 200;

// ---- Card Interface ----

export interface Card {
  /** Unique ID: "D1-hearts-A" or "D1-joker-BJ" */
  readonly id: string;
  readonly suit: Suit;
  readonly rank: Rank;
  readonly isJoker: boolean;
  readonly isBigJoker: boolean;
  /** Points carried by this card (0, 5, or 10). */
  readonly points: number;
  /** Which physical deck (1 or 2). */
  readonly deck: 1 | 2;
}

// ---- Factory ----

function cardId(deck: number, suit: Suit, rank: Rank): string {
  return `D${deck}-${suit}-${rank}`;
}

function makeCard(suit: Suit, rank: Rank, deck: 1 | 2): Card {
  if (suit === Suit.JOKER) {
    return {
      id: cardId(deck, suit, rank),
      suit: Suit.JOKER,
      rank,
      isJoker: true,
      isBigJoker: rank === Rank.BIG_JOKER,
      points: 0,
      deck,
    };
  }
  return {
    id: cardId(deck, suit, rank),
    suit,
    rank,
    isJoker: false,
    isBigJoker: false,
    points: POINTS_MAP[rank],
    deck,
  };
}

/**
 * Create 2 full 54-card decks = 108 cards.
 */
export function createDecks(): Card[] {
  const cards: Card[] = [];

  for (const deck of [1, 2] as const) {
    // 52 suited cards per deck
    for (const suit of SUITS) {
      for (const rank of SUITED_RANKS) {
        cards.push(makeCard(suit, rank, deck));
      }
    }
    // 2 jokers per deck
    cards.push(makeCard(Suit.JOKER, Rank.SMALL_JOKER, deck));
    cards.push(makeCard(Suit.JOKER, Rank.BIG_JOKER, deck));
  }

  return cards;
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

/**
 * Natural numeric rank (2=2, ..., A=14, SJ=15, BJ=16).
 */
export function naturalRank(card: Card): number {
  return RANK_ORDER[card.rank];
}

/**
 * Check if two cards are a pair (same suit and rank).
 */
export function isPair(a: Card, b: Card): boolean {
  if (a.suit !== b.suit) return false;
  if (a.id === b.id) return false; // Same physical card
  return a.rank === b.rank;
}

/**
 * Group cards by suit.
 */
export function groupBySuit(cards: Card[]): Map<Suit, Card[]> {
  const groups = new Map<Suit, Card[]>();
  for (const c of cards) {
    const arr = groups.get(c.suit) ?? [];
    arr.push(c);
    groups.set(c.suit, arr);
  }
  return groups;
}

/**
 * Sort cards for display: grouped by suit, then by rank descending.
 * Suit order: Joker > Hearts > Spades > Diamonds > Clubs
 */
export function sortHand(cards: Card[]): Card[] {
  const suitOrder: Record<Suit, number> = {
    [Suit.JOKER]: 0, [Suit.HEARTS]: 1, [Suit.SPADES]: 2,
    [Suit.DIAMONDS]: 3, [Suit.CLUBS]: 4,
  };
  return [...cards].sort((a, b) => {
    const sDiff = suitOrder[a.suit] - suitOrder[b.suit];
    if (sDiff !== 0) return sDiff;
    return naturalRank(b) - naturalRank(a); // descending rank
  });
}
