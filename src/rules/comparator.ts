/**
 * Trump-aware card comparison for 升级.
 *
 * Trump ordering (highest → lowest):
 *   1. Big Joker (大王)
 *   2. Small Joker (小王)
 *   3. Trump rank + Trump suit  (主牌)
 *   4. Trump rank + Other suits (副级牌), tie-broken by suit convention
 *   5. Trump suit other ranks: A > K > Q > J > 10 > ... (excluding trump rank)
 *   6. Non-trump suit other ranks: A > K > ... (excluding trump rank)
 *
 * Non-trump suit ordering:
 *   A > K > Q > J > 10 > 9 > ... > 2, EXCLUDING the trump rank
 *   (trump rank cards of non-trump suits are always trump)
 */

import { Card, Suit, Rank, RANK_ORDER } from '../core/card';

/**
 * Assign an effective ordering number to a card in the trump context.
 * Higher number = stronger card.
 *
 * Range allocation:
 *   100-109: Big Joker
 *   90-99:  Small Joker
 *   80-89:  Trump rank + trump suit
 *   70-79:  Trump rank + other suits
 *   60-69:  Trump suit cards (non-trump-rank)
 *   0-59:   Non-trump suit cards
 */
export function trumpOrder(card: Card, trumpSuit: Suit, trumpRank: Rank): number {
  // 1. Big Joker
  if (card.isJoker && card.isBigJoker) {
    return 100;
  }

  // 2. Small Joker
  if (card.isJoker && !card.isBigJoker) {
    return 90;
  }

  // 3. Trump rank + Trump suit (主牌)
  if (card.rank === trumpRank && card.suit === trumpSuit) {
    return 80;
  }

  // 4. Trump rank + Other suits (副级牌)
  if (card.rank === trumpRank && card.suit !== trumpSuit) {
    // All 副级牌 are equal rank, but we differentiate by suit for tractor detection
    // Use a fixed suit order: Hearts > Spades > Diamonds > Clubs
    const suitOffset: Record<Suit, number> = {
      [Suit.HEARTS]: 3, [Suit.SPADES]: 2,
      [Suit.DIAMONDS]: 1, [Suit.CLUBS]: 0,
      [Suit.JOKER]: -1,
    };
    return 70 + suitOffset[card.suit];
  }

  // 5. Trump suit cards (non-trump-rank)
  if (card.suit === trumpSuit && card.rank !== trumpRank) {
    // Natural rank order, but trump rank is excluded
    const rank = naturalRankExcludingTrump(card.rank, trumpRank);
    return 60 + rank;
  }

  // 6. Non-trump suit cards
  const rank = naturalRankExcludingTrump(card.rank, trumpRank);
  return rank; // 2-14 range
}

/**
 * Non-trump suit ordering: natural ranks, but the trump rank is excluded.
 * The trump rank cards are always trump.
 */
export function nonTrumpOrder(card: Card, trumpRank: Rank): number {
  if (card.rank === trumpRank) {
    // Trump rank cards are removed from non-trump suit ordering
    return -1;
  }
  return RANK_ORDER[card.rank];
}

/**
 * Natural rank but with trump rank excluded (it's "pulled out" of the suit).
 */
function naturalRankExcludingTrump(rank: Rank, trumpRank: Rank): number {
  if (rank === trumpRank) return -1; // shouldn't happen for suited non-trump-rank
  return RANK_ORDER[rank];
}

/**
 * Compare two cards in trump context.
 * Returns: negative if a < b, 0 if equal, positive if a > b.
 */
export function compareCards(a: Card, b: Card, trumpSuit: Suit, trumpRank: Rank): number {
  const orderA = trumpOrder(a, trumpSuit, trumpRank);
  const orderB = trumpOrder(b, trumpSuit, trumpRank);
  return orderA - orderB;
}

/**
 * Check if two cards are effectively equal in trump context
 * (used for pair matching).
 */
export function isEqualInTrump(a: Card, b: Card, trumpSuit: Suit, trumpRank: Rank): boolean {
  return trumpOrder(a, trumpSuit, trumpRank) === trumpOrder(b, trumpSuit, trumpRank);
}

/**
 * Get the effective suit of a card in trump context.
 * All trump cards belong to a virtual "trump" group.
 * Non-trump cards keep their suit.
 */
export function effectiveSuit(card: Card, trumpSuit: Suit, trumpRank: Rank): Suit | 'trump' {
  if (card.isJoker) return 'trump';
  if (card.rank === trumpRank) return 'trump';
  if (card.suit === trumpSuit) return 'trump';
  return card.suit;
}

/**
 * Compare two play actions to determine which wins the trick.
 * a and b must be of the same PlayType.
 */
export function comparePlays(
  a: Card[], b: Card[],
  trumpSuit: Suit, trumpRank: Rank,
  leadSuit: Suit | null,
): number {
  if (a.length === 0 || b.length === 0) return 0;
  if (a.length !== b.length) {
    // Different pattern lengths — should not happen in normal play
    return a.length - b.length;
  }

  const aIsTrump = isTrumpPlay(a, trumpSuit, trumpRank);
  const bIsTrump = isTrumpPlay(b, trumpSuit, trumpRank);

  // Trump beats non-trump
  if (aIsTrump && !bIsTrump) return 1;
  if (!aIsTrump && bIsTrump) return -1;

  // Both trump: compare by highest card in each play
  if (aIsTrump && bIsTrump) {
    const bestA = Math.max(...a.map(c => trumpOrder(c, trumpSuit, trumpRank)));
    const bestB = Math.max(...b.map(c => trumpOrder(c, trumpSuit, trumpRank)));
    return bestA - bestB;
  }

  // Both non-trump, same suit: compare by highest card
  if (a[0].suit === b[0].suit) {
    const bestA = Math.max(...a.map(c => RANK_ORDER[c.rank]));
    const bestB = Math.max(...b.map(c => RANK_ORDER[c.rank]));
    return bestA - bestB;
  }

  // Both non-trump, different suits: the one matching lead suit wins
  if (leadSuit && a[0].suit === leadSuit) return 1;
  if (leadSuit && b[0].suit === leadSuit) return -1;

  return 0; // both are off-suit discards
}

/**
 * Check if a set of cards is a trump play.
 */
function isTrumpPlay(cards: Card[], trumpSuit: Suit, trumpRank: Rank): boolean {
  return cards.some(c =>
    c.isJoker || c.rank === trumpRank || c.suit === trumpSuit
  );
}

/**
 * Sort cards in descending trump order.
 */
export function sortByTrumpOrder(cards: Card[], trumpSuit: Suit, trumpRank: Rank): Card[] {
  return [...cards].sort((a, b) =>
    trumpOrder(b, trumpSuit, trumpRank) - trumpOrder(a, trumpSuit, trumpRank)
  );
}
