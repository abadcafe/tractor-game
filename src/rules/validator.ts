/**
 * Play validation: enumerate all legal plays from a hand given the current trick state.
 *
 * Pure functions — the engine calls these to get legal actions for the current player.
 */

import { Card, Suit, Rank } from '../core/card';
import { PlayType, type PlayAction } from '../core/types';
import { detectSingles, detectPairs, detectTractors, detectThrowCandidates } from './pattern';
import { getLegalFollows } from './follow-rules';

/**
 * Enumerate all legal plays for the current player.
 *
 * @param hand - Current player's hand.
 * @param currentTrick - The current trick slots (null = not yet played).
 * @param trumpSuit - The trump suit.
 * @param trumpRank - The trump rank (= current level).
 * @param isLeading - True if this player is leading the trick.
 * @param leadPlayType - The play type of the lead (null if leading).
 * @param remainingCards - All cards not yet played (for throw validation).
 */
export function getLegalPlays(
  hand: Card[],
  currentTrick: { playerIndex: number; cards: Card[] | null }[],
  trumpSuit: Suit,
  trumpRank: Rank,
  isLeading: boolean,
  leadAction: PlayAction | null,
  remainingCards?: Card[],
): PlayAction[] {
  if (hand.length === 0) return [];

  if (isLeading || currentTrick.every(s => s.cards === null)) {
    return getLeadingPlays(hand, trumpSuit, trumpRank, remainingCards ?? []);
  }

  // Following
  if (!leadAction) return [];

  return getLegalFollows(hand, leadAction, trumpSuit, trumpRank);
}

/**
 * Get all legal plays when leading a trick.
 */
export function getLeadingPlays(
  hand: Card[],
  trumpSuit: Suit,
  trumpRank: Rank,
  remainingCards: Card[],
): PlayAction[] {
  const plays: PlayAction[] = [];

  // Single: every card
  plays.push(...detectSingles(hand));

  // Pair: every identical pair
  plays.push(...detectPairs(hand));

  // Tractor: consecutive pairs
  plays.push(...detectTractors(hand, trumpSuit, trumpRank));

  // Throw: for each non-trump suit
  for (const suit of [Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS, Suit.CLUBS]) {
    if (suit === trumpSuit) continue;
    const candidates = detectThrowCandidates(hand, suit, trumpSuit, trumpRank);
    // Filter throws: each card must be unbeatable by remaining cards of that suit
    for (const candidate of candidates) {
      if (isThrowValid(candidate, suit, trumpRank, remainingCards)) {
        plays.push(candidate);
      }
    }
  }

  return plays;
}

/**
 * Check if a throw is valid: all thrown cards must be the highest remaining
 * cards of that suit among all players.
 */
function isThrowValid(
  throwAction: PlayAction,
  suit: Suit,
  trumpRank: Rank,
  remainingCards: Card[],
): boolean {
  if (remainingCards.length === 0) return true;

  // Find the highest remaining card of this suit (excluding trump rank)
  const remainingOfSuit = remainingCards.filter(
    c => c.suit === suit && c.rank !== trumpRank
  );

  if (remainingOfSuit.length === 0) return true;

  // Sort remaining by rank descending
  const sortedRemaining = [...remainingOfSuit].sort((a, b) => {
    const rankOrder: Record<string, number> = {
      '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
      '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    };
    return (rankOrder[b.rank] || 0) - (rankOrder[a.rank] || 0);
  });

  const highestRemainingRank = sortedRemaining[0].rank;

  const rankOrder: Record<string, number> = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
  };

  // Each thrown card must be >= the highest remaining card of that suit
  return throwAction.cards.every(c =>
    (rankOrder[c.rank] || 0) >= (rankOrder[highestRemainingRank] || 0)
  );
}

/**
 * Filter legal plays to only those matching a specific play type.
 */
export function filterByType(plays: PlayAction[], type: PlayType): PlayAction[] {
  return plays.filter(p => p.type === type);
}

/**
 * Check if a specific play action is legal.
 */
export function isLegalPlay(
  action: PlayAction,
  legalPlays: PlayAction[],
): boolean {
  const actionIds = new Set(action.cards.map(c => c.id));
  return legalPlays.some(legal => {
    const legalIds = new Set(legal.cards.map(c => c.id));
    if (actionIds.size !== legalIds.size) return false;
    for (const id of actionIds) {
      if (!legalIds.has(id)) return false;
    }
    return true;
  });
}

/**
 * Get a human-readable list of legal play descriptions.
 * Useful for AI prompts.
 */
export function describeLegalPlays(plays: PlayAction[]): string[] {
  return plays.map(p => {
    const cards = p.cards.map(c => {
      if (c.isJoker) return c.isBigJoker ? '大王' : '小王';
      const s: Record<string, string> = {
        hearts: '♥', spades: '♠', diamonds: '♦', clubs: '♣',
      };
      return `${s[c.suit]}${c.rank}`;
    }).join(' ');

    switch (p.type) {
      case PlayType.SINGLE: return `单张: ${cards}`;
      case PlayType.PAIR: return `对子: ${cards}`;
      case PlayType.TRACTOR: return `拖拉机(${p.cards.length / 2}对): ${cards}`;
      case PlayType.THROW: return `甩牌: ${cards}`;
    }
  });
}
