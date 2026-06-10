import type { Card } from "../core/types.ts";
import { isJoker, isTrumpRank } from "../core/card.ts";

/**
 * Validate selected cards against a list of legal play options.
 * Returns the first matching card list, or null if no match.
 * Matching: every selected card ID must appear in one option's card IDs (subset match).
 */
export function validatePlay(
  selectedCards: Card[],
  legalActions: Card[][],
): Card[] | null {
  if (selectedCards.length === 0) {
    return null;
  }
  const selectedIds = new Set(selectedCards.map((c) => c.id));
  for (const cards of legalActions) {
    const actionIds = new Set(cards.map((c) => c.id));
    // Require exact match (same size + all selected IDs in action).
    // Subset match would let the player accidentally send extra cards
    // (e.g. clicking 1 card that's part of a 2-card pair),
    // causing hand-size imbalance that eventually deadlocks the game.
    if (selectedIds.size !== actionIds.size) {
      continue;
    }
    if ([...selectedIds].every((id) => actionIds.has(id))) {
      return cards;
    }
  }
  return null;
}

/**
 * Validate that the number of selected discard cards matches the expected count.
 */
export function validateDiscard(
  selectedCards: Card[],
  expectedCount: number,
): boolean {
  return selectedCards.length === expectedCount;
}

/**
 * Validate that all selected cards are valid for bidding:
 * each must be a joker or have the trump rank, and selection must not be empty.
 */
export function validateBidCards(
  selectedCards: Card[],
  trumpRank: string,
): boolean {
  if (selectedCards.length === 0) {
    return false;
  }
  return selectedCards.every(
    (c) => isJoker(c) || isTrumpRank(c, trumpRank),
  );
}
