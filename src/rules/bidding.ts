/**
 * Bidding rules for 升级, including 炒地皮 (Chaodipi / Stir-fry).
 *
 * Phase 1: Initial Bidding (叫牌)
 *   - Players bid in turn, starting from the first player after the dealer.
 *   - First bid must be at least the current level.
 *   - Subsequent bids must be higher than the current highest bid.
 *   - A player may pass.
 *   - Bidding ends when 3 consecutive players pass, or everyone passes.
 *
 * Phase 2: Stirring (炒地皮)
 *   - After initial bidding winner picks up bottom cards,
 *     other players may "stir" (炒) to steal declarer rights.
 *   - Stir at same level: must change trump suit.
 *   - Stir at higher level: free choice of trump.
 *   - Original winner can counter-stir (反炒).
 *   - Continues until all players pass in sequence.
 */

import { Rank, Suit } from '../core/card';
import { LEVELS, nextPlayer } from '../core/constants';
import type { BidAction, StirAction } from '../core/types';

// ---- Initial Bidding ----

/**
 * Check if a bid is valid given the current bidding state.
 */
export function isValidBid(
  bidLevel: Rank | null,
  pass: boolean,
  currentHighestBid: Rank | null,
  currentLevel: Rank,
  _isFirstBidder: boolean,
): boolean {
  if (pass) return true; // Can always pass

  if (bidLevel === null) return false;

  // Must bid at least the current level
  const bidIndex = LEVELS.indexOf(bidLevel);
  const currentIndex = LEVELS.indexOf(currentLevel);
  if (bidIndex < currentIndex) return false;

  // If there's already a bid, must bid higher
  if (currentHighestBid !== null) {
    const highestIndex = LEVELS.indexOf(currentHighestBid);
    if (bidIndex <= highestIndex) return false;
  }

  return true;
}

/**
 * Get the valid bid levels for a player given the current state.
 */
export function getValidBidLevels(
  currentHighestBid: Rank | null,
  currentLevel: Rank,
): Rank[] {
  const currentIndex = LEVELS.indexOf(currentLevel);
  const startIndex = currentHighestBid
    ? LEVELS.indexOf(currentHighestBid) + 1
    : currentIndex;

  return LEVELS.slice(startIndex);
}

/**
 * Check if the bidding round is over.
 * Ends when 3 consecutive passes after the first bid, or everyone passed.
 */
export function isBiddingOver(
  bids: BidAction[],
  playerCount: number,
): boolean {
  if (bids.length === 0) return false;

  // All players passed with no bid
  const anyBid = bids.some(b => !b.pass);
  if (!anyBid && bids.length >= playerCount) return true;

  // Three consecutive passes after a bid was made
  let consecutivePasses = 0;
  for (let i = bids.length - 1; i >= 0; i--) {
    if (bids[i].pass) {
      consecutivePasses++;
      if (consecutivePasses >= 3) return true;
    } else {
      break;
    }
  }

  return false;
}

/**
 * Get the winning bid from bidding history.
 */
export function getWinningBid(bids: BidAction[]): BidAction | null {
  let winner: BidAction | null = null;
  let highestIndex = -1;

  for (const bid of bids) {
    if (bid.pass || bid.level === null) continue;
    const idx = LEVELS.indexOf(bid.level);
    if (idx > highestIndex) {
      highestIndex = idx;
      winner = bid;
    }
  }

  return winner;
}

// ---- 炒地皮 (Stirring) ----

/**
 * Check if a stir action is valid.
 *
 * @param stir - The proposed stir action.
 * @param currentTrumpSuit - The current trump suit (set by initial bid winner or previous stir).
 * @param currentBidLevel - The current bid level.
 * @param stirringHistory - All previous stir actions in this round.
 * @param playerIndex - The player attempting to stir.
 */
export function isValidStir(
  stir: StirAction,
  currentTrumpSuit: Suit,
  currentBidLevel: Rank,
  _stirringHistory: StirAction[],
  _playerIndex: number,
): boolean {
  const stirLevelIndex = LEVELS.indexOf(stir.level);
  const bidLevelIndex = LEVELS.indexOf(currentBidLevel);

  // Must be at or above current bid level
  if (stirLevelIndex < bidLevelIndex) return false;

  // Same level: must change trump suit
  if (stirLevelIndex === bidLevelIndex) {
    if (stir.newTrumpSuit === currentTrumpSuit) return false;
  }

  // Can't stir twice in a row (but CAN counter-stir if someone else stirred)
  // Actually, the rule is: you can stir after someone else has stirred
  // The last stirrer can't stir again until someone else stirs

  return true;
}

/**
 * Get valid stir options for a player.
 */
export function getValidStirOptions(
  currentTrumpSuit: Suit,
  currentBidLevel: Rank,
  playerIndex: number,
  _stirringHistory: StirAction[],
): StirAction[] {
  const options: StirAction[] = [];

  // Stir at same level: change trump
  for (const suit of [Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS, Suit.CLUBS]) {
    if (suit === currentTrumpSuit) continue;
    options.push({
      playerIndex,
      newTrumpSuit: suit,
      level: currentBidLevel,
    });
  }

  // Stir at higher levels
  const currentIndex = LEVELS.indexOf(currentBidLevel);
  for (let i = currentIndex + 1; i < LEVELS.length; i++) {
    for (const suit of [Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS, Suit.CLUBS]) {
      options.push({
        playerIndex,
        newTrumpSuit: suit,
        level: LEVELS[i],
      });
    }
  }

  return options;
}

/**
 * Check if the stirring round is over.
 * Ends when all remaining players pass in sequence (one full round of passes).
 */
export function isStirringOver(
  stirPasses: number,  // Consecutive passes
  playerCount: number,
): boolean {
  return stirPasses >= playerCount;
}

/**
 * Get the next player to act in bidding/stirring.
 */
export function getNextBidder(currentPlayer: number, _direction: 1 | -1 = 1): number {
  return nextPlayer(currentPlayer);
}
