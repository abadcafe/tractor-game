/**
 * Scoring and level advancement for 升级.
 *
 * Based on defender's points at end of round:
 *   0 pts           → Declarer +3 levels
 *   5-35 pts        → Declarer +2 levels
 *   40-75 pts       → Declarer +1 level
 *   80-115 pts      → Switch: defender becomes next declarer
 *   120-155 pts     → Defender +1 level
 *   160-195 pts     → Defender +2 levels
 *   200 pts         → Defender +3 levels
 *
 * 扣底 (Ambush): If defender wins the last trick,
 *   bottom card points are doubled (×2 for singles, ×4 for pairs/tractors).
 */

import { LEVELS } from '../core/constants';
import { Rank } from '../core/card';
import type { Card } from '../core/card';

/** Result of scoring a round. */
export interface ScoreResult {
  /** How many levels the declarer team changes (>0 = up, <0 = down). */
  declarerLevelChange: number;
  /** Which team will be the next declarer (0 or 1). */
  nextDeclarerTeam: number;
  /** The next level to play. */
  nextLevel: Rank;
  /** The current team's new level. */
  team0NewLevel: Rank;
  team1NewLevel: Rank;
  /** Points breakdown. */
  defenderPoints: number;
  bottomCardBonus: number;
  totalDefenderPoints: number;
}

/**
 * Calculate the score result for a round.
 *
 * @param defenderPoints - Points collected by defender from tricks.
 * @param bottomCards - The 8 bottom cards (底牌).
 * @param lastTrickWinnerTeam - Which team won the last trick (0=declarer, 1=defender).
 * @param declarerTeamIndex - Which team was declarer this round.
 * @param currentLevel - The level being played.
 */
export function calculateScore(
  defenderPoints: number,
  bottomCards: Card[],
  lastTrickWinnerTeam: number,
  declarerTeamIndex: number,
  currentLevel: Rank,
): ScoreResult {
  const defenderTeamIndex = declarerTeamIndex === 0 ? 1 : 0;

  // Calculate bottom card bonus (扣底)
  let bottomCardPoints = 0;
  for (const card of bottomCards) {
    bottomCardPoints += card.points;
  }

  // Ambush multiplier: ×2 if defender wins last trick
  // ×4 if the winning play was a pair/tractor (simplified: ×2 for now)
  const ambushMultiplier = lastTrickWinnerTeam === defenderTeamIndex ? 2 : 0;
  const bottomCardBonus = bottomCardPoints * ambushMultiplier;

  const totalPoints = defenderPoints + bottomCardBonus;

  // Determine level change
  let declarerChange: number;
  let switchDeclarer: boolean;

  if (totalPoints === 0) {
    declarerChange = 3;
    switchDeclarer = false;
  } else if (totalPoints < 40) {
    declarerChange = 2;
    switchDeclarer = false;
  } else if (totalPoints < 80) {
    declarerChange = 1;
    switchDeclarer = false;
  } else if (totalPoints < 120) {
    declarerChange = 0;
    switchDeclarer = true;
  } else if (totalPoints < 160) {
    declarerChange = -1;
    switchDeclarer = true;
  } else if (totalPoints < 200) {
    declarerChange = -2;
    switchDeclarer = true;
  } else {
    // 200 points
    declarerChange = -3;
    switchDeclarer = true;
  }

  // Calculate new levels
  const currentLevelIndex = LEVELS.indexOf(currentLevel);
  const declarerTeamIndex_actual = declarerTeamIndex;

  let team0Change = 0;
  let team1Change = 0;

  if (declarerTeamIndex_actual === 0) {
    team0Change = declarerChange;
    team1Change = declarerChange < 0 ? -declarerChange : 0;
  } else {
    team1Change = declarerChange;
    team0Change = declarerChange < 0 ? -declarerChange : 0;
  }

  // But levels only change for the declarer side if positive,
  // and for the defender side if the declarer change is negative

  // Simplified: just track the level change
  const team0LevelIndex = Math.max(0, Math.min(LEVELS.length - 1, currentLevelIndex + team0Change));
  const team1LevelIndex = Math.max(0, Math.min(LEVELS.length - 1, currentLevelIndex + team1Change));

  const nextDeclarerTeam = switchDeclarer ? defenderTeamIndex : declarerTeamIndex;
  const nextLevel = nextDeclarerTeam === 0 ? LEVELS[team0LevelIndex] : LEVELS[team1LevelIndex];

  return {
    declarerLevelChange: declarerChange,
    nextDeclarerTeam,
    nextLevel,
    team0NewLevel: LEVELS[team0LevelIndex],
    team1NewLevel: LEVELS[team1LevelIndex],
    defenderPoints,
    bottomCardBonus,
    totalDefenderPoints: totalPoints,
  };
}

/**
 * Check if a team has reached the target level (game over).
 */
export function isGameOver(level: Rank, targetLevel: Rank): boolean {
  const levelIndex = LEVELS.indexOf(level);
  const targetIndex = LEVELS.indexOf(targetLevel);
  return levelIndex > targetIndex; // Must pass target (e.g., reach A and win one more round)
}
