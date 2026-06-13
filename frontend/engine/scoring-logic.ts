/** Scoring thresholds matching server-side SCORE_THRESHOLDS.
 *  Each threshold: max_points (inclusive), declarer_change, switch_declarer.
 */
const SCORE_THRESHOLDS = [
  { maxPoints: 0, declarerChange: 3, switch: false },
  { maxPoints: 39, declarerChange: 2, switch: false },
  { maxPoints: 79, declarerChange: 1, switch: false },
  { maxPoints: 119, declarerChange: 0, switch: true },
  { maxPoints: 159, declarerChange: -1, switch: true },
  { maxPoints: 199, declarerChange: -2, switch: true },
  { maxPoints: 200, declarerChange: -3, switch: true },
];

/** Level change result from computeLevelChange. */
interface LevelChangeResult {
  delta: number;
  switched: boolean;
}

/** Compute level change from total defender points.
 *  Matches server-side _determine_level_change() logic.
 *  delta > 0 = declarer levels up, delta < 0 = declarer levels down.
 */
export function computeLevelChange(totalPoints: number): LevelChangeResult {
  for (const t of SCORE_THRESHOLDS) {
    if (totalPoints <= t.maxPoints) {
      return { delta: t.declarerChange, switched: t.switch };
    }
  }
  // Fallback (should not reach here)
  return { delta: -3, switched: true };
}
