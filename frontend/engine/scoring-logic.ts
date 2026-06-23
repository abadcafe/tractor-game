/** Scoring thresholds matching server-side SCORE_THRESHOLDS.
 *  Each threshold: max_points (inclusive), declarer_change, switch_declarer.
 *  Levels never retreat. For defender_points >= 80, formula applies:
 *  defender_gain = max(0, (points - 80) / 40)
 */
const SCORE_THRESHOLDS = [
  { maxPoints: 0, declarerChange: 3, switch: false },
  { maxPoints: 39, declarerChange: 2, switch: false },
  { maxPoints: 79, declarerChange: 1, switch: false },
];

/** Level change result from computeLevelChange. */
interface LevelChangeResult {
  declarerDelta: number;
  defenderDelta: number;
  switched: boolean;
}

/** Compute level change from total defender points.
 *  Matches server-side _determine_level_change() logic.
 *  - declarerDelta: levels the declarer team gains (never negative)
 *  - defenderDelta: levels the defender team gains when they win (switch=true)
 *  - switched: whether the declarer switches to the defender team
 */
export function computeLevelChange(
  totalPoints: number,
): LevelChangeResult {
  for (const t of SCORE_THRESHOLDS) {
    if (totalPoints <= t.maxPoints) {
      return {
        declarerDelta: t.declarerChange,
        defenderDelta: 0,
        switched: t.switch,
      };
    }
  }
  // defender_points >= 80: switch declarer, new declarer gains levels
  const defenderGain = Math.max(0, Math.floor((totalPoints - 80) / 40));
  return {
    declarerDelta: 0,
    defenderDelta: defenderGain,
    switched: true,
  };
}
