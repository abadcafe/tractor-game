/** Types exported from engine layer to UI layer. */

/** Action type for onAction callback. Replaces raw string parameter. */
export type GameAction = "play" | "discard" | "next_round";

/** Interaction mode computed by GameLoop from awaiting_action.
 *  Passed to renderer so it knows which buttons/dialogs to show.
 *  null = spectator mode (no interaction). */
export type InteractionMode = "bid" | "stir" | "discard" | "play" | "next_round" | null;

/** Pre-computed button state for bidding/stirring dialogs.
 *  Calculated by engine layer, passed to UI layer for rendering. */
export interface BidButtonState {
  disabled: boolean;
  title?: string;
}

/** Pre-computed level change info for scoring overlay.
 *  Calculated by engine layer, passed to UI layer for rendering. */
export interface LevelChangeInfo {
  declarerDelta: number;
  defenderDelta: number;
  switched: boolean;
}
