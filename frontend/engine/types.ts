/** Types exported from engine layer to UI layer. */

/** Action type for onAction callback. */
export type GameAction = "play" | "discard" | "next_round";

/** Interaction mode computed by GameLoop from awaiting_action.
 *  Passed to renderer so it knows which buttons/dialogs to show.
 *  null = spectator mode (no interaction). */
export type InteractionMode = "bid" | "stir" | "discard" | "play" | "next_round" | null;

/** A single bid option computed from the player's hand.
 *  Displayed as a clickable pill in the bidding panel.
 *  Clicking sets the pending bid intent. */
export interface BidOption {
  /** Card IDs to bid with. */
  cardIds: string[];
  /** Human-readable label, e.g. "♠2对" or "大王对". */
  label: string;
  /** Resulting trump suit if this bid wins. null for joker bids. */
  trumpSuit: string | null;
  /** Bid priority (higher = stronger). */
  priority: number;
}

/** Pre-computed button state for stirring dialog. */
export interface StirButtonState {
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
