import type { StateSnapshot } from "../core/types.ts";
import type { GameAction, BidButtonState, LevelChangeInfo } from "../engine/types.ts";

/** Callbacks for user interactions. Created in main.ts, passed through renderer to components. */
export interface ActionCallbacks {
  /** Called when a card in hand is clicked. Toggles selection. */
  onCardClick: (cardId: string) => void;
  /** Called when an action button is clicked. */
  onAction: (action: GameAction) => void;
  /** Called when the player submits a bid with selected card IDs during DEAL_BID phase.
   *  Sends { type: "bid", cards: cardIds } to the server. */
  onBid: (cardIds: string[]) => void;
  /** Called when the player submits a stir with selected card IDs during STIRRING phase.
   *  Sends { type: "stir", cards: cardIds } to the server. */
  onStir: (cardIds: string[]) => void;
  /** Called when the player passes on stirring.
   *  Sends { type: "stir", pass: true } to the server. */
  onPass: () => void;
  /** Called when the player clicks "new game" on the game over screen. */
  onNewGame: () => void;
}

/** Context bundle passed to render() and UI components. */
export interface RenderContext {
  callbacks?: ActionCallbacks;
  selectedCardIds: Set<string>;
  /** Pre-computed legal card IDs for hand highlighting. */
  legalCardIds: Set<string>;
  /** Pre-computed bid button state from engine layer. */
  bidButtonState?: BidButtonState;
  /** Pre-computed stir button state from engine layer. */
  stirButtonState?: BidButtonState;
  /** Pre-computed level change info from engine layer. */
  levelChange?: LevelChangeInfo;
}
