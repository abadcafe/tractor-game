import type { StateSnapshot } from "../core/types.ts";
import type { GameAction, BidOption, StirButtonState, LevelChangeInfo } from "../engine/types.ts";

/** Callbacks for user interactions. Created in main.ts, passed through renderer to components. */
export interface ActionCallbacks {
  /** Called when a card in hand is clicked. Toggles selection. */
  onCardClick: (cardId: string) => void;
  /** Called when an action button is clicked. */
  onAction: (action: GameAction) => void;
  /** Called when the player clicks a bid option to set their pending bid intent. */
  onBidOptionSelect: (option: BidOption) => void;
  /** Called when the player submits a stir with selected card IDs during STIRRING phase. */
  onStir: (cardIds: string[]) => void;
  /** Called when the player passes on stirring. */
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
  /** Pre-computed bid options from engine layer. */
  bidOptions?: BidOption[];
  /** Current pending bid intent set by the player. */
  pendingBidIntent?: BidOption | null;
  /** Pre-computed stir button state from engine layer. */
  stirButtonState?: StirButtonState;
  /** Pre-computed level change info from engine layer. */
  levelChange?: LevelChangeInfo;
}
