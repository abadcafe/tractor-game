/** A single card from the backend. Matches _card_to_dict() output. */
export interface Card {
  id: string;
  suit: string;
  rank: string;
}

/** One player's contribution in a trick slot. */
export interface TrickSlot {
  player: number;
  cards: Card[];
}

/** A completed trick in trick_history. */
export interface CompletedTrick {
  lead_player: number;
  slots: TrickSlot[];
  winner: number;
  points: number;
}

/** A bid event during DEAL_BID phase. */
export interface BidEvent {
  player: number;
  cards: Card[];
  kind: "trump_rank" | "joker";
  suit: string | null;
  joker_type: "big" | "small" | null;
  count: number;
}

/** Full game state snapshot pushed by the server. */
export interface StateSnapshot {
  phase: "DEAL_BID" | "STIRRING" | "EXCHANGE" | "PLAYING" | "COMPLETE" | "GAME_OVER";

  player_hand: Card[];
  bottom_cards: Card[];

  trump_rank: string;
  trump_suit: string | null;

  declarer_team: number | null;
  declarer_player: number | null;

  current_player: number;
  defender_points: number;

  legal_actions: Card[][];

  trick: {
    lead_player: number;
    slots: TrickSlot[];
    current_player: number;
  } | null;

  trick_history: CompletedTrick[];

  bid_events: BidEvent[];
  bid_winner: BidEvent | null;

  awaiting_action: string | null;

  stirring_state: {
    phase: string;
    trump_suit: string | null;
    current_player: number;
  } | null;

  exchange_state: {
    phase: string;
    declarer_player: number;
    count: number;
  } | null;

  scoring: {
    declarer_team: number;
    defender_points: number;
    bottom_cards: Card[];
  } | null;

  winning_team: number | null;

  team0_level: string;
  team1_level: string;
}

/** Server -> Client WebSocket message. */
export type ServerMessage =
  | { type: "state"; awaiting: string | null; state: StateSnapshot }
  | { type: "error"; message: string };

/** Client -> Server WebSocket action. */
export type ClientAction =
  | { type: "bid"; cards: string[] }
  | { type: "stir"; cards: string[] }
  | { type: "stir"; pass: true }
  | { type: "discard"; cards: string[] }
  | { type: "play"; cards: string[] }
  | { type: "next_round" };

/** Interaction mode computed by GameLoop from awaiting + current_player.
 *  Passed to renderer so it knows which buttons/dialogs to show.
 *  null = spectator mode (no interaction). */
export type InteractionMode = "bid" | "stir" | "discard" | "play" | "next_round" | null;

/** Callbacks for user interactions. Created in main.ts, passed through renderer to components.
 *  These callbacks close over selectedCardIds state and the wsClient.send function.
 *  onStir is separate from onBid because the STIRRING phase sends { type: "stir" }
 *  while the DEAL_BID phase sends { type: "bid" } -- they are distinct server actions. */
export interface ActionCallbacks {
  /** Called when a card in hand is clicked. Toggles selection in the parent closure. */
  onCardClick: (cardId: string) => void;
  /** Called when an action button is clicked. The action string is "play", "discard", "next_round". */
  onAction: (action: string) => void;
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
