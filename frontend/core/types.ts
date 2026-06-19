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

/** Public event emitted when a throw attempt is forced to a smaller sub-play. */
export interface FailedThrow {
  player: number;
  attempted_cards: Card[];
  forced_cards: Card[];
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

/** Full game state snapshot pushed by the server.
 *  Matches server/snapshot.py SnapshotDict exactly. */
export interface StateSnapshot {
  phase: "DEAL_BID" | "STIRRING" | "PLAYING" | "WAITING" | "GAME_OVER";

  player_hand: Card[];
  player_hand_counts: number[];
  bottom_cards: Card[];

  trump_rank: string;
  trump_suit: string | null;

  declarer_team: number | null;
  declarer_player: number | null;

  defender_points: number;

  /**
   * Advisory card-group hints for the current awaiting_action.
   * Non-empty means the backend is providing a complete hint set that the UI
   * may use for highlighting or shortcuts. Empty means no hint is provided;
   * it must not disable user input. The backend still validates every action.
   */
  action_hints: Card[][];

  trick: {
    lead_player: number;
    slots: TrickSlot[];
    current_player: number;
  } | null;

  trick_history: CompletedTrick[];
  failed_throw: FailedThrow | null;

  bid_events: BidEvent[];
  bid_winner: BidEvent | null;

  awaiting_action: string | null;

  stirring_state: {
    phase: string;
    trump_suit: string | null;
    current_player: number;
    declarer_player: number;
    exchanging_player: number | null;
    exchange_count: number | null;
  } | null;

  scoring: {
    declarer_team: number | null;
    defender_points: number;
    total_defender_points: number;
    bottom_card_bonus: number;
    bottom_cards: Card[];
  } | null;

  winning_team: number | null;

  team0_level: string;
  team1_level: string;
  next_round_confirmed: number[];
}
