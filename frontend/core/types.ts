/** A single card from the backend. Matches _card_to_dict() output. */
export type Suit = "hearts" | "spades" | "diamonds" | "clubs" | "joker";
export type Rank =
  | "2"
  | "3"
  | "4"
  | "5"
  | "6"
  | "7"
  | "8"
  | "9"
  | "10"
  | "J"
  | "Q"
  | "K"
  | "A"
  | "SJ"
  | "BJ";
export type RoundPhase =
  | "DEAL_BID"
  | "STIRRING"
  | "PLAYING"
  | "SCORING"
  | "WAITING";
export type StirringPhase = "WAITING" | "EXCHANGING" | "COMPLETE";
export type AwaitingAction =
  | "bid"
  | "stir"
  | "discard"
  | "play"
  | "next_round";
export type BidEventKind = "trump_rank" | "joker";
export type JokerType = "big" | "small";
export type StirEventKind = "stir" | "pass";

export interface Card {
  id: string;
  suit: Suit;
  rank: Rank;
}

/** One player's contribution in a trick slot. */
export interface TrickSlot {
  player: number;
  cards: Card[];
}

/** Last completed trick visible to players. */
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
  kind: BidEventKind;
  suit: Suit | null;
  joker_type: JokerType | null;
  count: number;
}

export interface StirDeclarationEvent {
  player: number;
  kind: StirEventKind;
  cards: Card[];
  new_suit: Suit | null;
  priority: number | null;
  own_bottom_exchange: BottomExchange | null;
}

export interface BottomExchange {
  picked_up_bottom_cards: Card[];
  discarded_bottom_cards: Card[];
}

/** Full game state snapshot pushed by the server.
 *  Matches server/protocol/snapshot.py StateSnapshot exactly. */
export interface StateSnapshot {
  phase: RoundPhase;

  player_hand: Card[];
  player_hand_counts: number[];
  bottom_cards: Card[];

  trump_rank: Rank;
  trump_suit: Suit | null;

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

  last_completed_trick: CompletedTrick | null;
  defender_point_cards: Card[];
  failed_throw: FailedThrow | null;

  bid_events: BidEvent[];
  bid_winner: BidEvent | null;
  own_initial_bottom_exchange: BottomExchange | null;
  stir_events: StirDeclarationEvent[];

  awaiting_action: AwaitingAction | null;

  stirring_state: {
    phase: StirringPhase;
    trump_suit: Suit | null;
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

  team0_level: Rank;
  team1_level: Rank;
  next_round_confirmed: number[];
}
