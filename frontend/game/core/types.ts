import type { PartnershipId, SeatId } from "../config.ts";

export type Suit =
  | "hearts"
  | "spades"
  | "diamonds"
  | "clubs"
  | "joker";
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
  | "WAITING";
export type StirringPhase = "WAITING" | "EXCHANGING";
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

export type Trump =
  | { readonly kind: "pending"; readonly rank: Rank }
  | { readonly kind: "no_trump"; readonly rank: Rank }
  | {
    readonly kind: "suited";
    readonly rank: Rank;
    readonly suit: Suit;
  };

export interface RemainingCards {
  readonly a: number;
  readonly b: number;
  readonly c: number;
  readonly d: number;
}

export interface PartnershipLevels {
  readonly first: Rank;
  readonly second: Rank;
}

export interface TrickSlot {
  actor: SeatId;
  cards: Card[];
}

export interface FailedThrow {
  actor: SeatId;
  attempted_cards: Card[];
  forced_cards: Card[];
}

export interface CompletedTrick {
  lead_actor: SeatId;
  slots: TrickSlot[];
  winner: SeatId;
  points: number;
  failed_throw: FailedThrow | null;
}

export interface BidEvent {
  actor: SeatId;
  cards: Card[];
  kind: BidEventKind;
  suit: Suit | null;
  joker_type: JokerType | null;
  count: number;
  deal_ordinal: number;
}

export interface BottomExchange {
  actor: SeatId;
  trigger: "initial" | "stir";
  stir_event_index: number | null;
  picked_up_bottom_cards: Card[];
  discarded_bottom_cards: Card[];
  resulting_bottom_cards: Card[];
}

export interface StirDeclarationEvent {
  actor: SeatId;
  kind: StirEventKind;
  cards: Card[];
  new_suit: Suit | null;
  priority: number | null;
  own_bottom_exchange: BottomExchange | null;
}

export interface StateSnapshot {
  phase: RoundPhase;
  round_number: number;
  hand: Card[];
  remaining_cards: RemainingCards;
  bottom_cards: Card[];
  trump: Trump;
  declarer: SeatId | null;
  defender_points: number;
  action_hints: Card[][];
  trick: {
    lead_actor: SeatId;
    slots: TrickSlot[];
    current_actor: SeatId;
    failed_throw: FailedThrow | null;
  } | null;
  last_completed_trick: CompletedTrick | null;
  defender_point_cards: Card[];
  bid_events: BidEvent[];
  bid_winner: BidEvent | null;
  own_initial_bottom_exchange: BottomExchange | null;
  stir_events: StirDeclarationEvent[];
  awaiting_action: AwaitingAction | null;
  stirring_state: {
    phase: StirringPhase;
    trump_suit: Suit | null;
    current_actor: SeatId;
    declarer: SeatId;
    exchanging_actor: SeatId | null;
    exchange_count: number | null;
  } | null;
  scoring: {
    winning_partnership: PartnershipId;
    defender_points: number;
    total_defender_points: number;
    bottom_base_points: number;
    bottom_multiplier: number | null;
    bottom_points: number;
    bottom_cards: Card[];
  } | null;
  winning_partnership: PartnershipId | null;
  partnership_levels: PartnershipLevels;
  mandatory_levels: Rank[];
  next_round_confirmed: SeatId[];
}

export function trumpRank(state: StateSnapshot): Rank {
  return state.trump.rank;
}

export function trumpSuit(state: StateSnapshot): Suit | null {
  return state.trump.kind === "suited" ? state.trump.suit : null;
}
