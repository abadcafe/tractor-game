import type { StateSnapshot } from "./types.ts";

/** Server -> Client WebSocket message.
 *  Matches the actual server protocol: type is always "state",
 *  with an optional "error" field for action rejection feedback. */
export type ServerMessage = {
  type: "state";
  seq: number;
  awaiting: string | null;
  state: StateSnapshot;
  error?: string;
};

/** Client -> Server WebSocket action.
 *  Each action may include `seq` for stale-action detection. */
export type ClientAction =
  | { type: "bid"; seq: number; cards: string[] }
  | { type: "bid"; seq: number; cards?: undefined; pass: true }
  | { type: "stir"; seq: number; cards: string[]; pass?: false }
  | { type: "stir"; seq: number; cards?: undefined; pass: true }
  | { type: "discard"; seq: number; cards: string[] }
  | { type: "play"; seq: number; cards: string[] }
  | { type: "next_round"; seq: number };
