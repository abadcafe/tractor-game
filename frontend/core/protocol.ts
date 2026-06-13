import type { StateSnapshot } from "./types.ts";

/** Server -> Client WebSocket message. */
export type ServerMessage =
  | { type: "state"; awaiting: string | null; state: StateSnapshot }
  | { type: "error"; message: string };

/** Client -> Server WebSocket action. */
export type ClientAction =
  | { type: "bid"; cards: string[] }
  | { type: "stir"; cards: string[]; pass?: false }
  | { type: "stir"; cards?: undefined; pass: true }
  | { type: "discard"; cards: string[] }
  | { type: "play"; cards: string[] }
  | { type: "next_round" };
