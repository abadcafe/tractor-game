export type SeatDirection = "north" | "west" | "south" | "east";
export type SeatPosition = "北" | "西" | "南" | "东";

export type SeatInfo = {
  label: string;
  position: SeatPosition;
  direction: SeatDirection;
  team: number;
};

export const HUMAN_TEAM = 0;
export const HUMAN_SEAT = 2;

export const SEAT_MAP: Record<number, SeatInfo> = {
  0: { label: "同伴", position: "北", direction: "north", team: 0 },
  1: { label: "左家", position: "西", direction: "west", team: 1 },
  2: { label: "你", position: "南", direction: "south", team: 0 },
  3: { label: "右家", position: "东", direction: "east", team: 1 },
};

export function WS_PATH(gameId: string): string {
  return `/game/${gameId}`;
}

export const API_BASE = "/api/game";

/** Human-readable team labels. */
export const TEAM_LABELS: Record<number, string> = {
  0: "我方",
  1: "对方",
};
