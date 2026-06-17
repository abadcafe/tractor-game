export type SeatInfo = {
  label: string;
  position: string;
  team: number;
};

export const HUMAN_TEAM = 0;
export const HUMAN_SEAT = 3;

export const SEAT_MAP: Record<number, SeatInfo> = {
  0: { label: "同伴", position: "北", team: 0 },
  1: { label: "左家", position: "西", team: 1 },
  2: { label: "右家", position: "东", team: 1 },
  3: { label: "你", position: "南", team: 0 },
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
