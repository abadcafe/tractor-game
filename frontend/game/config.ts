export type SeatId = "a" | "b" | "c" | "d";
export type PartnershipId = "first" | "second";

export const SEAT_IDS: readonly [SeatId, SeatId, SeatId, SeatId] = [
  "a",
  "b",
  "c",
  "d",
];

export function seatIdFromString(value: string): SeatId | null {
  switch (value) {
    case "a":
    case "b":
    case "c":
    case "d":
      return value;
    default:
      return null;
  }
}

export function nextSeat(seat: SeatId): SeatId {
  switch (seat) {
    case "a":
      return "b";
    case "b":
      return "c";
    case "c":
      return "d";
    case "d":
      return "a";
  }
}

export function previousSeat(seat: SeatId): SeatId {
  switch (seat) {
    case "a":
      return "d";
    case "b":
      return "a";
    case "c":
      return "b";
    case "d":
      return "c";
  }
}

export function partnerSeat(seat: SeatId): SeatId {
  switch (seat) {
    case "a":
      return "c";
    case "b":
      return "d";
    case "c":
      return "a";
    case "d":
      return "b";
  }
}

export function partnershipOf(seat: SeatId): PartnershipId {
  return seat === "a" || seat === "c" ? "first" : "second";
}

export function WS_PATH(
  gameId: string,
  seatId: SeatId,
  userId: string,
): string {
  return GAME_SEAT_PATH(gameId, seatId, userId);
}

export function GAME_SEAT_PATH(
  gameId: string,
  seatId: SeatId,
  userId: string,
): string {
  return `/game/${encodeURIComponent(gameId)}/seat/${seatId}` +
    `?user_id=${encodeURIComponent(userId)}`;
}

export const API_BASE = "/api/game";
export const SEAT_VACATED_WS_CLOSE_CODE = 4408;

export const PARTNERSHIP_LABELS: Record<PartnershipId, string> = {
  first: "搭档方 A",
  second: "搭档方 B",
};
