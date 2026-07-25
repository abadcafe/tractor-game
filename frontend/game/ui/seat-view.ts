import {
  nextSeat,
  partnerSeat,
  type PartnershipId,
  partnershipOf,
  previousSeat,
  type SeatId,
} from "../config.ts";

export type TableSlot = "top" | "right" | "bottom" | "left";
export type SeatPosition = "上家" | "下家" | "对家" | "本人";

export interface SeatView {
  seat: SeatId;
  label: string;
  avatarText: string;
  position: SeatPosition;
  slot: TableSlot;
  partnership: PartnershipId;
  partnershipLabel: string;
  isViewer: boolean;
}

export function seatView(
  seat: SeatId,
  viewer: SeatId,
): SeatView {
  const partnership = partnershipOf(seat);
  return {
    seat,
    label: seat === viewer ? `座位 ${seat} / 你` : `座位 ${seat}`,
    avatarText: seat.toUpperCase(),
    position: positionFor(seat, viewer),
    slot: slotFor(seat, viewer),
    partnership,
    partnershipLabel: partnershipLabelForViewer(
      partnership,
      viewer,
    ),
    isViewer: seat === viewer,
  };
}

export function partnershipLabelForViewer(
  partnership: PartnershipId,
  viewer: SeatId,
): string {
  return partnership === partnershipOf(viewer) ? "我方" : "对方";
}

export function viewerPartnership(
  viewer: SeatId,
): PartnershipId {
  return partnershipOf(viewer);
}

function positionFor(
  seat: SeatId,
  viewer: SeatId,
): SeatPosition {
  if (seat === viewer) return "本人";
  if (seat === nextSeat(viewer)) return "下家";
  if (seat === partnerSeat(viewer)) return "对家";
  if (seat === previousSeat(viewer)) return "上家";
  throw new Error("invalid seat topology");
}

function slotFor(seat: SeatId, viewer: SeatId): TableSlot {
  if (seat === viewer) return "bottom";
  if (seat === nextSeat(viewer)) return "right";
  if (seat === partnerSeat(viewer)) return "top";
  if (seat === previousSeat(viewer)) return "left";
  throw new Error("invalid seat topology");
}
