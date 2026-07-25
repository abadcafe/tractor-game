import type {
  Card,
  CompletedTrick,
  FailedThrow,
  StateSnapshot,
  TrickSlot,
} from "../../core/types.ts";
import { trumpRank, trumpSuit } from "../../core/types.ts";
import { el } from "../dom.ts";
import { cardDisplay, sortHand, suitSymbol } from "../../core/card.ts";
import { SEAT_IDS, type SeatId } from "../../config.ts";
import { seatView } from "../seat-view.ts";

/**
 * Render the current trick area showing played cards with player labels,
 * previous-trick preview, and failed-throw preview.
 */
export function renderTrickView(
  snapshot: StateSnapshot,
  viewerSeat: SeatId,
  previousTrickPreview?: CompletedTrick | null,
  failedThrowPreview?: FailedThrow | null,
): HTMLElement {
  const showingFailedThrow = failedThrowPreview !== null &&
    failedThrowPreview !== undefined;
  const showingPrevious = !showingFailedThrow &&
    previousTrickPreview !== null &&
    previousTrickPreview !== undefined;
  const scoringTrick = scoringTrickPreview(snapshot);
  const showingScoringTrick = !showingPrevious && !showingFailedThrow &&
    scoringTrick !== null;
  const trickView = el("div", {
    class: trickViewClass(
      showingPrevious || showingScoringTrick,
      showingFailedThrow,
    ),
  });

  const grid = showingPrevious
    ? renderCompletedTrickGrid(
      previousTrickPreview,
      snapshot,
      viewerSeat,
    )
    : showingScoringTrick
    ? renderCompletedTrickGrid(scoringTrick, snapshot, viewerSeat)
    : renderCurrentTrickGrid(snapshot, viewerSeat);
  if (grid !== null) {
    trickView.appendChild(grid);
  }
  if (showingFailedThrow) {
    trickView.appendChild(
      renderFailedThrowPreview(
        failedThrowPreview,
        snapshot,
        viewerSeat,
      ),
    );
  }

  return trickView;
}

function scoringTrickPreview(
  snapshot: StateSnapshot,
): CompletedTrick | null {
  if (
    snapshot.phase !== "WAITING" ||
    snapshot.scoring === null ||
    snapshot.trick !== null
  ) {
    return null;
  }
  return snapshot.last_completed_trick;
}

function trickViewClass(
  showingPrevious: boolean,
  showingFailedThrow: boolean,
): string {
  if (showingFailedThrow) {
    return "trick-view showing-failed-throw";
  }
  if (showingPrevious) {
    return "trick-view showing-previous";
  }
  return "trick-view";
}

function renderCurrentTrickGrid(
  snapshot: StateSnapshot,
  viewerSeat: SeatId,
): HTMLElement | null {
  if (!snapshot.trick) {
    return null;
  }
  const grid = el("div", { class: "trick-grid" });
  const slotsBySeat = new Map(
    snapshot.trick.slots.map((slot) => [slot.actor, slot]),
  );

  for (const seat of SEAT_IDS) {
    const tableSlot = seatView(seat, viewerSeat).slot;
    const isLead = seat === snapshot.trick.lead_actor;
    const isCurrent = seat === snapshot.trick.current_actor;
    const slot = slotsBySeat.get(seat);

    let slotClass = `${
      slot ? "trick-slot" : "trick-placeholder-slot"
    } trick-slot-${tableSlot}`;
    if (isLead) slotClass += " lead";
    if (isCurrent && !isLead) slotClass += " current";
    if (!slot) slotClass += " empty";

    grid.appendChild(
      renderTrickSlot(
        seat,
        slot,
        slotClass,
        isLead,
        snapshot,
        viewerSeat,
      ),
    );
  }

  return grid;
}

function renderCompletedTrickGrid(
  trick: CompletedTrick,
  snapshot: StateSnapshot,
  viewerSeat: SeatId,
): HTMLElement {
  const grid = el("div", { class: "trick-grid trick-grid--previous" });
  const slotsBySeat = new Map(
    trick.slots.map((slot) => [slot.actor, slot]),
  );

  for (const seat of SEAT_IDS) {
    const tableSlot = seatView(seat, viewerSeat).slot;
    const slot = slotsBySeat.get(seat);
    let slotClass = `trick-slot trick-slot-${tableSlot}`;
    const isLead = seat === trick.lead_actor;
    if (isLead) slotClass += " lead";
    if (seat === trick.winner) slotClass += " winner";
    grid.appendChild(
      renderTrickSlot(
        seat,
        slot,
        slotClass,
        isLead,
        snapshot,
        viewerSeat,
      ),
    );
  }

  return grid;
}

function renderTrickSlot(
  seat: SeatId,
  slot: TrickSlot | undefined,
  slotClass: string,
  isLead: boolean,
  snapshot: StateSnapshot,
  viewerSeat: SeatId,
): HTMLElement {
  const slotEl = el("div", { class: slotClass });
  const seatInfo = seatView(seat, viewerSeat);
  if (slot && slot.cards.length > 0) {
    slotEl.appendChild(
      el("span", { class: "trick-player-label" }, seatInfo.label),
    );
    if (isLead) {
      slotEl.appendChild(
        el("span", { class: "trick-lead-marker" }, "先出"),
      );
    }
  }

  const cardsDiv = el("div", { class: "trick-cards" });
  const cards = sortTrickSlotCards(slot?.cards ?? [], snapshot);
  for (const card of cards) {
    cardsDiv.appendChild(renderTrickCard(card));
  }
  slotEl.appendChild(cardsDiv);
  return slotEl;
}

function sortTrickSlotCards(
  cards: Card[],
  snapshot: StateSnapshot,
): Card[] {
  return sortHand(cards, trumpSuit(snapshot), trumpRank(snapshot));
}

function renderTrickCard(card: Card): HTMLElement {
  return el("span", {
    class: trickCardClass(card),
    "data-rank": card.rank,
    "data-suit-symbol": suitSymbol(card.suit),
  }, cardDisplay(card));
}

function trickCardClass(card: Card): string {
  let className = `trick-card suit-${card.suit}`;
  if (card.rank === "5" || card.rank === "10" || card.rank === "K") {
    className += " point-card";
  }
  return className;
}

function renderFailedThrowPreview(
  event: FailedThrow,
  snapshot: StateSnapshot,
  viewerSeat: SeatId,
): HTMLElement {
  const seatLabel = seatView(event.actor, viewerSeat).label;
  const preview = el("div", { class: "failed-throw-preview" });
  preview.appendChild(
    el(
      "div",
      { class: "failed-throw-preview__title" },
      `${seatLabel}甩牌失败`,
    ),
  );
  preview.appendChild(
    renderFailedThrowRow(
      "暴露",
      event.attempted_cards,
      false,
      snapshot,
    ),
  );
  preview.appendChild(
    renderFailedThrowRow("捡小", event.forced_cards, true, snapshot),
  );
  return preview;
}

function renderFailedThrowRow(
  label: string,
  cards: Card[],
  forced: boolean,
  snapshot: StateSnapshot,
): HTMLElement {
  const row = el("div", {
    class: forced
      ? "failed-throw-preview__row failed-throw-preview__row--forced"
      : "failed-throw-preview__row",
  });
  row.appendChild(
    el("span", { class: "failed-throw-preview__label" }, label),
  );
  const cardsEl = el("div", { class: "failed-throw-preview__cards" });
  for (
    const card of sortHand(
      cards,
      trumpSuit(snapshot),
      trumpRank(snapshot),
    )
  ) {
    cardsEl.appendChild(renderTrickCard(card));
  }
  row.appendChild(cardsEl);
  return row;
}
