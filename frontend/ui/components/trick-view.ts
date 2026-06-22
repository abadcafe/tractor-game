import type {
  Card,
  CompletedTrick,
  FailedThrow,
  StateSnapshot,
  TrickSlot,
} from "../../core/types.ts";
import { el } from "../dom.ts";
import { cardDisplay, sortHand, suitSymbol } from "../../core/card.ts";
import { SEAT_MAP } from "../../config.ts";

/**
 * Render the current trick area showing played cards with player labels,
 * previous-trick preview, and failed-throw preview.
 */
export function renderTrickView(
  snapshot: StateSnapshot,
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
    ? renderCompletedTrickGrid(previousTrickPreview, snapshot)
    : showingScoringTrick
    ? renderCompletedTrickGrid(scoringTrick, snapshot)
    : renderCurrentTrickGrid(snapshot);
  if (grid !== null) {
    trickView.appendChild(grid);
  }
  if (showingFailedThrow) {
    trickView.appendChild(
      renderFailedThrowPreview(failedThrowPreview, snapshot),
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
): HTMLElement | null {
  if (!snapshot.trick) {
    return null;
  }
  const grid = el("div", { class: "trick-grid" });
  const slotsByPlayer = new Map(
    snapshot.trick.slots.map((slot) => [slot.player, slot]),
  );

  for (const player of [0, 1, 2, 3]) {
    const direction = SEAT_MAP[player]?.direction ?? "north";
    const isLead = player === snapshot.trick.lead_player;
    const isCurrent = player === snapshot.trick.current_player;
    const slot = slotsByPlayer.get(player);

    let slotClass = `${
      slot ? "trick-slot" : "trick-placeholder-slot"
    } trick-slot-${direction}`;
    if (isLead) slotClass += " lead";
    if (isCurrent && !isLead) slotClass += " current";
    if (!slot) slotClass += " empty";

    grid.appendChild(
      renderTrickSlot(player, slot, slotClass, isLead, snapshot),
    );
  }

  return grid;
}

function renderCompletedTrickGrid(
  trick: CompletedTrick,
  snapshot: StateSnapshot,
): HTMLElement {
  const grid = el("div", { class: "trick-grid trick-grid--previous" });
  const slotsByPlayer = new Map(
    trick.slots.map((slot) => [slot.player, slot]),
  );

  for (const player of [0, 1, 2, 3]) {
    const direction = SEAT_MAP[player]?.direction ?? "north";
    const slot = slotsByPlayer.get(player);
    let slotClass = `trick-slot trick-slot-${direction}`;
    const isLead = player === trick.lead_player;
    if (isLead) slotClass += " lead";
    if (player === trick.winner) slotClass += " winner";
    grid.appendChild(
      renderTrickSlot(player, slot, slotClass, isLead, snapshot),
    );
  }

  return grid;
}

function renderTrickSlot(
  player: number,
  slot: TrickSlot | undefined,
  slotClass: string,
  isLead: boolean,
  snapshot: StateSnapshot,
): HTMLElement {
  const slotEl = el("div", { class: slotClass });
  const seatInfo = SEAT_MAP[player];
  if (seatInfo && slot && slot.cards.length > 0) {
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
  return sortHand(cards, snapshot.trump_suit, snapshot.trump_rank);
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
): HTMLElement {
  const seatInfo = SEAT_MAP[event.player];
  const playerLabel = seatInfo?.label ?? `玩家 ${event.player}`;
  const preview = el("div", { class: "failed-throw-preview" });
  preview.appendChild(
    el(
      "div",
      { class: "failed-throw-preview__title" },
      `${playerLabel}甩牌失败`,
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
  for (const card of sortHand(
    cards,
    snapshot.trump_suit,
    snapshot.trump_rank,
  )) {
    cardsEl.appendChild(renderTrickCard(card));
  }
  row.appendChild(cardsEl);
  return row;
}
