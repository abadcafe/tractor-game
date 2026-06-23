import type {
  CompletedTrick,
  FailedThrow,
  StateSnapshot,
} from "../core/types.ts";

export class TrickPreviewController {
  #hasSeenState = false;
  #lastSeenCompletedTrickKey: string | null = null;
  #previousTrickPreview: CompletedTrick | null = null;
  #previousTrickPreviewTimer: ReturnType<typeof setTimeout> | null =
    null;
  #failedThrowPreview: FailedThrow | null = null;
  #failedThrowPreviewTimer: ReturnType<typeof setTimeout> | null = null;
  #lastFailedThrowKey: string | null = null;

  constructor(
    private readonly previousTrickPreviewMs: number,
    private readonly failedThrowPreviewMs: number,
    private readonly onChange: () => void,
  ) {}

  get previousTrickPreview(): CompletedTrick | null {
    return this.#previousTrickPreview;
  }

  get failedThrowPreview(): FailedThrow | null {
    return this.#failedThrowPreview;
  }

  reset(): void {
    this.#hasSeenState = false;
    this.#lastSeenCompletedTrickKey = null;
    this.clearPreviousTrickPreview();
    this.clearFailedThrowPreview();
  }

  showPreviousTrickPreview(
    trick: CompletedTrick,
    renderNow: boolean,
  ): void {
    this.clearFailedThrowPreview();
    this.#previousTrickPreview = trick;
    if (this.#previousTrickPreviewTimer !== null) {
      clearTimeout(this.#previousTrickPreviewTimer);
    }
    this.#previousTrickPreviewTimer = setTimeout(() => {
      this.#previousTrickPreview = null;
      this.#previousTrickPreviewTimer = null;
      this.onChange();
    }, this.previousTrickPreviewMs);
    if (renderNow) {
      this.onChange();
    }
  }

  update(snapshot: StateSnapshot): void {
    this.updatePreviousTrickPreview(snapshot);
    this.updateFailedThrowPreview(snapshot);
  }

  private clearPreviousTrickPreview(): void {
    this.#previousTrickPreview = null;
    if (this.#previousTrickPreviewTimer !== null) {
      clearTimeout(this.#previousTrickPreviewTimer);
      this.#previousTrickPreviewTimer = null;
    }
  }

  private clearFailedThrowPreview(): void {
    this.#failedThrowPreview = null;
    this.#lastFailedThrowKey = null;
    if (this.#failedThrowPreviewTimer !== null) {
      clearTimeout(this.#failedThrowPreviewTimer);
      this.#failedThrowPreviewTimer = null;
    }
  }

  private updateFailedThrowPreview(snapshot: StateSnapshot): void {
    if (snapshot.phase !== "PLAYING") {
      this.clearFailedThrowPreview();
      return;
    }

    const event = snapshot.failed_throw;
    if (event === null) {
      return;
    }

    const key = failedThrowKey(snapshot, event);
    if (key === this.#lastFailedThrowKey) {
      return;
    }

    this.#lastFailedThrowKey = key;
    this.#failedThrowPreview = event;
    this.clearPreviousTrickPreview();
    if (this.#failedThrowPreviewTimer !== null) {
      clearTimeout(this.#failedThrowPreviewTimer);
    }
    this.#failedThrowPreviewTimer = setTimeout(() => {
      if (this.#lastFailedThrowKey === key) {
        this.#failedThrowPreview = null;
        this.#failedThrowPreviewTimer = null;
        this.onChange();
      }
    }, this.failedThrowPreviewMs);
  }

  private updatePreviousTrickPreview(snapshot: StateSnapshot): void {
    const trickKey = completedTrickKey(snapshot.last_completed_trick);
    if (!this.#hasSeenState) {
      this.#hasSeenState = true;
      this.#lastSeenCompletedTrickKey = trickKey;
      return;
    }
    if (trickKey === this.#lastSeenCompletedTrickKey) {
      return;
    }

    this.#lastSeenCompletedTrickKey = trickKey;
    if (snapshot.last_completed_trick === null) {
      this.clearPreviousTrickPreview();
      this.clearFailedThrowPreview();
      return;
    }
    this.showPreviousTrickPreview(snapshot.last_completed_trick, false);
  }
}

function completedTrickKey(
  trick: CompletedTrick | null,
): string | null {
  if (trick === null) {
    return null;
  }
  const slotParts = trick.slots.map((slot) =>
    `${slot.player}:${slot.cards.map((card) => card.id).join(",")}`
  );
  return [
    trick.lead_player,
    trick.winner,
    trick.points,
    ...slotParts,
  ].join("|");
}

function failedThrowKey(
  snapshot: StateSnapshot,
  event: FailedThrow,
): string {
  const attemptedIds = event.attempted_cards.map((card) => card.id)
    .join(",");
  const forcedIds = event.forced_cards.map((card) => card.id).join(
    ",",
  );
  return [
    completedTrickKey(snapshot.last_completed_trick) ?? "none",
    event.player,
    attemptedIds,
    forcedIds,
  ].join("|");
}
