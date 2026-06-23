import type { ServerMessage } from "../core/protocol.ts";
import type { StateSnapshot } from "../core/types.ts";
import {
  computeBidOptionsFromHints,
  computeBidPriority,
  computeDealBidAction,
  type DealBidActionDecision,
} from "./bid-logic.ts";
import type { BidOption, InteractionMode } from "./types.ts";

export interface BidRenderState {
  bidOptions: BidOption[];
  pendingBidIntent: BidOption | null;
}

export class PendingBidController {
  #pendingBidIntent: BidOption | null = null;
  #visibleBidOptions: BidOption[] = [];
  #pendingBidInFlight = false;

  reset(): void {
    this.#pendingBidIntent = null;
    this.#visibleBidOptions = [];
    this.#pendingBidInFlight = false;
  }

  updateRenderState(
    snapshot: StateSnapshot,
    interactionMode: InteractionMode,
  ): BidRenderState {
    const currentBidOptions = computeBidOptionsFromHints(
      snapshot.action_hints ?? [],
      snapshot.trump_rank,
    );
    if (snapshot.phase !== "DEAL_BID") {
      this.reset();
    } else if (interactionMode === "bid") {
      this.#visibleBidOptions = currentBidOptions;
    }

    this.#visibleBidOptions = filterAllowedBidOptions(
      this.#visibleBidOptions,
      snapshot,
    );
    if (
      !this.#pendingBidInFlight &&
      this.#pendingBidIntent !== null &&
      !containsBidOption(
        this.#visibleBidOptions,
        this.#pendingBidIntent,
      )
    ) {
      this.#pendingBidIntent = null;
    }

    return {
      bidOptions: snapshot.phase === "DEAL_BID"
        ? this.#visibleBidOptions
        : [],
      pendingBidIntent: this.#pendingBidIntent,
    };
  }

  select(option: BidOption): boolean {
    if (
      this.#pendingBidIntent !== null ||
      !containsBidOption(this.#visibleBidOptions, option)
    ) {
      return false;
    }
    this.#pendingBidIntent = option;
    return true;
  }

  computeDealBidAction(seq: number): DealBidActionDecision {
    return computeDealBidAction([], this.#pendingBidIntent, seq);
  }

  markActionSent(decision: DealBidActionDecision): void {
    if (decision.matchedPending) {
      this.#pendingBidInFlight = true;
    }
  }

  acknowledgeMessage(message: ServerMessage): void {
    if (!message.error && this.#pendingBidInFlight) {
      this.#pendingBidInFlight = false;
      this.#pendingBidIntent = null;
    }
  }

  consumeInFlightFailure(): boolean {
    if (!this.#pendingBidInFlight) {
      return false;
    }
    this.#pendingBidIntent = null;
    this.#pendingBidInFlight = false;
    return true;
  }

  get isInFlight(): boolean {
    return this.#pendingBidInFlight;
  }
}

function filterAllowedBidOptions(
  options: BidOption[],
  snapshot: StateSnapshot,
): BidOption[] {
  if (snapshot.phase !== "DEAL_BID") {
    return [];
  }
  const handIds = new Set(snapshot.player_hand.map((card) => card.id));
  const currentBidPriority = snapshot.bid_winner === null
    ? 0
    : computeBidPriority(
      snapshot.bid_winner.cards,
      snapshot.trump_rank,
    );
  return options.filter((option) =>
    option.priority > currentBidPriority &&
    option.cardIds.every((cardId) => handIds.has(cardId))
  );
}

function containsBidOption(
  options: BidOption[],
  target: BidOption,
): boolean {
  const targetKey = bidOptionKey(target);
  return options.some((option) => bidOptionKey(option) === targetKey);
}

function bidOptionKey(option: BidOption): string {
  return [...option.cardIds].sort().join(",");
}
