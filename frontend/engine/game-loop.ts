import type { StateSnapshot } from "../core/types.ts";
import type { ServerMessage } from "../core/protocol.ts";
import type { InteractionMode } from "./types.ts";
import type { StateManager } from "../core/state.ts";

/** Error handler callback type. */
type ErrorHandler = (message: string) => void;

/** Reconnecting state provider callback type. */
type ReconnectingProvider = () => boolean;

/**
 * Core game loop orchestrator.
 * Subscribes to WS messages, updates StateManager, triggers re-renders
 * with the computed interactionMode, and handles error messages.
 */
export class GameLoop {
  private stateManager: StateManager;
  private renderFn: (snapshot: StateSnapshot, container: Element, interactionMode: InteractionMode) => void;
  private container: Element;
  private isReconnecting: ReconnectingProvider;
  private onError: ErrorHandler | null;
  private lastError: string | null = null;

  constructor(
    stateManager: StateManager,
    renderFn: (snapshot: StateSnapshot, container: Element, interactionMode: InteractionMode) => void,
    container: Element,
    _humanPlayerIndex?: number,
    isReconnecting?: ReconnectingProvider,
    onError?: ErrorHandler,
  ) {
    this.stateManager = stateManager;
    this.renderFn = renderFn;
    this.container = container;
    this.isReconnecting = isReconnecting ?? (() => false);
    this.onError = onError ?? null;
  }

  /**
   * Handle an incoming server message.
   * - State messages: update StateManager (with seq), handle error field,
   *   compute interactionMode, call renderFn.
   */
  handleMessage(msg: ServerMessage): void {
    if (msg.type === "state") {
      this.stateManager.update(msg.state, msg.seq);

      // Handle error field from server (action rejection feedback)
      if (msg.error) {
        this.lastError = msg.error;
        this.onError?.(msg.error);
      }

      const interactionMode = this.computeInteractionMode(msg.state);
      this.renderFn(msg.state, this.container, interactionMode);
    }
  }

  /**
   * Get the last error message, or null if no error has occurred.
   */
  getLastError(): string | null {
    return this.lastError;
  }

  /**
   * Compute the interaction mode from the state snapshot.
   *
   * Uses state.awaiting_action as the authoritative source for what action
   * the human player should take. This correctly
   * handles:
   * - DEAL_BID: only shows "bid" when awaiting_action="bid" (human's turn)
   * - STIRRING: shows "stir" or "discard" based on awaiting_action
   * - PLAYING: shows "play" when awaiting_action="play"
   * - WAITING: shows "next_round"
   * - Game over: null, because awaiting_action is null
   * - Reconnecting: null (all interaction disabled)
   */
  private computeInteractionMode(state: StateSnapshot): InteractionMode {
    // Disable all interaction while reconnecting
    if (this.isReconnecting()) {
      return null;
    }

    if (state.awaiting_action !== null) {
      switch (state.awaiting_action) {
        case "bid":
          return "bid";
        case "stir":
          return "stir";
        case "discard":
          return "discard";
        case "play":
          return "play";
        case "next_round":
          return "next_round";
        default:
          return null;
      }
    }

    return null;
  }
}
