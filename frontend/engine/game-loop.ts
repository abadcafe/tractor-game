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
  private humanPlayerIndex: number;
  private lastError: string | null = null;

  constructor(
    stateManager: StateManager,
    renderFn: (snapshot: StateSnapshot, container: Element, interactionMode: InteractionMode) => void,
    container: Element,
    humanPlayerIndex: number,
    isReconnecting?: ReconnectingProvider,
    onError?: ErrorHandler,
  ) {
    this.stateManager = stateManager;
    this.renderFn = renderFn;
    this.container = container;
    this.humanPlayerIndex = humanPlayerIndex;
    this.isReconnecting = isReconnecting ?? (() => false);
    this.onError = onError ?? null;
  }

  /**
   * Handle an incoming server message.
   * - Error messages: store the error, do not update state or re-render.
   * - State messages: update StateManager, compute interactionMode, call renderFn.
   */
  handleMessage(msg: ServerMessage): void {
    if (msg.type === "error") {
      this.lastError = msg.message;
      this.onError?.(msg.message);
      return;
    }

    if (msg.type === "state") {
      this.stateManager.update(msg.state);
      const interactionMode = this.computeInteractionMode(msg.state, msg.awaiting);
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
   * Compute the interaction mode from the state snapshot and awaiting value.
   *
   * Rules:
   * - If reconnecting -> null (disable all interaction)
   * - If phase is "DEAL_BID" -> always "bid" (show bidding panel)
   * - If phase is "COMPLETE" -> "next_round" (show scoring + next round button)
   * - If phase is "GAME_OVER" -> "next_round" (show scoring overlay without button)
   * - Else if awaiting is not null and current_player is human -> map awaiting to interaction mode
   * - Otherwise -> null (spectator mode)
   */
  private computeInteractionMode(state: StateSnapshot, awaiting: string | null): InteractionMode {
    // Disable all interaction while reconnecting
    if (this.isReconnecting()) {
      return null;
    }

    if (state.phase === "DEAL_BID") {
      return "bid";
    }

    if (state.phase === "COMPLETE") {
      return "next_round";
    }

    if (state.phase === "GAME_OVER") {
      return "next_round";
    }

    if (awaiting !== null && state.current_player === this.humanPlayerIndex) {
      switch (awaiting) {
        case "stir":
          return "stir";
        case "discard":
          return "discard";
        case "play":
          return "play";
        default:
          return null;
      }
    }

    return null;
  }
}
