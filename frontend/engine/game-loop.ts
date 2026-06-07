import type { StateSnapshot, ServerMessage, InteractionMode } from "../core/types.ts";
import type { StateManager } from "../core/state.ts";
import { HUMAN_PLAYER_INDEX } from "../config.ts";

/**
 * Core game loop orchestrator.
 * Subscribes to WS messages, updates StateManager, triggers re-renders
 * with the computed interactionMode, and handles error messages.
 */
export class GameLoop {
  private stateManager: StateManager;
  private renderFn: (snapshot: StateSnapshot, container: Element, interactionMode: InteractionMode) => void;
  private container: Element;
  private lastError: string | null = null;

  constructor(
    stateManager: StateManager,
    renderFn: (snapshot: StateSnapshot, container: Element, interactionMode: InteractionMode) => void,
    container: Element,
  ) {
    this.stateManager = stateManager;
    this.renderFn = renderFn;
    this.container = container;
  }

  /**
   * Handle an incoming server message.
   * - Error messages: store the error, do not update state or re-render.
   * - State messages: update StateManager, compute interactionMode, call renderFn.
   */
  handleMessage(msg: ServerMessage): void {
    if (msg.type === "error") {
      this.lastError = msg.message;
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
   * - If phase is "DEAL_BID" -> always "bid" (show bidding panel)
   * - Else if awaiting is not null and current_player is human -> map awaiting to interaction mode
   * - Otherwise -> null (spectator mode)
   */
  private computeInteractionMode(state: StateSnapshot, awaiting: string | null): InteractionMode {
    if (state.phase === "DEAL_BID") {
      return "bid";
    }

    if (awaiting !== null && state.current_player === HUMAN_PLAYER_INDEX) {
      switch (awaiting) {
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
