import type { StateSnapshot } from "./types.ts";

/**
 * Centralized game state storage.
 * Each StateManager instance manages its own state so tests can create independent instances.
 * Also tracks the latest server seq for outgoing actions.
 */
export class StateManager {
  private _snapshot: StateSnapshot | null = null;
  private _seq: number = 0;

  /** Stores the snapshot and seq. */
  update(snapshot: StateSnapshot, seq: number): void {
    this._snapshot = snapshot;
    this._seq = seq;
  }

  /** Sets snapshot to null and resets seq. */
  reset(): void {
    this._snapshot = null;
    this._seq = 0;
  }

  /** Returns the current snapshot, or null if no state has been set (or after reset). */
  get(): StateSnapshot | null {
    return this._snapshot;
  }

  /** Returns the latest server seq number. */
  get seq(): number {
    return this._seq;
  }
}
