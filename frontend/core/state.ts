import type { StateSnapshot } from "./types.ts";

/**
 * Centralized game state storage.
 * Each StateManager instance manages its own state so tests can create independent instances.
 */
export class StateManager {
  private _snapshot: StateSnapshot | null = null;

  /** Stores the snapshot. */
  update(snapshot: StateSnapshot): void {
    this._snapshot = snapshot;
  }

  /** Sets snapshot to null. */
  reset(): void {
    this._snapshot = null;
  }

  /** Returns the current snapshot, or null if no state has been set (or after reset). */
  get(): StateSnapshot | null {
    return this._snapshot;
  }
}
