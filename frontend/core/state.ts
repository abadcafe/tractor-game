import type { StateSnapshot } from "./types.ts";

/**
 * Centralized game state storage with change notification.
 * Uses a singleton pattern per task requirement, but each StateManager instance
 * manages its own state so tests can create independent instances.
 */
export class StateManager {
  private _snapshot: StateSnapshot | null = null;
  private _listeners: Set<(snap: StateSnapshot) => void> = new Set();

  /** Stores the snapshot, then notifies all listeners. */
  update(snapshot: StateSnapshot): void {
    this._snapshot = snapshot;
    for (const listener of this._listeners) {
      listener(snapshot);
    }
  }

  /** Sets snapshot to null without notifying listeners. */
  reset(): void {
    this._snapshot = null;
  }

  /** Returns the current snapshot, or null if no state has been set (or after reset). */
  get(): StateSnapshot | null {
    return this._snapshot;
  }

  /**
   * Subscribes to state changes. Returns an unsubscribe function.
   * The listener is called with the new snapshot each time update() is called.
   */
  onChange(fn: (snap: StateSnapshot) => void): () => void {
    this._listeners.add(fn);
    return () => {
      this._listeners.delete(fn);
    };
  }
}
