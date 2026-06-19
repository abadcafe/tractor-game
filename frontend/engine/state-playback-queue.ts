/** Plays server state messages in order at a readable UI cadence. */
export class StatePlaybackQueue<T> {
  private pending: T[] = [];
  private timer: ReturnType<typeof setTimeout> | null = null;
  private lastRenderedAt = 0;

  constructor(
    private readonly onMessage: (message: T) => void,
    private readonly options: {
      minFrameMs?: number;
      minFrameMsForMessage?: (message: T) => number;
      onCaughtUpChange?: (caughtUp: boolean) => void;
      now?: () => number;
    } = {},
  ) {}

  enqueue(message: T): void {
    this.pending.push(message);
    this.notifyCaughtUp();
    this.schedule();
  }

  clear(): void {
    this.pending = [];
    this.lastRenderedAt = 0;
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.notifyCaughtUp();
  }

  isCaughtUp(): boolean {
    return this.pending.length === 0;
  }

  private schedule(): void {
    if (this.timer !== null || this.pending.length === 0) return;

    const next = this.pending[0];
    if (next === undefined) return;

    const now = this.options.now?.() ?? performance.now();
    const minFrameMs = this.minFrameMsFor(next);
    const elapsed = this.lastRenderedAt === 0
      ? minFrameMs
      : now - this.lastRenderedAt;
    const delayMs = Math.max(0, minFrameMs - elapsed);

    this.timer = setTimeout(() => {
      this.timer = null;
      const next = this.pending.shift();
      if (next === undefined) {
        this.notifyCaughtUp();
        return;
      }

      this.lastRenderedAt = this.options.now?.() ?? performance.now();
      this.notifyCaughtUp();
      this.onMessage(next);
      this.schedule();
    }, delayMs);
  }

  private notifyCaughtUp(): void {
    this.options.onCaughtUpChange?.(this.isCaughtUp());
  }

  private minFrameMsFor(message: T): number {
    return this.options.minFrameMsForMessage?.(message) ??
      this.options.minFrameMs ??
      500;
  }
}
