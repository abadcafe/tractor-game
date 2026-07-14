import { parseMetrics, type TrainingMetrics } from "./types.ts";
import type { WebSocketLocation } from "./process.ts";
import { parseTrainingStreamFrame } from "./stream-frame.ts";

export interface MetricsStreamTarget {
  readonly runDir: string;
  readonly updateLimit: number;
  readonly seriesPoints: number;
}

export interface MetricsStreamHandlers {
  readonly onSnapshot: (value: TrainingMetrics) => void;
  readonly onError: (message: string) => void;
}

export class MetricsSnapshotStream {
  #socket: WebSocket | null = null;
  #reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  #stopped = true;

  constructor(
    private readonly target: () => MetricsStreamTarget | null,
    private readonly handlers: MetricsStreamHandlers,
  ) {}

  connect(): void {
    this.disconnect();
    this.#stopped = false;
    const target = this.target();
    if (target === null) return;
    const socket = new WebSocket(metricsStreamUrl(target));
    this.#socket = socket;
    socket.addEventListener("message", (event) => {
      if (this.#socket !== socket || typeof event.data !== "string") {
        return;
      }
      try {
        const frame = parseTrainingStreamFrame(JSON.parse(event.data));
        if (frame.type === "rejected") {
          this.#stopped = true;
          this.handlers.onError(frame.error);
          return;
        }
        this.handlers.onSnapshot(parseMetrics(frame.value));
      } catch (error: unknown) {
        this.handlers.onError(errorText(error));
      }
    });
    socket.addEventListener("close", () => {
      if (this.#socket !== socket) return;
      this.#socket = null;
      if (!this.#stopped) {
        this.#reconnectTimer = setTimeout(() => this.connect(), 1000);
      }
    });
  }

  disconnect(): void {
    this.#stopped = true;
    if (this.#reconnectTimer !== null) {
      clearTimeout(this.#reconnectTimer);
    }
    this.#reconnectTimer = null;
    this.#socket?.close();
    this.#socket = null;
  }
}

export function metricsStreamUrl(
  target: MetricsStreamTarget,
  location: WebSocketLocation = globalThis.location,
): string {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const search = new URLSearchParams({
    run_dir: target.runDir,
    update_limit: String(target.updateLimit),
    series_points: String(target.seriesPoints),
  });
  return `${protocol}//${location.host}/ws/training/metrics?${search}`;
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
