import {
  type CheckpointStreamMessage,
  parseCheckpointStreamMessage,
} from "./types.ts";
import type { WebSocketLocation } from "./process.ts";

export interface CheckpointStreamHandlers {
  readonly onMessage: (value: CheckpointStreamMessage) => void;
  readonly onError: (message: string) => void;
}

export class CheckpointInvalidationStream {
  #socket: WebSocket | null = null;
  #reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  #stopped = true;

  constructor(
    private readonly runDir: () => string | null,
    private readonly storeId: () => string | null,
    private readonly handlers: CheckpointStreamHandlers,
  ) {}

  connect(): void {
    this.disconnect();
    this.#stopped = false;
    const runDir = this.runDir();
    if (runDir === null) return;
    const socket = new WebSocket(
      checkpointStreamUrl(runDir, this.storeId()),
    );
    this.#socket = socket;
    socket.addEventListener("message", (event) => {
      if (this.#socket !== socket || typeof event.data !== "string") {
        return;
      }
      try {
        this.handlers.onMessage(
          parseCheckpointStreamMessage(JSON.parse(event.data)),
        );
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

export function checkpointStreamUrl(
  runDir: string,
  storeId: string | null,
  location: WebSocketLocation = globalThis.location,
): string {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const search = new URLSearchParams({ run_dir: runDir });
  if (storeId !== null) search.set("store_id", storeId);
  return `${protocol}//${location.host}/ws/training/checkpoints?${search}`;
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
