import { parseProcessState, type ProcessState } from "./types.ts";
import { parseTrainingStreamFrame } from "./stream-frame.ts";

export interface WebSocketLocation {
  readonly protocol: string;
  readonly host: string;
}

export interface ProcessStreamHandlers {
  readonly onSnapshot: (snapshot: ProcessState) => void;
  readonly onConnectionChange: (connected: boolean) => void;
  readonly onError: (message: string) => void;
}

export class ProcessStreamClient {
  #socket: WebSocket | null = null;
  #reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  #stopped = true;

  constructor(
    private readonly runDir: () => string | null,
    private readonly handlers: ProcessStreamHandlers,
  ) {}

  connect(): void {
    this.disconnect();
    this.#stopped = false;
    const runDir = this.runDir();
    if (runDir === null) return;
    const socket = new WebSocket(processStreamUrl(runDir));
    this.#socket = socket;
    socket.addEventListener("open", () => {
      if (this.#socket === socket) {
        this.handlers.onConnectionChange(true);
      }
    });
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
        this.handlers.onSnapshot(
          parseProcessState(frame.value),
        );
      } catch (error: unknown) {
        this.handlers.onError(errorText(error));
      }
    });
    socket.addEventListener("close", () => {
      if (this.#socket !== socket) return;
      this.#socket = null;
      this.handlers.onConnectionChange(false);
      if (!this.#stopped) {
        this.#reconnectTimer = setTimeout(
          () => this.connect(),
          1000,
        );
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

export function processStreamUrl(
  runDir: string,
  location: WebSocketLocation = globalThis.location,
): string {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const search = new URLSearchParams({ run_dir: runDir });
  return `${protocol}//${location.host}/ws/training/process?${search}`;
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
