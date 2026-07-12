import { parseLogMessage, type TrainingLogMessage } from "./types.ts";

export interface TrainingStreamTarget {
  readonly runDir: string;
  readonly window: number;
  readonly eventTypes: readonly string[];
  readonly sessionId: string | null;
}

export interface TrainingStreamHandlers {
  readonly onMessage: (message: TrainingLogMessage) => void;
  readonly onConnectionChange: (connected: boolean) => void;
  readonly onError: (message: string) => void;
}

export interface WebSocketLocation {
  readonly protocol: string;
  readonly host: string;
}

export class TrainingStreamClient {
  private socket: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private stopped = true;

  constructor(
    private readonly target: () => TrainingStreamTarget | null,
    private readonly handlers: TrainingStreamHandlers,
  ) {}

  connect(): void {
    this.stopped = false;
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.socket?.close();
    const target = this.target();
    if (target === null) return;
    const socket = new WebSocket(trainingStreamUrl(target));
    this.socket = socket;
    socket.addEventListener("open", () => {
      if (this.socket !== socket) return;
      this.handlers.onConnectionChange(true);
    });
    socket.addEventListener("message", (event) => {
      if (this.socket !== socket || typeof event.data !== "string") {
        return;
      }
      try {
        this.handlers.onMessage(
          parseLogMessage(JSON.parse(event.data)),
        );
      } catch (error: unknown) {
        this.handlers.onError(errorText(error));
      }
    });
    socket.addEventListener("close", () => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.handlers.onConnectionChange(false);
      if (!this.stopped) {
        this.reconnectTimer = setTimeout(() => this.connect(), 1000);
      }
    });
  }

  disconnect(): void {
    this.stopped = true;
    if (this.reconnectTimer !== null) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
    this.socket?.close();
    this.socket = null;
  }
}

export function trainingStreamUrl(
  target: TrainingStreamTarget,
  location: WebSocketLocation = globalThis.location,
): string {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const search = new URLSearchParams({
    run_dir: target.runDir,
    window: String(target.window),
  });
  for (const eventType of target.eventTypes) {
    search.append("event", eventType);
  }
  if (target.sessionId !== null) {
    search.set("session_id", target.sessionId);
  }
  return `${protocol}//${location.host}/ws/training/logs?${search}`;
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
