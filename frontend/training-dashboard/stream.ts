import {
  parseTrainingStreamMessage,
  type TrainingStreamSnapshot,
} from "./types.ts";

export interface TrainingStreamTarget {
  readonly runDir: string;
  readonly metricSequence: number | null;
  readonly telemetrySequence: number | null;
  readonly logStream: "stdout" | "stderr" | null;
}

export interface WebSocketLocation {
  readonly protocol: string;
  readonly host: string;
}

export interface TrainingStreamHandlers {
  readonly onSnapshot: (snapshot: TrainingStreamSnapshot) => void;
  readonly onConnectionChange: (connected: boolean) => void;
  readonly onError: (message: string) => void;
}

export class TrainingStreamClient {
  private socket: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private generation = 0;
  private stopped = true;

  constructor(
    private readonly target: () => TrainingStreamTarget | null,
    private readonly handlers: TrainingStreamHandlers,
  ) {}

  connect(): void {
    this.stopped = false;
    this.replaceSocket();
  }

  disconnect(): void {
    this.stopped = true;
    this.generation += 1;
    this.clearReconnectTimer();
    this.socket?.close();
    this.socket = null;
  }

  private replaceSocket(): void {
    const target = this.target();
    if (target === null) {
      this.disconnect();
      return;
    }
    this.generation += 1;
    const generation = this.generation;
    this.clearReconnectTimer();
    this.socket?.close();
    const socket = new WebSocket(trainingStreamUrl(target));
    this.socket = socket;

    socket.addEventListener("open", () => {
      if (!this.isCurrent(socket, generation)) return;
      this.reconnectAttempts = 0;
      this.handlers.onConnectionChange(true);
    });
    socket.addEventListener("message", (event) => {
      if (!this.isCurrent(socket, generation)) return;
      this.receive(event.data);
    });
    socket.addEventListener("close", () => {
      if (!this.isCurrent(socket, generation)) return;
      this.socket = null;
      this.handlers.onConnectionChange(false);
      this.scheduleReconnect(generation);
    });
    socket.addEventListener("error", () => {
      // The close event owns reconnection and connection state.
    });
  }

  private receive(data: unknown): void {
    try {
      if (typeof data !== "string") {
        throw new Error("Training stream message must be text");
      }
      const raw: unknown = JSON.parse(data);
      const message = parseTrainingStreamMessage(raw);
      if (message.type === "error") {
        this.handlers.onError(message.message);
      } else {
        this.handlers.onSnapshot(message);
      }
    } catch (error: unknown) {
      this.handlers.onError(errorText(error));
    }
  }

  private scheduleReconnect(generation: number): void {
    if (this.stopped || generation !== this.generation) return;
    const delay = Math.min(
      1000 * 2 ** this.reconnectAttempts,
      10_000,
    );
    this.reconnectAttempts += 1;
    this.reconnectTimer = globalThis.setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.stopped && generation === this.generation) {
        this.replaceSocket();
      }
    }, delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer === null) return;
    globalThis.clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
  }

  private isCurrent(socket: WebSocket, generation: number): boolean {
    return generation === this.generation && socket === this.socket;
  }
}

export function trainingStreamUrl(
  target: TrainingStreamTarget,
  location: WebSocketLocation = globalThis.location,
): string {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const url = new URL(`${protocol}//${location.host}/ws/training`);
  url.searchParams.set("run_dir", target.runDir);
  if (target.metricSequence !== null) {
    url.searchParams.set(
      "metric_sequence",
      String(target.metricSequence),
    );
  }
  if (target.telemetrySequence !== null) {
    url.searchParams.set(
      "telemetry_sequence",
      String(target.telemetrySequence),
    );
  }
  if (target.logStream !== null) {
    url.searchParams.set("log_stream", target.logStream);
  }
  return url.toString();
}

function errorText(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "Training stream failed";
}
