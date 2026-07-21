import { EventStreamConnection } from "./event-source.ts";
import { parseProcessState, type ProcessState } from "./types.ts";

export interface ProcessEventHandlers {
  readonly onSnapshot: (snapshot: ProcessState) => void;
  readonly onConnectionChange: (connected: boolean) => void;
  readonly onError: (message: string) => void;
}

export class ProcessEventStream {
  readonly #connection: EventStreamConnection;

  constructor(
    private readonly runDir: () => string | null,
    private readonly handlers: ProcessEventHandlers,
  ) {
    this.#connection = new EventStreamConnection({
      onConnectionChange: handlers.onConnectionChange,
      onError: handlers.onError,
    });
  }

  connect(): void {
    this.disconnect();
    const runDir = this.runDir();
    if (runDir === null) return;
    this.#connection.connect(processEventUrl(runDir), [
      {
        name: "process",
        receive: (value) =>
          this.handlers.onSnapshot(parseProcessState(value)),
      },
    ]);
  }

  disconnect(): void {
    this.#connection.disconnect();
  }
}

export function processEventUrl(runDir: string): string {
  const search = new URLSearchParams({ run_dir: runDir });
  return `/api/training/events/process?${search}`;
}
