import { EventStreamConnection } from "./event-source.ts";
import {
  parseLogEntry,
  parseStoreReplacement,
  type TrainingLogMessage,
} from "./types.ts";

export interface LogEventTarget {
  readonly runDir: string;
  readonly afterSequence: number;
  readonly storeId: string | null;
}

export interface LogEventHandlers {
  readonly onMessage: (message: TrainingLogMessage) => void;
  readonly onConnectionChange: (connected: boolean) => void;
  readonly onError: (message: string) => void;
}

export class LogEventStream {
  readonly #connection: EventStreamConnection;

  constructor(
    private readonly target: () => LogEventTarget | null,
    private readonly handlers: LogEventHandlers,
  ) {
    this.#connection = new EventStreamConnection({
      onConnectionChange: handlers.onConnectionChange,
      onError: handlers.onError,
    });
  }

  connect(): void {
    this.disconnect();
    const target = this.target();
    if (target === null) return;
    this.#connection.connect(logEventUrl(target), [
      {
        name: "log",
        receive: (value) => {
          this.handlers.onMessage({
            type: "event",
            ...parseLogEntry(value),
          });
        },
      },
      {
        name: "replacement",
        receive: (value) => {
          this.handlers.onMessage({
            type: "replacement",
            ...parseStoreReplacement(value),
          });
        },
      },
    ]);
  }

  disconnect(): void {
    this.#connection.disconnect();
  }
}

export function logEventUrl(target: LogEventTarget): string {
  const search = new URLSearchParams({
    run_dir: target.runDir,
    after_sequence: String(target.afterSequence),
  });
  if (target.storeId !== null) search.set("store_id", target.storeId);
  return `/api/training/events/logs?${search}`;
}
