import { EventStreamConnection } from "./event-source.ts";
import {
  type CheckpointStreamMessage,
  parseCheckpointCursor,
} from "./types.ts";

export interface CheckpointEventHandlers {
  readonly onMessage: (value: CheckpointStreamMessage) => void;
  readonly onError: (message: string) => void;
}

export class CheckpointEventStream {
  readonly #connection: EventStreamConnection;

  constructor(
    private readonly runDir: () => string | null,
    private readonly storeId: () => string | null,
    private readonly handlers: CheckpointEventHandlers,
  ) {
    this.#connection = new EventStreamConnection({
      onError: handlers.onError,
    });
  }

  connect(): void {
    this.disconnect();
    const runDir = this.runDir();
    if (runDir === null) return;
    this.#connection.connect(
      checkpointEventUrl(runDir, this.storeId()),
      [
        {
          name: "invalidation",
          receive: (value) =>
            this.handlers.onMessage({
              type: "invalidation",
              ...parseCheckpointCursor(value),
            }),
        },
        {
          name: "replacement",
          receive: (value) =>
            this.handlers.onMessage({
              type: "replacement",
              ...parseCheckpointCursor(value),
            }),
        },
      ],
    );
  }

  disconnect(): void {
    this.#connection.disconnect();
  }
}

export function checkpointEventUrl(
  runDir: string,
  storeId: string | null,
): string {
  const search = new URLSearchParams({ run_dir: runDir });
  if (storeId !== null) search.set("store_id", storeId);
  return `/api/training/events/checkpoints?${search}`;
}
