import { recordValue } from "../browser/json.ts";

export interface EventStreamListener {
  readonly name: string;
  readonly receive: (
    value: unknown,
    lastEventId: string,
  ) => void;
}

export interface EventStreamHandlers {
  readonly onConnectionChange?: (connected: boolean) => void;
  readonly onError: (message: string) => void;
}

export class EventStreamConnection {
  #source: EventSource | null = null;

  constructor(private readonly handlers: EventStreamHandlers) {}

  connect(
    url: string,
    listeners: readonly EventStreamListener[],
  ): void {
    this.disconnect();
    const source = new EventSource(url);
    this.#source = source;
    source.addEventListener("open", () => {
      if (this.#source === source) {
        this.handlers.onConnectionChange?.(true);
      }
    });
    source.addEventListener("error", () => {
      if (this.#source === source) {
        this.handlers.onConnectionChange?.(false);
      }
    });
    source.addEventListener("rejected", (event) => {
      if (this.#source !== source) return;
      try {
        const message = messageEvent(event);
        const value: unknown = JSON.parse(message.data);
        const error = parseRejection(value);
        this.#close(source);
        this.handlers.onError(error);
      } catch (error: unknown) {
        this.#fail(source, error);
      }
    });
    for (const listener of listeners) {
      source.addEventListener(listener.name, (event) => {
        if (this.#source !== source) return;
        try {
          const message = messageEvent(event);
          const value: unknown = JSON.parse(message.data);
          listener.receive(value, message.lastEventId);
        } catch (error: unknown) {
          this.#fail(source, error);
        }
      });
    }
  }

  disconnect(): void {
    if (this.#source !== null) this.#close(this.#source);
  }

  #fail(source: EventSource, error: unknown): void {
    this.#close(source);
    this.handlers.onError(errorText(error));
  }

  #close(source: EventSource): void {
    source.close();
    if (this.#source === source) this.#source = null;
  }
}

function messageEvent(event: Event): MessageEvent<string> {
  if (
    !(event instanceof MessageEvent) || typeof event.data !== "string"
  ) {
    throw new Error("Invalid server event");
  }
  return event;
}

function parseRejection(value: unknown): string {
  const record = recordValue(value);
  if (record === null || Object.keys(record).length !== 1) {
    throw new Error("Invalid stream rejection");
  }
  if (typeof record.error !== "string" || record.error.trim() === "") {
    throw new Error("Invalid stream rejection error");
  }
  return record.error;
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
