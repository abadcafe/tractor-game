import { fetchLogPage } from "./api.ts";
import {
  TrainingStreamClient,
  type TrainingStreamTarget,
} from "./stream.ts";
import type {
  TrainingEvent,
  TrainingLogEntry,
  TrainingLogMessage,
} from "./types.ts";

const PAGE_SIZE = 200;
const DOM_LIMIT = 1000;
const MEMORY_LIMIT = 5000;

export interface LogsDomainCallbacks {
  readonly reportError: (message: string) => void;
  readonly clearError: () => void;
}

export class LogsDomain {
  #entries: readonly TrainingLogEntry[] = [];
  #storeId: string | null = null;
  #nextBefore: number | null = null;
  #follow = true;
  #loadingOlder = false;
  #generation = 0;
  #renderFrame: number | null = null;
  #rendering = false;
  readonly #stream = new TrainingStreamClient(
    () => this.#streamTarget(),
    {
      onMessage: (message) => this.#receive(message),
      onConnectionChange: (connected) => {
        if (connected) this.callbacks.clearError();
      },
      onError: (message) => this.callbacks.reportError(message),
    },
  );

  constructor(
    private readonly runDir: () => string,
    private readonly isActive: () => boolean,
    private readonly callbacks: LogsDomainCallbacks,
  ) {
    element("log-content", HTMLElement).addEventListener(
      "scroll",
      () => {
        if (this.#rendering) return;
        const target = element("log-content", HTMLElement);
        this.#setFollow(isAtLogBottom(
          target.scrollTop,
          target.clientHeight,
          target.scrollHeight,
        ));
      },
    );
    element("load-older", HTMLButtonElement).addEventListener(
      "click",
      () => void this.loadOlder(),
    );
    element("toggle-follow", HTMLButtonElement).addEventListener(
      "click",
      () => {
        this.#setFollow(!this.#follow);
        if (this.#follow) this.#scheduleRender();
      },
    );
  }

  activate(): void {
    if (this.#entries.length === 0) void this.refresh();
    else this.#stream.connect();
  }

  deactivate(): void {
    this.#stream.disconnect();
  }

  reset(): void {
    this.#generation += 1;
    this.#stream.disconnect();
    this.#entries = [];
    this.#storeId = null;
    this.#nextBefore = null;
    if (this.#renderFrame !== null) {
      cancelAnimationFrame(this.#renderFrame);
    }
    this.#renderFrame = null;
    this.callbacks.clearError();
    this.render();
  }

  async refresh(): Promise<void> {
    const runDir = this.runDir();
    if (runDir === "") return;
    const generation = this.#generation;
    this.#stream.disconnect();
    try {
      const page = await fetchLogPage(runDir, null, PAGE_SIZE);
      if (generation !== this.#generation || runDir !== this.runDir()) {
        return;
      }
      this.#entries = sortedUnique(page.events);
      this.#storeId = page.store_id;
      this.#nextBefore = page.next_before_sequence;
      this.callbacks.clearError();
      this.render();
      if (this.isActive()) this.#stream.connect();
    } catch (error: unknown) {
      if (generation === this.#generation && runDir === this.runDir()) {
        this.callbacks.reportError(errorText(error));
      }
    }
  }

  async loadOlder(): Promise<void> {
    if (this.#loadingOlder || this.#nextBefore === null) return;
    const runDir = this.runDir();
    const generation = this.#generation;
    const cursor = this.#nextBefore;
    this.#loadingOlder = true;
    this.#renderControls();
    try {
      const page = await fetchLogPage(runDir, cursor, PAGE_SIZE);
      if (generation !== this.#generation || runDir !== this.runDir()) {
        return;
      }
      if (page.store_id !== this.#storeId) {
        this.#entries = sortedUnique(page.events);
        this.#storeId = page.store_id;
      } else {
        this.#entries = sortedUnique([
          ...page.events,
          ...this.#entries,
        ]).slice(-MEMORY_LIMIT);
      }
      this.#nextBefore = page.next_before_sequence;
      this.callbacks.clearError();
      this.render(false);
    } catch (error: unknown) {
      if (generation === this.#generation && runDir === this.runDir()) {
        this.callbacks.reportError(errorText(error));
      }
    } finally {
      this.#loadingOlder = false;
      this.#renderControls();
    }
  }

  render(scrollToEnd = this.#follow): void {
    const target = element("log-content", HTMLElement);
    const previousScroll = target.scrollTop;
    const visible = this.#follow
      ? this.#entries.slice(-DOM_LIMIT)
      : this.#entries.slice(0, DOM_LIMIT);
    this.#rendering = true;
    try {
      target.replaceChildren(
        ...visible.map(({ sequence, event }) =>
          logRow(sequence, event)
        ),
      );
      this.#renderControls();
      if (scrollToEnd) target.scrollTop = target.scrollHeight;
      else target.scrollTop = previousScroll;
    } finally {
      this.#rendering = false;
    }
  }

  #streamTarget(): TrainingStreamTarget | null {
    if (this.runDir() === "" || !this.isActive()) return null;
    return {
      runDir: this.runDir(),
      afterSequence: this.#entries.at(-1)?.sequence ?? 0,
      storeId: this.#storeId,
    };
  }

  #receive(message: TrainingLogMessage): void {
    if (message.type === "replacement") {
      this.#generation += 1;
      this.#storeId = message.store_id;
      this.#entries = [];
      this.#nextBefore = null;
      this.render();
      void this.refresh();
      return;
    }
    if (
      this.#entries.some((item) => item.sequence === message.sequence)
    ) {
      return;
    }
    this.#entries = sortedUnique([...this.#entries, message]).slice(
      -MEMORY_LIMIT,
    );
    if (this.#follow) this.#scheduleRender();
    else this.#renderControls();
  }

  #scheduleRender(): void {
    if (this.#renderFrame !== null) return;
    this.#renderFrame = requestAnimationFrame(() => {
      this.#renderFrame = null;
      this.render();
    });
  }

  #renderControls(): void {
    element("log-count", HTMLElement).textContent = `${
      this.#entries.length.toLocaleString("en-US")
    } events`;
    const older = element("load-older", HTMLButtonElement);
    older.disabled = this.#loadingOlder || this.#nextBefore === null;
    older.textContent = this.#loadingOlder ? "Loading…" : "Load older";
  }

  #setFollow(value: boolean): void {
    this.#follow = value;
    element("toggle-follow", HTMLButtonElement).textContent =
      `Auto-follow: ${value ? "on" : "off"}`;
  }
}

export function isAtLogBottom(
  scrollTop: number,
  clientHeight: number,
  scrollHeight: number,
  threshold = 8,
): boolean {
  return scrollHeight - scrollTop - clientHeight <= threshold;
}

function sortedUnique(
  entries: readonly TrainingLogEntry[],
): readonly TrainingLogEntry[] {
  const values = new Map<number, TrainingLogEntry>();
  for (const entry of entries) values.set(entry.sequence, entry);
  return [...values.values()].sort((a, b) => a.sequence - b.sequence);
}

function logRow(sequence: number, event: TrainingEvent): HTMLElement {
  const details = document.createElement("details");
  details.className = event.error === undefined
    ? "log-row"
    : "log-row failed";
  const heading = document.createElement("summary");
  const time = document.createElement("time");
  time.textContent = new Date(event.recorded_at_ms).toLocaleTimeString(
    "en-GB",
  );
  const state = document.createElement("span");
  state.className = "log-state";
  state.textContent = event.error === undefined ? "OK" : "ERROR";
  const name = document.createElement("strong");
  name.textContent = event.event;
  const cursor = document.createElement("code");
  cursor.textContent = `#${sequence}`;
  heading.append(time, state, name, cursor);
  const body = document.createElement("pre");
  body.textContent = JSON.stringify(event, null, 2);
  details.append(heading, body);
  return details;
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function element<T extends HTMLElement>(
  id: string,
  constructor: { new (): T },
): T {
  const value = document.getElementById(id);
  if (!(value instanceof constructor)) {
    throw new Error(`Missing element: ${id}`);
  }
  return value;
}
