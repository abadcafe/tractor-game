import { fetchProcess } from "./api.ts";
import { ProcessController, ProcessStreamClient } from "./process.ts";
import type { ProcessEnvelope, ProcessSnapshot } from "./types.ts";

export interface ProcessDomainCallbacks {
  readonly reportError: (message: string) => void;
  readonly clearError: () => void;
  readonly connectionChanged: (connected: boolean) => void;
}

export interface ProcessOperations {
  readonly initializing: boolean;
  readonly resuming: boolean;
  readonly stopping: boolean;
}

export class ProcessDomain {
  #envelope: ProcessEnvelope | null = null;
  #generation = 0;
  #operations: ProcessOperations = {
    initializing: false,
    resuming: false,
    stopping: false,
  };
  readonly #controller = new ProcessController((value) => {
    this.#envelope = value;
    this.callbacks.clearError();
    this.render();
  });
  readonly #stream = new ProcessStreamClient(
    () => this.runDir() || null,
    {
      onSnapshot: (value) => this.#controller.apply(value),
      onConnectionChange: (connected) =>
        this.callbacks.connectionChanged(connected),
      onError: (message) => this.callbacks.reportError(message),
    },
  );

  constructor(
    private readonly runDir: () => string,
    private readonly callbacks: ProcessDomainCallbacks,
  ) {
    setInterval(() => this.#renderUptime(), 1000);
  }

  get process(): ProcessSnapshot | null {
    return this.#envelope?.process ?? null;
  }

  connect(): void {
    this.#stream.connect();
  }

  reset(): void {
    this.#generation += 1;
    this.#controller.reset();
    this.#envelope = null;
    this.render();
  }

  apply(value: ProcessEnvelope): void {
    this.#controller.apply(value);
  }

  setOperations(value: ProcessOperations): void {
    this.#operations = value;
    this.render();
  }

  async refresh(): Promise<void> {
    const runDir = this.runDir();
    if (runDir === "") return;
    const generation = this.#generation;
    try {
      const value = await fetchProcess(runDir);
      if (generation !== this.#generation || runDir !== this.runDir()) {
        return;
      }
      this.apply(value);
    } catch (error: unknown) {
      if (generation === this.#generation && runDir === this.runDir()) {
        this.callbacks.reportError(errorText(error));
      }
    }
  }

  render(): void {
    const process = this.process;
    const caption = element("run-caption", HTMLElement);
    caption.textContent = this.runDir();
    caption.title = this.runDir();
    const presence = element("process-presence", HTMLElement);
    presence.textContent = process === null
      ? "STOPPED"
      : process.command === "initialize"
      ? "INITIALIZING"
      : process.ready
      ? "RUNNING"
      : "STARTING";
    presence.className = process === null
      ? "badge neutral"
      : "badge running";
    replaceWithRows(element("process-details", HTMLElement), [
      ["Command", process?.command ?? "-", "plain"],
      [
        "Readiness",
        process === null
          ? "-"
          : process.ready
          ? "Ready"
          : process.command === "initialize"
          ? "Initializing"
          : "Starting",
        "plain",
      ],
      ["PID", process === null ? "-" : String(process.pid), "plain"],
      ["Started", formatTime(process?.started_at_ms ?? null), "plain"],
      [
        "Uptime",
        process === null ? "-" : formatUptime(process),
        "plain",
      ],
      ["Kernel state", process?.kernel_state ?? "-", "plain"],
      [
        "Start ticks",
        process === null ? "-" : String(process.start_ticks),
        "plain",
      ],
      ["Executable", process?.executable ?? "-", "code"],
      ["Working directory", process?.working_directory ?? "-", "code"],
      ["Canonical run directory", process?.run_dir ?? "-", "code"],
      [
        "Process group ID",
        process === null ? "-" : String(process.process_group_id),
        "plain",
      ],
      [
        "Unix session ID",
        process === null ? "-" : String(process.unix_session_id),
        "plain",
      ],
    ]);
    element("process-command", HTMLElement).textContent =
      process === null
        ? "No managed CLI process"
        : process.argv.map(shellQuote).join(" ");
    const busy = this.#operations.initializing ||
      this.#operations.resuming || this.#operations.stopping;
    element("open-init", HTMLButtonElement).disabled =
      this.runDir() === "" || process !== null || busy;
    element("open-resume", HTMLButtonElement).disabled =
      this.runDir() === "" || process !== null || busy;
    const stop = element("stop-training", HTMLButtonElement);
    stop.disabled = process === null || this.#operations.stopping;
    stop.textContent = this.#operations.stopping ? "Stopping…" : "Stop";
  }

  #renderUptime(): void {
    const target = document.getElementById("process-uptime-value");
    if (target !== null && this.process !== null) {
      target.textContent = formatUptime(this.process);
    }
  }
}

function replaceWithRows(
  parent: HTMLElement,
  rows: readonly (readonly [string, string, "plain" | "code"])[],
): void {
  parent.replaceChildren(...rows.map(([label, value, kind]) => {
    const item = document.createElement("div");
    item.className = "detail-row";
    const key = document.createElement("span");
    key.textContent = label;
    const output = document.createElement("strong");
    output.textContent = value;
    output.className = kind === "code" ? "code-value" : "";
    if (label === "Uptime") output.id = "process-uptime-value";
    item.append(key, output);
    return item;
  }));
}

function formatTime(value: number | null): string {
  return value === null ? "-" : new Date(value).toLocaleString("en-GB");
}

function formatUptime(process: ProcessSnapshot): string {
  const seconds = Math.max(
    0,
    Math.floor((Date.now() - process.started_at_ms) / 1000),
  );
  return [
    Math.floor(seconds / 3600),
    Math.floor((seconds % 3600) / 60),
    seconds % 60,
  ].map((value) => String(value).padStart(2, "0")).join(":");
}

function shellQuote(value: string): string {
  return /^[A-Za-z0-9_./,:+=-]+$/.test(value)
    ? value
    : `'${value.replaceAll("'", `'"'"'`)}'`;
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
