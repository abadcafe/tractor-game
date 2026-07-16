import { ProcessStreamClient } from "./process.ts";
import type {
  ProcessDetails,
  ProcessSnapshot,
  ProcessState,
} from "./types.ts";

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
  #state: ProcessState | null = null;
  #operations: ProcessOperations = {
    initializing: false,
    resuming: false,
    stopping: false,
  };
  readonly #stream = new ProcessStreamClient(
    () => this.runDir() || null,
    {
      onSnapshot: (value) => {
        this.#state = value;
        this.callbacks.clearError();
        this.render();
      },
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
    return this.#state?.process ?? null;
  }

  connect(): void {
    this.#stream.connect();
  }

  reset(): void {
    this.#state = null;
    this.callbacks.clearError();
    this.render();
  }

  setOperations(value: ProcessOperations): void {
    this.#operations = value;
    this.render();
  }

  render(): void {
    const process = this.process;
    const details = process?.inspection.kind === "details"
      ? process.inspection
      : null;
    const inspectionError = process?.inspection.kind === "error"
      ? process.inspection.error
      : null;
    const caption = element("run-caption", HTMLElement);
    caption.textContent = this.runDir();
    caption.title = this.runDir();
    const presence = element("process-presence", HTMLElement);
    presence.textContent = process === null ? "STOPPED" : "RUNNING";
    presence.className = process === null
      ? "badge neutral"
      : "badge running";
    replaceWithRows(element("process-details", HTMLElement), [
      ["PID", process === null ? "-" : String(process.pid), "plain"],
      ["Started", formatTime(details?.started_at_ms ?? null), "plain"],
      [
        "Uptime",
        details === null ? "-" : formatUptime(details),
        "plain",
      ],
      ["Kernel state", details?.kernel_state ?? "-", "plain"],
      ["Executable", details?.executable ?? "-", "code"],
      ["Working directory", details?.working_directory ?? "-", "code"],
      [
        "Process group ID",
        details === null ? "-" : String(details.process_group_id),
        "plain",
      ],
      [
        "Unix session ID",
        details === null ? "-" : String(details.unix_session_id),
        "plain",
      ],
      ["Inspection error", inspectionError ?? "-", "plain"],
    ]);
    element("process-command", HTMLElement).textContent =
      process === null
        ? "No live PID"
        : details === null
        ? inspectionError ?? "Process information is unavailable"
        : details.argv.map(shellQuote).join(" ");
    const busy = this.#operations.initializing ||
      this.#operations.resuming || this.#operations.stopping;
    element("open-init", HTMLButtonElement).disabled =
      this.runDir() === "" || busy;
    element("open-resume", HTMLButtonElement).disabled =
      this.runDir() === "" || process !== null || busy;
    const stop = element("stop-training", HTMLButtonElement);
    stop.disabled = process === null || busy;
    stop.textContent = this.#operations.stopping ? "Stopping…" : "Stop";
  }

  #renderUptime(): void {
    const target = document.getElementById("process-uptime-value");
    const inspection = this.process?.inspection;
    if (
      target !== null && inspection !== undefined &&
      inspection.kind === "details"
    ) {
      target.textContent = formatUptime(inspection);
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

function formatUptime(process: ProcessDetails): string {
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
