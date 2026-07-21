import { fetchCheckpoints } from "./api.ts";
import { CheckpointEventStream } from "./checkpoint-events.ts";
import type {
  CheckpointCatalog,
  CheckpointManifest,
  JsonPrimitive,
} from "./types.ts";

export interface CheckpointsDomainCallbacks {
  readonly reportError: (message: string) => void;
  readonly clearError: () => void;
  readonly canResume: () => boolean;
  readonly resumeFrom: (manifest: CheckpointManifest) => void;
  readonly inspect: (manifest: CheckpointManifest) => void;
}

export class CheckpointsDomain {
  #catalog: CheckpointCatalog | null = null;
  #generation = 0;
  #storeId: string | null = null;
  #dirtyThrough = -1;
  #appliedThrough = -1;
  readonly #stream = new CheckpointEventStream(
    () => this.runDir() || null,
    () => this.#storeId,
    {
      onMessage: (message) => {
        const replaced = message.type === "replacement" ||
          message.store_id !== this.#storeId;
        if (replaced) {
          this.#generation += 1;
          this.#catalog = null;
          this.#dirtyThrough = -1;
          this.#appliedThrough = -1;
          this.render();
        }
        const storeChanged = message.store_id !== this.#storeId;
        this.#storeId = message.store_id;
        this.#dirtyThrough = Math.max(
          this.#dirtyThrough,
          message.through_sequence,
        );
        if (storeChanged) this.#stream.connect();
        if (this.isActive() && !document.hidden) void this.refresh();
      },
      onError: (message) => this.callbacks.reportError(message),
    },
  );

  constructor(
    private readonly runDir: () => string,
    private readonly isActive: () => boolean,
    private readonly callbacks: CheckpointsDomainCallbacks,
  ) {
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) this.#stream.disconnect();
      else if (this.isActive()) this.activate();
    });
  }

  get loaded(): boolean {
    return this.#catalog !== null;
  }

  activate(): void {
    if (document.hidden || !this.isActive()) return;
    this.#stream.connect();
    if (
      this.#catalog === null ||
      this.#dirtyThrough > this.#appliedThrough
    ) void this.refresh();
  }

  deactivate(): void {
    this.#stream.disconnect();
  }

  reset(): void {
    this.#generation += 1;
    this.#stream.disconnect();
    this.#catalog = null;
    this.#storeId = null;
    this.#dirtyThrough = -1;
    this.#appliedThrough = -1;
    this.callbacks.clearError();
    this.render();
  }

  async refresh(): Promise<void> {
    const runDir = this.runDir();
    if (runDir === "") return;
    const generation = this.#generation;
    try {
      const value = await fetchCheckpoints(runDir);
      if (generation !== this.#generation || runDir !== this.runDir()) {
        return;
      }
      this.#catalog = value;
      this.#appliedThrough = this.#dirtyThrough;
      this.callbacks.clearError();
      this.render();
    } catch (error: unknown) {
      if (generation === this.#generation && runDir === this.runDir()) {
        this.callbacks.reportError(errorText(error));
      }
    }
  }

  render(): void {
    const directory = this.#catalog?.checkpoint_directory ??
      `${this.runDir()}/checkpoints`;
    const caption = element("checkpoint-directory", HTMLElement);
    caption.textContent = directory;
    caption.title = directory;
    element("checkpoint-summary", HTMLElement).replaceChildren(
      metricCell(
        "Valid manifests",
        String(
          this.#catalog?.manifests.filter((item) => item.valid)
            .length ?? 0,
        ),
      ),
      metricCell("Objects", String(this.#catalog?.objects.length ?? 0)),
      metricCell(
        "Orphans",
        String(
          this.#catalog?.objects.filter((item) => item.orphan).length ??
            0,
        ),
      ),
      metricCell(
        "Unique storage",
        formatBytes(this.#catalog?.total_unique_state_bytes ?? 0),
      ),
    );
    element("manifest-rows", HTMLTableSectionElement).replaceChildren(
      ...(this.#catalog?.manifests ?? []).map((manifest) => {
        const actions = document.createElement("div");
        actions.className = "table-actions";
        actions.append(
          actionButton(
            "Inspect",
            () => this.callbacks.inspect(manifest),
          ),
          actionButton(
            "Resume",
            () => {
              if (this.callbacks.canResume()) {
                this.callbacks.resumeFrom(manifest);
              }
            },
            manifest.valid && this.callbacks.canResume(),
          ),
        );
        return rowWithNode(
          [
            manifest.name,
            formatValue(manifest.total_updates),
            formatValue(manifest.total_samples),
            manifest.state_exists ? "Available" : "Missing",
            formatBytes(manifest.state_size_bytes),
            manifest.error ?? "Manifest valid / state present",
          ],
          actions,
          manifest.valid ? "" : "invalid-row",
        );
      }),
    );
    element("object-rows", HTMLTableSectionElement).replaceChildren(
      ...(this.#catalog?.objects ?? []).map((item) =>
        row([
          item.checkpoint_id,
          item.state_path,
          formatBytes(item.state_size_bytes),
          item.referenced_by.join(", ") || "None",
          item.error ?? (item.orphan ? "Orphan" : "Referenced"),
        ], item.valid ? "" : "invalid-row")
      ),
    );
  }
}

function metricCell(label: string, value: string): HTMLElement {
  const cell = document.createElement("div");
  cell.className = "metric-cell";
  const caption = document.createElement("span");
  caption.textContent = label;
  const output = document.createElement("strong");
  output.textContent = value;
  cell.append(caption, output);
  return cell;
}

function row(
  values: readonly string[],
  className = "",
): HTMLTableRowElement {
  const item = document.createElement("tr");
  item.className = className;
  for (const value of values) {
    const cell = document.createElement("td");
    cell.textContent = value;
    item.append(cell);
  }
  return item;
}

function rowWithNode(
  values: readonly string[],
  node: Node,
  className: string,
): HTMLTableRowElement {
  const item = row(values, className);
  const cell = document.createElement("td");
  cell.append(node);
  item.append(cell);
  return item;
}

function actionButton(
  label: string,
  action: () => void,
  enabled = true,
): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "table-action";
  button.textContent = label;
  button.disabled = !enabled;
  button.addEventListener("click", action);
  return button;
}

function formatValue(value: JsonPrimitive | undefined): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "-";
    return value.toLocaleString("en-US", { maximumFractionDigits: 5 });
  }
  return String(value);
}

function formatBytes(value: number | null): string {
  if (value === null) return "-";
  if (value < 1024) return `${value} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let output = value / 1024;
  let unit = units[0] ?? "KiB";
  for (
    let index = 1;
    index < units.length && output >= 1024;
    index += 1
  ) {
    output /= 1024;
    unit = units[index] ?? unit;
  }
  return `${output.toFixed(output >= 10 ? 1 : 2)} ${unit}`;
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
