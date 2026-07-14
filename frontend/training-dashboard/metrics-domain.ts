import { fetchMetrics } from "./api.ts";
import { DashboardCharts, type MetricAxis } from "./charts.ts";
import { MetricsInvalidationStream } from "./metrics.ts";
import type { JsonPrimitive, TrainingMetrics } from "./types.ts";

const DEBOUNCE_MS = 350;

export interface MetricsDomainCallbacks {
  readonly reportError: (message: string) => void;
  readonly clearError: () => void;
}

export class MetricsDomain {
  #metrics: TrainingMetrics | null = null;
  #axis: MetricAxis = "update";
  #running = false;
  #followUp = false;
  #dirtyThrough = -1;
  #generation = 0;
  #timer: ReturnType<typeof setTimeout> | null = null;
  readonly #charts = new DashboardCharts();
  readonly #stream = new MetricsInvalidationStream(
    () => this.runDir() || null,
    () => this.#metrics?.store_id ?? null,
    {
      onMessage: (message) => {
        if (
          message.type === "replacement" ||
          message.store_id !== (this.#metrics?.store_id ?? null)
        ) {
          this.#generation += 1;
          this.#metrics = null;
          this.#dirtyThrough = -1;
          this.render();
          if (this.isActive()) void this.refresh();
          return;
        }
        if (
          message.through_sequence <=
            (this.#metrics?.through_sequence ?? -1)
        ) return;
        this.#dirtyThrough = Math.max(
          this.#dirtyThrough,
          message.through_sequence,
        );
        if (this.isActive() && !document.hidden) this.#schedule();
      },
      onError: (message) => this.callbacks.reportError(message),
    },
  );

  constructor(
    private readonly runDir: () => string,
    private readonly isActive: () => boolean,
    private readonly callbacks: MetricsDomainCallbacks,
  ) {
    element("metrics-range", HTMLSelectElement).addEventListener(
      "change",
      () => void this.refresh(),
    );
    element("metrics-resolution", HTMLSelectElement).addEventListener(
      "change",
      () => void this.refresh(),
    );
    for (
      const button of document.querySelectorAll<HTMLButtonElement>(
        "[data-axis]",
      )
    ) {
      button.addEventListener("click", () => {
        this.#axis = button.dataset.axis === "elapsed"
          ? "elapsed"
          : "update";
        this.#renderAxisButtons();
        this.render();
      });
    }
    document.addEventListener("visibilitychange", () => {
      if (
        !document.hidden && this.isActive() &&
        this.#dirtyThrough > (this.#metrics?.through_sequence ?? -1)
      ) this.#schedule();
    });
  }

  connect(): void {
    this.#stream.connect();
  }

  activate(): void {
    this.#charts.resize();
    if (
      this.#metrics === null ||
      this.#dirtyThrough > this.#metrics.through_sequence
    ) void this.refresh();
  }

  reset(): void {
    this.#generation += 1;
    this.#metrics = null;
    this.#dirtyThrough = -1;
    this.#followUp = false;
    if (this.#timer !== null) clearTimeout(this.#timer);
    this.#timer = null;
    this.callbacks.clearError();
    this.render();
  }

  async refresh(): Promise<void> {
    if (this.#running) {
      this.#followUp = true;
      return;
    }
    const runDir = this.runDir();
    if (runDir === "") return;
    const generation = this.#generation;
    this.#running = true;
    try {
      const value = await fetchMetrics(
        runDir,
        selectedNumber("metrics-range"),
        selectedNumber("metrics-resolution"),
      );
      if (generation !== this.#generation || runDir !== this.runDir()) {
        return;
      }
      if (
        this.#metrics === null ||
        value.store_id !== this.#metrics.store_id ||
        value.through_sequence >= this.#metrics.through_sequence
      ) {
        const storeChanged = value.store_id !== this.#metrics?.store_id;
        this.#metrics = value;
        this.render();
        if (storeChanged) this.#stream.connect();
      }
      this.callbacks.clearError();
      if (value.through_sequence < this.#dirtyThrough) {
        this.#followUp = true;
      }
    } catch (error: unknown) {
      if (generation === this.#generation && runDir === this.runDir()) {
        this.callbacks.reportError(errorText(error));
      }
    } finally {
      this.#running = false;
      if (this.#followUp && this.isActive()) {
        this.#followUp = false;
        this.#schedule();
      }
    }
  }

  render(): void {
    const totals = this.#metrics?.totals ?? {};
    element("metric-strip", HTMLElement).replaceChildren(
      metricCell("Rounds", formatValue(totals.total_rounds)),
      metricCell("Samples", formatValue(totals.total_samples)),
      metricCell("Updates", formatValue(totals.total_updates)),
      metricCell("Samples/s", formatValue(totals.samples_per_second)),
      metricCell("Update time", formatSeconds(totals.update_seconds)),
      metricCell(
        "Log integrity",
        this.#metrics === null
          ? "-"
          : this.#metrics.complete
          ? "COMPLETE"
          : "INCOMPLETE",
      ),
    );
    if (this.#metrics === null) this.#charts.clear();
    else this.#charts.setData(this.#metrics, this.#axis);
  }

  #schedule(): void {
    if (this.#timer !== null) clearTimeout(this.#timer);
    this.#timer = setTimeout(() => {
      this.#timer = null;
      void this.refresh();
    }, DEBOUNCE_MS);
  }

  #renderAxisButtons(): void {
    for (
      const button of document.querySelectorAll<HTMLButtonElement>(
        "[data-axis]",
      )
    ) {
      button.classList.toggle(
        "active",
        button.dataset.axis === this.#axis,
      );
    }
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

function formatValue(value: JsonPrimitive | undefined): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "-";
    return value.toLocaleString("en-US", { maximumFractionDigits: 5 });
  }
  return String(value);
}

function formatSeconds(value: JsonPrimitive | undefined): string {
  return typeof value === "number" ? `${formatValue(value)} s` : "-";
}

function selectedNumber(id: string): number {
  const value = Number(element(id, HTMLSelectElement).value);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`Invalid selection: ${id}`);
  }
  return value;
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
