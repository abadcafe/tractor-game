import { DashboardCharts, type MetricAxis } from "./charts.ts";
import {
  MetricsSnapshotStream,
  type MetricsStreamTarget,
} from "./metrics.ts";
import type { JsonPrimitive, TrainingMetrics } from "./types.ts";

export interface MetricsDomainCallbacks {
  readonly reportError: (message: string) => void;
  readonly clearError: () => void;
}

export class MetricsDomain {
  #metrics: TrainingMetrics | null = null;
  #axis: MetricAxis = "update";
  readonly #charts = new DashboardCharts();
  readonly #stream = new MetricsSnapshotStream(
    () => this.#streamTarget(),
    {
      onSnapshot: (snapshot) => {
        if (
          snapshot.store_id === this.#metrics?.store_id &&
          snapshot.through_sequence < this.#metrics.through_sequence
        ) {
          return;
        }
        this.#metrics = snapshot;
        this.callbacks.clearError();
        this.render();
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
      () => this.refresh(),
    );
    element("metrics-resolution", HTMLSelectElement).addEventListener(
      "change",
      () => this.refresh(),
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
      if (document.hidden) this.#stream.disconnect();
      else if (this.isActive()) this.#stream.connect();
    });
  }

  activate(): void {
    this.#charts.resize();
    if (!document.hidden) this.#stream.connect();
  }

  deactivate(): void {
    this.#stream.disconnect();
  }

  reset(): void {
    this.#stream.disconnect();
    this.#metrics = null;
    this.callbacks.clearError();
    this.render();
  }

  refresh(): void {
    if (this.isActive() && !document.hidden) this.#stream.connect();
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

  #streamTarget(): MetricsStreamTarget | null {
    const runDir = this.runDir();
    if (runDir === "" || !this.isActive() || document.hidden) {
      return null;
    }
    return {
      runDir,
      updateLimit: selectedNumber("metrics-range"),
      seriesPoints: selectedNumber("metrics-resolution"),
    };
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
