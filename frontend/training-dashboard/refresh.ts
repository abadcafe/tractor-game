import { DashboardSelection } from "./selection.ts";
import type { TrainingMetrics, TrainingSummary } from "./types.ts";

export type RefreshErrorSource = "metrics" | "summary";

export interface MetricRefreshOptions {
  readonly updateLimit: number;
  readonly seriesPoints: number;
}

export interface DashboardRefreshSource {
  fetchSummary(runDir: string): Promise<TrainingSummary>;
  fetchMetrics(
    runDir: string,
    updateLimit: number,
    seriesPoints: number,
    sessionId: string | null,
  ): Promise<TrainingMetrics>;
}

export interface DashboardRefreshSink {
  currentSummary(): TrainingSummary | null;
  applySummary(summary: TrainingSummary): void;
  applyMetrics(metrics: TrainingMetrics | null): void;
  setRefreshPending(): void;
  setRefreshIdle(): void;
  reportError(source: RefreshErrorSource, message: string): void;
  clearError(source: RefreshErrorSource): void;
}

export type RefreshResult = "applied" | "failed" | "skipped" | "stale";

export class DashboardRefreshController {
  #summaryRevision = 0;
  #metricRevision = 0;
  #refreshRevision = 0;
  #pendingFullRefreshRevision: number | null = null;

  constructor(
    private readonly selection: DashboardSelection,
    private readonly source: DashboardRefreshSource,
    private readonly sink: DashboardRefreshSink,
  ) {}

  async refreshAll(
    options: MetricRefreshOptions,
  ): Promise<RefreshResult> {
    const origin = this.selection.captureMetrics();
    if (origin.runDir === "") return "skipped";
    this.sink.setRefreshPending();
    const summaryRevision = ++this.#summaryRevision;
    const metricRevision = ++this.#metricRevision;
    const refreshRevision = ++this.#refreshRevision;
    this.#pendingFullRefreshRevision = refreshRevision;
    let summary: TrainingSummary;
    try {
      summary = await this.source.fetchSummary(origin.runDir);
    } catch (error: unknown) {
      if (
        this.selection.ownsRun(origin) &&
        summaryRevision === this.#summaryRevision
      ) {
        this.sink.reportError("summary", errorText(error));
        this.#finishRefresh(refreshRevision);
        return "failed";
      }
      this.#finishRefresh(refreshRevision);
      return "stale";
    }
    if (
      !this.selection.ownsRun(origin) ||
      summaryRevision !== this.#summaryRevision
    ) {
      this.#finishRefresh(refreshRevision);
      return "stale";
    }
    this.sink.applySummary(summary);
    this.sink.clearError("summary");
    if (
      !this.selection.ownsMetrics(origin) ||
      metricRevision !== this.#metricRevision
    ) {
      this.#finishRefresh(refreshRevision);
      return "applied";
    }
    if (!canQueryMetrics(summary)) {
      this.sink.applyMetrics(null);
      this.sink.clearError("metrics");
      this.#finishRefresh(refreshRevision);
      return "applied";
    }
    let metrics: TrainingMetrics;
    try {
      metrics = await this.source.fetchMetrics(
        origin.runDir,
        options.updateLimit,
        options.seriesPoints,
        origin.sessionId,
      );
    } catch (error: unknown) {
      if (
        this.selection.ownsMetrics(origin) &&
        summaryRevision === this.#summaryRevision &&
        metricRevision === this.#metricRevision
      ) {
        this.sink.reportError("metrics", errorText(error));
        this.#finishRefresh(refreshRevision);
        return "failed";
      }
      if (
        this.selection.ownsRun(origin) &&
        summaryRevision === this.#summaryRevision
      ) {
        this.#finishRefresh(refreshRevision);
        return "applied";
      }
      this.#finishRefresh(refreshRevision);
      return "stale";
    }
    if (summaryRevision !== this.#summaryRevision) {
      this.#finishRefresh(refreshRevision);
      return "stale";
    }
    if (
      !this.selection.ownsMetrics(origin) ||
      metricRevision !== this.#metricRevision
    ) {
      this.#finishRefresh(refreshRevision);
      return "applied";
    }
    this.sink.applyMetrics(metrics);
    this.sink.clearError("metrics");
    this.#finishRefresh(refreshRevision);
    return "applied";
  }

  async refreshSummary(): Promise<RefreshResult> {
    const origin = this.selection.captureRun();
    if (origin.runDir === "") return "skipped";
    const summaryRevision = ++this.#summaryRevision;
    this.#releasePendingFullRefresh();
    let summary: TrainingSummary;
    try {
      summary = await this.source.fetchSummary(origin.runDir);
    } catch (error: unknown) {
      if (
        this.selection.ownsRun(origin) &&
        summaryRevision === this.#summaryRevision
      ) {
        this.sink.reportError("summary", errorText(error));
        return "failed";
      }
      return "stale";
    }
    if (
      !this.selection.ownsRun(origin) ||
      summaryRevision !== this.#summaryRevision
    ) return "stale";
    this.sink.applySummary(summary);
    this.sink.clearError("summary");
    if (!canQueryMetrics(summary)) {
      this.#metricRevision += 1;
      this.sink.applyMetrics(null);
      this.sink.clearError("metrics");
    }
    return "applied";
  }

  async refreshMetrics(
    options: MetricRefreshOptions,
  ): Promise<RefreshResult> {
    const origin = this.selection.captureMetrics();
    if (origin.runDir === "") return "skipped";
    const summary = this.sink.currentSummary();
    const summaryRevision = this.#summaryRevision;
    const metricRevision = ++this.#metricRevision;
    const refreshRevision = ++this.#refreshRevision;
    this.#pendingFullRefreshRevision = null;
    this.sink.setRefreshPending();
    if (!canQueryMetrics(summary)) {
      this.sink.applyMetrics(null);
      this.sink.clearError("metrics");
      this.#finishRefresh(refreshRevision);
      return "applied";
    }
    let metrics: TrainingMetrics;
    try {
      metrics = await this.source.fetchMetrics(
        origin.runDir,
        options.updateLimit,
        options.seriesPoints,
        origin.sessionId,
      );
    } catch (error: unknown) {
      if (
        this.selection.ownsMetrics(origin) &&
        summaryRevision === this.#summaryRevision &&
        metricRevision === this.#metricRevision
      ) {
        this.sink.reportError("metrics", errorText(error));
        this.#finishRefresh(refreshRevision);
        return "failed";
      }
      this.#finishRefresh(refreshRevision);
      return "stale";
    }
    if (
      !this.selection.ownsMetrics(origin) ||
      summaryRevision !== this.#summaryRevision ||
      metricRevision !== this.#metricRevision
    ) {
      this.#finishRefresh(refreshRevision);
      return "stale";
    }
    this.sink.applyMetrics(metrics);
    this.sink.clearError("metrics");
    this.#finishRefresh(refreshRevision);
    return "applied";
  }

  #finishRefresh(revision: number): void {
    if (revision === this.#pendingFullRefreshRevision) {
      this.#pendingFullRefreshRevision = null;
    }
    if (revision === this.#refreshRevision) {
      this.sink.setRefreshIdle();
    }
  }

  #releasePendingFullRefresh(): void {
    if (this.#pendingFullRefreshRevision !== this.#refreshRevision) {
      return;
    }
    this.#pendingFullRefreshRevision = null;
    this.sink.setRefreshIdle();
  }
}

function canQueryMetrics(value: TrainingSummary | null): boolean {
  return value !== null &&
    (value.state === "READY" || value.process !== null);
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
