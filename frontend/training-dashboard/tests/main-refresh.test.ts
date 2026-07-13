import {
  DashboardRefreshController,
  type DashboardRefreshSink,
  type DashboardRefreshSource,
  type RefreshErrorSource,
} from "../refresh.ts";
import { DashboardSelection } from "../selection.ts";
import { DashboardStatus } from "../status.ts";
import type {
  TrainingMetrics,
  TrainingProcess,
  TrainingSummary,
} from "../types.ts";

Deno.test("newer summary response wins within the same run", async () => {
  const fixture = refreshFixture();

  const older = fixture.controller.refreshSummary();
  const newer = fixture.controller.refreshSummary();
  fixture.source.summaryRequests[1]?.resolve(summary("READY", null));
  await newer;
  fixture.source.summaryRequests[0]?.resolve(
    summary("RUNNING", trainingProcess(101)),
  );
  await older;

  if (fixture.sink.summary?.state !== "READY") {
    throw new Error("Older RUNNING response overwrote READY");
  }
  if (fixture.sink.summaryApplyCount !== 1) {
    throw new Error("Only the latest summary may be applied");
  }
});

Deno.test("delayed stopped state cannot clear a resumed process", async () => {
  const fixture = refreshFixture();

  const stopped = fixture.controller.refreshSummary();
  const resumed = fixture.controller.refreshSummary();
  fixture.source.summaryRequests[1]?.resolve(
    summary("RUNNING", trainingProcess(202)),
  );
  await resumed;
  fixture.source.summaryRequests[0]?.resolve(summary("READY", null));
  await stopped;

  if (fixture.sink.summary?.process?.pid !== 202) {
    throw new Error(
      "Delayed READY response cleared the resumed process",
    );
  }
});

Deno.test("summary-only refresh supersedes older full refresh", async () => {
  const fixture = refreshFixture();

  const older = fixture.controller.refreshAll(metricOptions());
  const newer = fixture.controller.refreshSummary();
  fixture.source.summaryRequests[1]?.resolve(summary("READY", null));
  await newer;
  fixture.source.summaryRequests[0]?.resolve(
    summary("RUNNING", trainingProcess(102)),
  );
  await older;

  if (fixture.sink.summary?.state !== "READY") {
    throw new Error(
      "Full refresh bypassed the shared summary revision",
    );
  }
  if (fixture.source.metricRequests.length !== 0) {
    throw new Error(
      "Stale full refresh must not start a metrics query",
    );
  }
});

Deno.test("directory change discards pending summary response", async () => {
  const fixture = refreshFixture();

  const pending = fixture.controller.refreshSummary();
  fixture.selection.setRunDirectory("/runs/second");
  fixture.source.summaryRequests[0]?.resolve(
    summary("RUNNING", trainingProcess(103)),
  );
  await pending;

  if (fixture.sink.summary !== null) {
    throw new Error("Old directory response must not be applied");
  }
});

Deno.test("broken live run still loads metrics", async () => {
  const fixture = refreshFixture();

  const pending = fixture.controller.refreshAll(metricOptions());
  fixture.source.summaryRequests[0]?.resolve(
    summary("BROKEN", trainingProcess(104)),
  );
  await Promise.resolve();
  fixture.source.metricRequests[0]?.resolve(metrics("broken-session"));
  await pending;

  if (fixture.sink.metrics?.session_id !== "broken-session") {
    throw new Error("Live BROKEN run lost its valid metrics");
  }
});

Deno.test("metrics selection change preserves returned run summary", async () => {
  const fixture = refreshFixture();

  const full = fixture.controller.refreshAll(metricOptions());
  fixture.selection.setMetricSession("historical-session");
  fixture.source.summaryRequests[0]?.resolve(
    summary("RUNNING", trainingProcess(105)),
  );
  await full;

  if (fixture.sink.summary?.process?.pid !== 105) {
    throw new Error(
      "Metrics selection discarded the current run summary",
    );
  }
  if (fixture.source.metricRequests.length !== 0) {
    throw new Error(
      "Full refresh queried metrics for a stale selection",
    );
  }
});

Deno.test("successful metrics clears stale full refresh state", async () => {
  const fixture = refreshFixture();
  fixture.sink.summary = summary("READY", null);

  const full = fixture.controller.refreshAll(metricOptions());
  fixture.selection.setMetricSession("selected-session");
  const metricsRefresh = fixture.controller.refreshMetrics(
    metricOptions(),
  );
  fixture.source.metricRequests[0]?.resolve(
    metrics("selected-session"),
  );
  await metricsRefresh;
  fixture.source.summaryRequests[0]?.resolve(summary("READY", null));
  await full;

  if (fixture.sink.connection !== "online") {
    throw new Error(
      "Stale full refresh left the connection refreshing",
    );
  }
});

Deno.test("older full refresh preserves newer metrics failure", async () => {
  const fixture = refreshFixture();
  fixture.sink.summary = summary("READY", null);

  const full = fixture.controller.refreshAll(metricOptions());
  fixture.selection.setMetricSession("selected-session");
  const metricsRefresh = fixture.controller.refreshMetrics(
    metricOptions(),
  );
  fixture.source.metricRequests[0]?.reject(
    new Error("metrics unavailable"),
  );
  await metricsRefresh;
  fixture.source.summaryRequests[0]?.resolve(
    summary("RUNNING", trainingProcess(106)),
  );
  await full;

  if (fixture.sink.summary?.process?.pid !== 106) {
    throw new Error(
      "Older full refresh did not apply its valid summary",
    );
  }
  if (fixture.sink.error !== "metrics unavailable") {
    throw new Error("Older full refresh hid the newer metrics failure");
  }
  if (fixture.sink.connection !== "error") {
    throw new Error("Metrics failure no longer owns refresh status");
  }
});

Deno.test("summary refresh settles superseded full metrics failure", async () => {
  const fixture = refreshFixture();

  const full = fixture.controller.refreshAll(metricOptions());
  fixture.source.summaryRequests[0]?.resolve(
    summary("RUNNING", trainingProcess(107)),
  );
  await Promise.resolve();
  const summaryRefresh = fixture.controller.refreshSummary();
  if (fixture.sink.connection !== "online") {
    throw new Error(
      "Summary refresh did not release pending full refresh",
    );
  }
  fixture.source.summaryRequests[1]?.resolve(
    summary("RUNNING", trainingProcess(107)),
  );
  await summaryRefresh;

  if (fixture.sink.connection !== "online") {
    throw new Error("Superseded full refresh remained pending");
  }
  fixture.source.metricRequests[0]?.reject(
    new Error("stale metrics failure"),
  );
  await full;

  if (fixture.sink.error !== null) {
    throw new Error("Superseded metrics failure became visible");
  }
});

Deno.test("summary refresh preserves newer metrics pending state", async () => {
  const fixture = refreshFixture();
  fixture.sink.summary = summary("READY", null);

  const metricsRefresh = fixture.controller.refreshMetrics(
    metricOptions(),
  );
  const summaryRefresh = fixture.controller.refreshSummary();
  fixture.source.summaryRequests[0]?.resolve(summary("READY", null));
  await summaryRefresh;

  if (connectionState(fixture.sink) !== "refreshing") {
    throw new Error("Summary refresh ended a newer metrics request");
  }
  fixture.source.metricRequests[0]?.resolve(metrics("latest-session"));
  await metricsRefresh;
  if (connectionState(fixture.sink) !== "online") {
    throw new Error(
      "Metrics refresh did not release its pending state",
    );
  }
});

Deno.test("summary clears metrics error when metrics become unavailable", async () => {
  const fixture = refreshFixture();
  fixture.sink.summary = summary("READY", null);

  const metricsRefresh = fixture.controller.refreshMetrics(
    metricOptions(),
  );
  fixture.source.metricRequests[0]?.reject(
    new Error("metrics unavailable"),
  );
  await metricsRefresh;
  const summaryRefresh = fixture.controller.refreshSummary();
  fixture.source.summaryRequests[0]?.resolve(summary("BROKEN", null));
  await summaryRefresh;

  if (fixture.sink.metrics !== null) {
    throw new Error("Unavailable metrics data was not cleared");
  }
  if (fixture.sink.error !== null) {
    throw new Error("Unavailable metrics retained a stale error");
  }
  if (connectionState(fixture.sink) !== "online") {
    throw new Error("Stale metrics error kept the dashboard offline");
  }
});

interface Deferred<Value> {
  readonly promise: Promise<Value>;
  resolve(value: Value): void;
  reject(reason: Error): void;
}

interface SummaryRequest extends Deferred<TrainingSummary> {
  readonly runDir: string;
}

interface MetricRequest extends Deferred<TrainingMetrics> {
  readonly runDir: string;
  readonly sessionId: string | null;
}

class ControlledSource implements DashboardRefreshSource {
  readonly summaryRequests: SummaryRequest[] = [];
  readonly metricRequests: MetricRequest[] = [];

  fetchSummary(runDir: string): Promise<TrainingSummary> {
    const response = deferred<TrainingSummary>();
    this.summaryRequests.push({ runDir, ...response });
    return response.promise;
  }

  fetchMetrics(
    runDir: string,
    _updateLimit: number,
    _seriesPoints: number,
    sessionId: string | null,
  ): Promise<TrainingMetrics> {
    const response = deferred<TrainingMetrics>();
    this.metricRequests.push({ runDir, sessionId, ...response });
    return response.promise;
  }
}

class RecordingSink implements DashboardRefreshSink {
  readonly #status = new DashboardStatus();
  summary: TrainingSummary | null = null;
  metrics: TrainingMetrics | null = null;
  summaryApplyCount = 0;

  constructor() {
    this.#status.setStreamConnection("online");
  }

  get error(): string | null {
    return this.#status.snapshot().message || null;
  }

  get connection(): "error" | "online" | "refreshing" {
    const label = this.#status.snapshot().label;
    if (label === "ERROR") return "error";
    if (label === "REFRESHING") return "refreshing";
    return "online";
  }

  currentSummary(): TrainingSummary | null {
    return this.summary;
  }

  applySummary(value: TrainingSummary): void {
    this.summary = value;
    this.summaryApplyCount += 1;
  }

  applyMetrics(value: TrainingMetrics | null): void {
    this.metrics = value;
  }

  setRefreshPending(): void {
    this.#status.setRefreshPending();
  }

  setRefreshIdle(): void {
    this.#status.setRefreshIdle();
  }

  reportError(source: RefreshErrorSource, message: string): void {
    this.#status.reportError(source, message);
  }

  clearError(source: RefreshErrorSource): void {
    this.#status.clearError(source);
  }
}

function connectionState(
  sink: RecordingSink,
): "error" | "online" | "refreshing" {
  return sink.connection;
}

interface RefreshFixture {
  readonly selection: DashboardSelection;
  readonly source: ControlledSource;
  readonly sink: RecordingSink;
  readonly controller: DashboardRefreshController;
}

function refreshFixture(): RefreshFixture {
  const selection = new DashboardSelection();
  selection.setRunDirectory("/runs/first");
  const source = new ControlledSource();
  const sink = new RecordingSink();
  return {
    selection,
    source,
    sink,
    controller: new DashboardRefreshController(selection, source, sink),
  };
}

function deferred<Value>(): Deferred<Value> {
  let complete: ((value: Value) => void) | null = null;
  let fail: ((reason: Error) => void) | null = null;
  const promise = new Promise<Value>((resolve, reject) => {
    complete = resolve;
    fail = reject;
  });
  return {
    promise,
    resolve(value: Value): void {
      if (complete === null) {
        throw new Error("Deferred is not initialized");
      }
      complete(value);
    },
    reject(reason: Error): void {
      if (fail === null) {
        throw new Error("Deferred is not initialized");
      }
      fail(reason);
    },
  };
}

function metricOptions(): {
  readonly updateLimit: number;
  readonly seriesPoints: number;
} {
  return { updateLimit: 100, seriesPoints: 50 };
}

function trainingProcess(pid: number): TrainingProcess {
  return {
    pid,
    name: "python",
    kernel_state: "S",
    executable: "/usr/bin/python",
    working_directory: "/workspace",
    run_dir: "/runs/first",
    argv: ["/usr/bin/python"],
    process_group_id: pid,
    session_id: pid,
    start_ticks: pid * 100,
  };
}

function summary(
  state: "BROKEN" | "READY" | "RUNNING",
  process: TrainingProcess | null,
): TrainingSummary {
  return {
    schema_version: 2,
    run_dir: "/runs/first",
    state,
    reason: state === "BROKEN" ? "checkpoint missing" : null,
    process,
    details: state === "BROKEN" ? null : {
      checkpoint_id: "checkpoint-1",
      checkpoint_path: "/runs/first/checkpoints/latest.json",
      state_size_bytes: 1,
      model_config_values: {},
      train_config_values: {},
      total_rounds: 0,
      total_samples: 0,
      total_updates: 0,
    },
    checkpoints: {
      checkpoint_directory: "/runs/first/checkpoints",
      manifests: [],
      objects: [],
      total_unique_state_bytes: 0,
    },
  };
}

function metrics(sessionId: string): TrainingMetrics {
  const datasets = {
    throughput: [],
    optimization: [],
    ppo_timing: [],
    rollout: [],
    rewards: [],
    inference: [],
    processes: [],
  };
  return {
    schema_version: 1,
    through_sequence: 0,
    session_id: sessionId,
    sessions: [],
    complete: true,
    dropped_event_count: 0,
    totals: {},
    datasets,
  };
}
