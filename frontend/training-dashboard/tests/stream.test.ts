import { checkpointEventUrl } from "../checkpoint-events.ts";
import { logEventUrl } from "../log-events.ts";
import { metricEventUrl } from "../metric-events.ts";
import {
  parseCheckpointCursor,
  parseLogEntry,
  parseLogPage,
  parseMetrics,
  parseStoreReplacement,
} from "../types.ts";

Deno.test("log events resume strictly after the last sequence", () => {
  const url = logEventUrl({
    runDir: "/tmp/run with spaces",
    afterSequence: 81,
    storeId: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  });
  const parsed = new URL(url, "https://training.example:8443");
  if (parsed.protocol !== "https:") throw new Error(url);
  if (parsed.pathname !== "/api/training/events/logs") {
    throw new Error(url);
  }
  if (parsed.searchParams.get("after_sequence") !== "81") {
    throw new Error(url);
  }
  if (
    parsed.searchParams.get("store_id") !==
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  ) {
    throw new Error(url);
  }
  const keys = [...parsed.searchParams.keys()].sort();
  if (keys.join(",") !== "after_sequence,run_dir,store_id") {
    throw new Error(url);
  }
});

Deno.test("streams omit store_id before a database exists", () => {
  const origin = "http://training.example";
  const logs = new URL(
    logEventUrl({
      runDir: "/tmp/run",
      afterSequence: 0,
      storeId: null,
    }),
    origin,
  );
  const metrics = new URL(
    metricEventUrl({
      runDir: "/tmp/run",
      updateLimit: 200,
      seriesPoints: 500,
    }),
    origin,
  );
  const checkpoints = new URL(
    checkpointEventUrl("/tmp/run", null),
    origin,
  );
  if (
    logs.searchParams.has("store_id") ||
    metrics.searchParams.has("store_id") ||
    checkpoints.searchParams.has("store_id")
  ) throw new Error("Null store IDs must be omitted");
});

Deno.test("metrics stream owns snapshot projection parameters", () => {
  const url = new URL(
    metricEventUrl({
      runDir: "/tmp/run with spaces",
      updateLimit: 50,
      seriesPoints: 200,
    }),
    "https://training.example:8443",
  );

  if (
    url.protocol !== "https:" ||
    url.pathname !== "/api/training/events/metrics"
  ) {
    throw new Error(url.toString());
  }
  if (
    url.searchParams.get("run_dir") !== "/tmp/run with spaces" ||
    url.searchParams.get("update_limit") !== "50" ||
    url.searchParams.get("series_points") !== "200" ||
    url.searchParams.has("store_id")
  ) throw new Error(url.toString());
});

Deno.test("replacement messages preserve strict store generations", () => {
  const storeId = "0123456789abcdef0123456789abcdef";
  const logs = parseStoreReplacement({
    store_id: storeId,
  });
  const checkpoints = parseCheckpointCursor({
    store_id: storeId,
    through_sequence: 9,
  });
  const page = parseLogPage({
    store_id: storeId,
    events: [],
    next_before_sequence: null,
  });
  if (
    logs.store_id !== storeId ||
    checkpoints.store_id !== storeId ||
    page.store_id !== storeId
  ) throw new Error("Replacement generation was lost");
});

Deno.test("metrics stream frames are complete snapshots", () => {
  const metrics = parseMetrics({
    schema_version: 2,
    store_id: null,
    through_sequence: 0,
    complete: true,
    dropped_event_count: 0,
    totals: {},
    datasets: {
      throughput: [],
      optimization: [],
      ppo_timing: [],
      rollout: [],
      rewards: [],
      inference: [],
      processes: [],
    },
  });

  if (
    metrics.store_id !== null || metrics.through_sequence !== 0 ||
    metrics.datasets.inference.length !== 0
  ) throw new Error("Metrics snapshot was not preserved");
});

Deno.test("structured log parser accepts the terminal event protocol", () => {
  const event = parseLogEntry({
    sequence: 7,
    event: {
      schema_version: 2,
      event: "update",
      recorded_at_ms: 1,
      process: { kind: "coordinator", index: null, pid: 9 },
      context: {},
      fields: {},
    },
  });
  if (event.sequence !== 7) {
    throw new Error("event");
  }
});

Deno.test("event parser rejects unknown correlation fields", () => {
  let rejected = false;
  try {
    parseLogEntry({
      sequence: 7,
      event: {
        schema_version: 2,
        event: "update",
        recorded_at_ms: 1,
        process: { kind: "coordinator", index: null, pid: 9 },
        context: { unexpected: true },
        fields: {},
      },
    });
  } catch {
    rejected = true;
  }
  if (!rejected) throw new Error("Unknown context field was accepted");
});
