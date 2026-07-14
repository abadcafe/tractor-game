import { trainingStreamUrl } from "../stream.ts";
import { metricsStreamUrl } from "../metrics.ts";
import { checkpointStreamUrl } from "../checkpoints.ts";
import { parseTrainingStreamFrame } from "../stream-frame.ts";
import {
  parseCheckpointStreamMessage,
  parseLogMessage,
  parseLogPage,
  parseMetricsStreamMessage,
} from "../types.ts";

Deno.test("training stream resumes strictly after the last sequence", () => {
  const url = trainingStreamUrl(
    {
      runDir: "/tmp/run with spaces",
      afterSequence: 81,
      storeId: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    },
    { protocol: "https:", host: "training.example:8443" },
  );
  const parsed = new URL(url);
  if (parsed.protocol !== "wss:") throw new Error(url);
  if (parsed.pathname !== "/ws/training/logs") throw new Error(url);
  if (parsed.searchParams.get("after_sequence") !== "81") {
    throw new Error(url);
  }
  if (
    parsed.searchParams.get("store_id") !==
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  ) {
    throw new Error(url);
  }
  if (
    parsed.searchParams.has("window") ||
    parsed.searchParams.has("event") ||
    parsed.searchParams.has("session_id")
  ) {
    throw new Error(url);
  }
});

Deno.test("streams omit store_id before a database exists", () => {
  const location = { protocol: "http:", host: "training.example" };
  const logs = new URL(trainingStreamUrl({
    runDir: "/tmp/run",
    afterSequence: 0,
    storeId: null,
  }, location));
  const metrics = new URL(metricsStreamUrl("/tmp/run", null, location));
  const checkpoints = new URL(
    checkpointStreamUrl("/tmp/run", null, location),
  );
  if (
    logs.searchParams.has("store_id") ||
    metrics.searchParams.has("store_id") ||
    checkpoints.searchParams.has("store_id")
  ) throw new Error("Null store IDs must be omitted");
});

Deno.test("replacement messages preserve strict store generations", () => {
  const storeId = "0123456789abcdef0123456789abcdef";
  const logs = parseLogMessage({
    type: "replacement",
    store_id: storeId,
  });
  const metrics = parseMetricsStreamMessage({
    type: "replacement",
    store_id: storeId,
    through_sequence: 0,
  });
  const checkpoints = parseCheckpointStreamMessage({
    type: "replacement",
    store_id: storeId,
    through_sequence: 9,
  });
  const page = parseLogPage({
    store_id: storeId,
    events: [],
    next_before_sequence: null,
  });
  if (
    logs.type !== "replacement" ||
    metrics.type !== "replacement" ||
    checkpoints.type !== "replacement" ||
    page.store_id !== storeId
  ) throw new Error("Replacement generation was lost");
});

Deno.test("structured log parser rejects legacy lifecycle suffixes", () => {
  let rejected = false;
  try {
    parseLogMessage({
      type: "event",
      sequence: 8,
      event: {
        schema_version: 2,
        event: "update.completed",
        recorded_at_ms: 1,
        process: { kind: "coordinator", index: null, pid: 9 },
        context: {},
        fields: {},
      },
    });
  } catch {
    rejected = true;
  }
  if (!rejected) throw new Error("Legacy event name was accepted");
});

Deno.test("structured log parser accepts the terminal event protocol", () => {
  const event = parseLogMessage({
    type: "event",
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
  if (event.type !== "event" || event.sequence !== 7) {
    throw new Error("event");
  }
});

Deno.test("event parser rejects unknown correlation fields", () => {
  let rejected = false;
  try {
    parseLogMessage({
      type: "event",
      sequence: 7,
      event: {
        schema_version: 2,
        event: "update",
        recorded_at_ms: 1,
        process: { kind: "coordinator", index: null, pid: 9 },
        context: { session_id: "legacy" },
        fields: {},
      },
    });
  } catch {
    rejected = true;
  }
  if (!rejected) throw new Error("Unknown context field was accepted");
});

Deno.test("rejected stream frame preserves the terminal error", () => {
  const frame = parseTrainingStreamFrame({
    type: "rejected",
    error: "unsupported training database schema",
  });

  if (
    frame.type !== "rejected" ||
    frame.error !== "unsupported training database schema"
  ) {
    throw new Error("Rejected run reason was lost");
  }
});

Deno.test("domain stream frame passes through unchanged", () => {
  const message: unknown = {
    type: "invalidation",
    through_sequence: 9,
  };
  const frame = parseTrainingStreamFrame(message);

  if (frame.type !== "message" || frame.value !== message) {
    throw new Error("Domain message was changed");
  }
});
