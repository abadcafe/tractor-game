import {
  parseTrainingStreamMessage,
  type TrainingStreamSnapshot,
} from "../types.ts";
import { trainingStreamUrl } from "../stream.ts";

Deno.test("training stream URL carries cursors and selected log", () => {
  const url = trainingStreamUrl(
    {
      runDir: "/tmp/run with spaces",
      metricSequence: 7,
      telemetrySequence: 9,
      logStream: "stderr",
    },
    { protocol: "https:", host: "training.example:8443" },
  );

  const parsed = new URL(url);
  if (parsed.protocol !== "wss:") throw new Error(url);
  if (parsed.pathname !== "/ws/training") throw new Error(url);
  if (parsed.searchParams.get("run_dir") !== "/tmp/run with spaces") {
    throw new Error(url);
  }
  if (parsed.searchParams.get("metric_sequence") !== "7") {
    throw new Error(url);
  }
  if (parsed.searchParams.get("telemetry_sequence") !== "9") {
    throw new Error(url);
  }
  if (parsed.searchParams.get("log_stream") !== "stderr") {
    throw new Error(url);
  }
});

Deno.test("training stream parser validates snapshot envelope", () => {
  const message = parseTrainingStreamMessage({
    type: "snapshot",
    summary: {
      schema_version: 1,
      run_dir: "/tmp/run",
      state: "NOT_INITIALIZED",
      reason: null,
      process: null,
      details: null,
      metrics: [],
      telemetry: [],
      checkpoints: {
        checkpoint_directory: "/tmp/run/checkpoints",
        manifests: [],
        objects: [],
        total_unique_state_bytes: 0,
      },
    },
    log_stream: null,
    log_content: null,
  });

  if (message.type !== "snapshot") throw new Error(message.type);
  assertEmptySnapshot(message);
});

Deno.test("training stream parser validates error envelope", () => {
  const message = parseTrainingStreamMessage({
    type: "error",
    message: "database unavailable",
  });

  if (message.type !== "error") throw new Error(message.type);
  if (message.message !== "database unavailable") {
    throw new Error(message.message);
  }
});

function assertEmptySnapshot(message: TrainingStreamSnapshot): void {
  if (message.summary.state !== "NOT_INITIALIZED") {
    throw new Error("expected an uninitialized run");
  }
  if (message.summary.metrics.length !== 0) {
    throw new Error("expected no metrics");
  }
  if (message.summary.telemetry.length !== 0) {
    throw new Error("expected no telemetry");
  }
  if (message.log_stream !== null || message.log_content !== null) {
    throw new Error("expected no log subscription");
  }
}
