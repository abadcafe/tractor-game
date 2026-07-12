import { recordValue } from "../browser/json.ts";

export interface TrainingConfig {
  readonly default_run_dir: string;
  readonly stop_timeout_seconds: number;
}

export interface TrainingProcess {
  readonly pid: number;
  readonly name: string;
  readonly kernel_state: string;
  readonly executable: string;
  readonly working_directory: string;
  readonly run_dir: string;
  readonly argv: readonly string[];
  readonly process_group_id: number;
  readonly session_id: number;
  readonly start_ticks: number;
}

export type TrainingRunState =
  | "NOT_INITIALIZED"
  | "BROKEN"
  | "READY"
  | "RUNNING";

export interface TrainingRunDetails {
  readonly checkpoint_id: string;
  readonly checkpoint_path: string;
  readonly state_size_bytes: number;
  readonly model_config_values: Readonly<Record<string, JsonPrimitive>>;
  readonly train_config_values: Readonly<Record<string, JsonPrimitive>>;
  readonly total_rounds: number;
  readonly total_samples: number;
  readonly total_updates: number;
  readonly metric_count: number;
}

export interface TrainingRunStatus {
  readonly state: TrainingRunState;
  readonly reason: string | null;
  readonly process: TrainingProcess | null;
  readonly details: TrainingRunDetails | null;
}

export interface TrainingMetric {
  readonly sequence: number;
  readonly recorded_at_ms: number;
  readonly total_games: number;
  readonly total_samples: number;
  readonly total_updates: number;
  readonly process_games_per_second: number;
  readonly process_samples_per_second: number;
  readonly last_round_decisions_per_second: number;
  readonly last_team0_reward: number;
  readonly last_team1_reward: number;
  readonly last_generated_action_count: number;
  readonly last_accepted_action_count: number;
  readonly last_decision_count: number;
  readonly last_average_action_choices: number;
  readonly policy_loss: number | null;
  readonly value_loss: number | null;
  readonly entropy: number | null;
  readonly approx_kl: number | null;
  readonly clip_fraction: number | null;
  readonly ppo_update_seconds: number | null;
  readonly ppo_minibatch_loss_seconds: number | null;
  readonly ppo_observation_batch_seconds: number | null;
  readonly ppo_observation_encode_seconds: number | null;
  readonly ppo_value_head_seconds: number | null;
  readonly ppo_argument_select_seconds: number | null;
  readonly ppo_argument_decode_seconds: number | null;
  readonly ppo_argument_distribution_seconds: number | null;
  readonly ppo_backward_seconds: number | null;
  readonly ppo_optimizer_step_seconds: number | null;
  readonly ppo_argument_decode_fraction: number | null;
  readonly ppo_argument_trace_batch_count: number | null;
  readonly ppo_argument_trace_row_count: number | null;
  readonly ppo_argument_trace_token_count: number | null;
  readonly ppo_argument_trace_valid_token_count: number | null;
  readonly ppo_argument_trace_padding_token_count: number | null;
  readonly checkpoint_path: string | null;
}

export interface TelemetryMeasurement {
  readonly key: string;
  readonly value: number;
}

export interface TelemetryEvent {
  readonly sequence: number;
  readonly recorded_at_ms: number;
  readonly process_label: string;
  readonly stage: string;
  readonly total_rounds: number;
  readonly total_updates: number;
  readonly progress_numerator: number;
  readonly progress_denominator: number;
  readonly unix_seconds: number;
  readonly measurements: readonly TelemetryMeasurement[];
}

export interface CheckpointManifest {
  readonly name: string;
  readonly kind: "latest" | "archive" | "invalid";
  readonly valid: boolean;
  readonly error: string | null;
  readonly checkpoint_id: string | null;
  readonly state_path: string | null;
  readonly state_exists: boolean;
  readonly state_size_bytes: number | null;
  readonly modified_at_ms: number | null;
  readonly state_modified_at_ms: number | null;
  readonly state_sha256: string | null;
  readonly total_rounds: number | null;
  readonly total_samples: number | null;
  readonly total_updates: number | null;
  readonly model_config_values:
    | Readonly<Record<string, JsonPrimitive>>
    | null;
  readonly train_config_values:
    | Readonly<Record<string, JsonPrimitive>>
    | null;
}

export interface CheckpointObject {
  readonly checkpoint_id: string;
  readonly state_path: string;
  readonly valid: boolean;
  readonly error: string | null;
  readonly state_size_bytes: number | null;
  readonly state_modified_at_ms: number | null;
  readonly referenced_by: readonly string[];
  readonly orphan: boolean;
}

export interface CheckpointCatalog {
  readonly checkpoint_directory: string;
  readonly manifests: readonly CheckpointManifest[];
  readonly objects: readonly CheckpointObject[];
  readonly total_unique_state_bytes: number;
}

export interface TrainingStreamSnapshot {
  readonly type: "snapshot";
  readonly summary: TrainingSummary;
  readonly log_stream: "stdout" | "stderr" | null;
  readonly log_content: string | null;
}

export interface TrainingSummary extends TrainingRunStatus {
  readonly schema_version: 1;
  readonly run_dir: string;
  readonly metrics: readonly TrainingMetric[];
  readonly telemetry: readonly TelemetryEvent[];
  readonly checkpoints: CheckpointCatalog;
}

export interface TrainingStreamError {
  readonly type: "error";
  readonly message: string;
}

export type TrainingStreamMessage =
  | TrainingStreamSnapshot
  | TrainingStreamError;

type JsonPrimitive = string | number | boolean | null;

export function parseConfig(value: unknown): TrainingConfig {
  const record = requiredRecord(value, "training config");
  return {
    default_run_dir: requiredString(
      record.default_run_dir,
      "default_run_dir",
    ),
    stop_timeout_seconds: requiredNumber(
      record.stop_timeout_seconds,
      "stop_timeout_seconds",
    ),
  };
}

export function parseProcess(value: unknown): TrainingProcess | null {
  const envelope = requiredRecord(value, "process response");
  if (envelope.process === null) return null;
  const record = requiredRecord(envelope.process, "training process");
  return {
    pid: requiredNumber(record.pid, "pid"),
    name: requiredString(record.name, "name"),
    kernel_state: requiredString(record.kernel_state, "kernel_state"),
    executable: requiredString(record.executable, "executable"),
    working_directory: requiredString(
      record.working_directory,
      "working_directory",
    ),
    run_dir: requiredString(
      record.run_dir,
      "run_dir",
    ),
    argv: stringArray(record.argv, "argv"),
    process_group_id: requiredNumber(
      record.process_group_id,
      "process_group_id",
    ),
    session_id: requiredNumber(record.session_id, "session_id"),
    start_ticks: requiredNumber(record.start_ticks, "start_ticks"),
  };
}

export function parseMetrics(
  value: unknown,
): readonly TrainingMetric[] {
  const envelope = requiredRecord(value, "metrics response");
  if (!Array.isArray(envelope.records)) {
    throw new Error("Invalid metrics records");
  }
  return envelope.records.map((item) => {
    const record = requiredRecord(item, "metric");
    return record as unknown as TrainingMetric;
  });
}

export function parseTelemetry(
  value: unknown,
): readonly TelemetryEvent[] {
  const envelope = requiredRecord(value, "telemetry response");
  if (!Array.isArray(envelope.records)) {
    throw new Error("Invalid telemetry records");
  }
  return envelope.records.map((item) => {
    const record = requiredRecord(item, "telemetry event");
    if (!Array.isArray(record.measurements)) {
      throw new Error("Invalid telemetry measurements");
    }
    return record as unknown as TelemetryEvent;
  });
}

export function parseCheckpointCatalog(
  value: unknown,
): CheckpointCatalog {
  const record = requiredRecord(value, "checkpoint catalog");
  if (
    !Array.isArray(record.manifests) || !Array.isArray(record.objects)
  ) {
    throw new Error("Invalid checkpoint catalog");
  }
  return record as unknown as CheckpointCatalog;
}

export function parseTrainingStreamMessage(
  value: unknown,
): TrainingStreamMessage {
  const record = requiredRecord(value, "training stream message");
  if (record.type === "error") {
    return {
      type: "error",
      message: requiredString(record.message, "message"),
    };
  }
  if (record.type !== "snapshot") {
    throw new Error("Invalid training stream message type");
  }
  const logStream = optionalLogStream(record.log_stream);
  const logContent = optionalString(record.log_content, "log_content");
  if ((logStream === null) !== (logContent === null)) {
    throw new Error("Invalid training stream log subscription");
  }
  return {
    type: "snapshot",
    summary: parseSummary(record.summary),
    log_stream: logStream,
    log_content: logContent,
  };
}

function parseSummary(value: unknown): TrainingSummary {
  const record = requiredRecord(value, "training summary");
  if (record.schema_version !== 1) {
    throw new Error("Unsupported training summary schema");
  }
  return {
    ...parseRunStatus(record),
    schema_version: 1,
    run_dir: requiredString(record.run_dir, "run_dir"),
    metrics: parseMetrics({ records: record.metrics }),
    telemetry: parseTelemetry({ records: record.telemetry }),
    checkpoints: parseCheckpointCatalog(record.checkpoints),
  };
}

function parseRunStatus(value: unknown): TrainingRunStatus {
  const record = requiredRecord(value, "training run status");
  const state = record.state;
  if (
    state !== "NOT_INITIALIZED" && state !== "BROKEN" &&
    state !== "READY" && state !== "RUNNING"
  ) {
    throw new Error("Invalid training run state");
  }
  const reason = optionalString(record.reason, "reason");
  const process = parseProcess({ process: record.process });
  const details = record.details === null
    ? null
    : parseRunDetails(record.details);
  return { state, reason, process, details };
}

function parseRunDetails(value: unknown): TrainingRunDetails {
  const record = requiredRecord(value, "training run details");
  requiredString(record.checkpoint_id, "checkpoint_id");
  requiredString(record.checkpoint_path, "checkpoint_path");
  requiredNumber(record.state_size_bytes, "state_size_bytes");
  requiredRecord(record.model_config_values, "model_config_values");
  requiredRecord(record.train_config_values, "train_config_values");
  requiredNumber(record.total_rounds, "total_rounds");
  requiredNumber(record.total_samples, "total_samples");
  requiredNumber(record.total_updates, "total_updates");
  requiredNumber(record.metric_count, "metric_count");
  return record as unknown as TrainingRunDetails;
}

function requiredRecord(
  value: unknown,
  label: string,
): Record<string, unknown> {
  const record = recordValue(value);
  if (record === null) throw new Error(`Invalid ${label}`);
  return record;
}

function requiredString(value: unknown, field: string): string {
  if (typeof value !== "string") throw new Error(`Invalid ${field}`);
  return value;
}

function requiredNumber(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Invalid ${field}`);
  }
  return value;
}

function optionalString(
  value: unknown,
  field: string,
): string | null {
  if (value === null) return null;
  return requiredString(value, field);
}

function optionalLogStream(
  value: unknown,
): "stdout" | "stderr" | null {
  if (value === null || value === "stdout" || value === "stderr") {
    return value;
  }
  throw new Error("Invalid log_stream");
}

function stringArray(value: unknown, field: string): readonly string[] {
  if (
    !Array.isArray(value) ||
    !value.every((item) => typeof item === "string")
  ) {
    throw new Error(`Invalid ${field}`);
  }
  return value;
}
