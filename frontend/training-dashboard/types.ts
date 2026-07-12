import { recordValue } from "../browser/json.ts";

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonArray | JsonObject;
export interface JsonArray extends ReadonlyArray<JsonValue> {}
export interface JsonObject {
  readonly [key: string]: JsonValue;
}

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
  readonly run_dir: string | null;
  readonly argv: readonly string[];
  readonly process_group_id: number;
  readonly session_id: number;
  readonly start_ticks: number;
}

export interface TrainingRunDetails {
  readonly checkpoint_id: string;
  readonly checkpoint_path: string;
  readonly state_size_bytes: number;
  readonly model_config_values: Readonly<Record<string, JsonPrimitive>>;
  readonly train_config_values: Readonly<Record<string, JsonPrimitive>>;
  readonly total_rounds: number;
  readonly total_samples: number;
  readonly total_updates: number;
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

export interface TrainingSummary {
  readonly schema_version: 2;
  readonly run_dir: string;
  readonly state: "NOT_INITIALIZED" | "BROKEN" | "READY" | "RUNNING";
  readonly reason: string | null;
  readonly process: TrainingProcess | null;
  readonly details: TrainingRunDetails | null;
  readonly checkpoints: CheckpointCatalog;
}

export interface MetricPoint {
  readonly update: number | null;
  readonly elapsed_seconds: number;
  readonly recorded_at_ms: number;
  readonly values: Readonly<Record<string, JsonPrimitive>>;
}

export interface MetricDatasets {
  readonly throughput: readonly MetricPoint[];
  readonly optimization: readonly MetricPoint[];
  readonly ppo_timing: readonly MetricPoint[];
  readonly rollout: readonly MetricPoint[];
  readonly rewards: readonly MetricPoint[];
  readonly inference: readonly MetricPoint[];
  readonly processes: readonly MetricPoint[];
}

export interface TrainingMetrics {
  readonly schema_version: 1;
  readonly through_sequence: number;
  readonly session_id: string | null;
  readonly sessions: readonly {
    readonly session_id: string;
    readonly started_at_ms: number;
  }[];
  readonly complete: boolean;
  readonly dropped_event_count: number;
  readonly totals: Readonly<Record<string, JsonPrimitive>>;
  readonly datasets: MetricDatasets;
}

export interface TrainingEvent {
  readonly schema_version: number;
  readonly event: string;
  readonly level: "DEBUG" | "INFO" | "WARNING" | "ERROR";
  readonly recorded_at_ms: number;
  readonly session_id: string | null;
  readonly process: JsonObject;
  readonly context: JsonObject;
  readonly fields: JsonObject;
}

export type TrainingLogMessage =
  | { readonly type: "reset"; readonly window: number }
  | {
    readonly type: "event";
    readonly sequence: number;
    readonly event: TrainingEvent;
  }
  | { readonly type: "error"; readonly message: string };

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
    run_dir: nullableString(record.run_dir, "run_dir"),
    argv: stringArray(record.argv, "argv"),
    process_group_id: requiredNumber(
      record.process_group_id,
      "process_group_id",
    ),
    session_id: requiredNumber(record.session_id, "session_id"),
    start_ticks: requiredNumber(record.start_ticks, "start_ticks"),
  };
}

export function parseSummary(value: unknown): TrainingSummary {
  return value as TrainingSummary;
}

export function parseMetrics(value: unknown): TrainingMetrics {
  return value as TrainingMetrics;
}

export function parseLogMessage(value: unknown): TrainingLogMessage {
  const record = requiredRecord(value, "training log message");
  const type = requiredString(record.type, "type");
  if (type === "reset") {
    return { type, window: requiredNumber(record.window, "window") };
  }
  if (type === "error") {
    return { type, message: requiredString(record.message, "message") };
  }
  if (type !== "event") throw new Error("Unknown training log message");
  return {
    type,
    sequence: requiredNumber(record.sequence, "sequence"),
    event: record.event as TrainingEvent,
  };
}

function requiredRecord(
  value: unknown,
  label: string,
): Record<string, unknown> {
  const record = recordValue(value);
  if (record === null) throw new Error(`Invalid ${label}`);
  return record;
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== "string") throw new Error(`Invalid ${label}`);
  return value;
}

function nullableString(value: unknown, label: string): string | null {
  if (value === null) return null;
  return requiredString(value, label);
}

function requiredNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Invalid ${label}`);
  }
  return value;
}

function stringArray(value: unknown, label: string): readonly string[] {
  if (
    !Array.isArray(value) ||
    !value.every((item) => typeof item === "string")
  ) {
    throw new Error(`Invalid ${label}`);
  }
  return value;
}
