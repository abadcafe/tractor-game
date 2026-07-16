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

export interface ProcessDetails {
  readonly kind: "details";
  readonly started_at_ms: number;
  readonly kernel_state: string;
  readonly executable: string;
  readonly working_directory: string;
  readonly argv: readonly string[];
  readonly process_group_id: number;
  readonly unix_session_id: number;
}

export interface ProcessInspectionError {
  readonly kind: "error";
  readonly error: string;
}

export interface ProcessSnapshot {
  readonly pid: number;
  readonly inspection: ProcessDetails | ProcessInspectionError;
}

export interface ProcessState {
  readonly process: ProcessSnapshot | null;
}

export interface StopResult {
  readonly forced: boolean;
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

export interface MetricPoint {
  readonly sequence: number;
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
  readonly schema_version: number;
  readonly store_id: string | null;
  readonly through_sequence: number;
  readonly complete: boolean;
  readonly dropped_event_count: number;
  readonly totals: Readonly<Record<string, JsonPrimitive>>;
  readonly datasets: MetricDatasets;
}

export interface CheckpointStreamMessage {
  readonly type: "invalidation" | "replacement";
  readonly store_id: string | null;
  readonly through_sequence: number;
}

export interface TrainingEvent {
  readonly schema_version: 2;
  readonly event: TrainingEventName;
  readonly recorded_at_ms: number;
  readonly process: EventProcess;
  readonly context: EventContext;
  readonly fields: JsonObject;
  readonly error?: string;
}

export type ProcessKind =
  | "initializer"
  | "coordinator"
  | "worker"
  | "model_rank";

export interface EventProcess {
  readonly kind: ProcessKind;
  readonly index: number | null;
  readonly pid: number;
}

export interface EventContext {
  readonly policy_version?: number;
  readonly rollout_id?: string;
  readonly worker_index?: number;
  readonly model_rank_index?: number;
  readonly game_env_index?: number;
  readonly episode_id?: number;
  readonly player_index?: number;
  readonly decision_index?: number;
  readonly request_id?: number;
  readonly batch_id?: number;
}

export type TrainingEventName =
  | "initialize"
  | "training"
  | "process.start"
  | "process.stop"
  | "rollout"
  | "sampling"
  | "round"
  | "update"
  | "update.rank"
  | "checkpoint"
  | "inference.batch"
  | "decision"
  | "logging.drop";

export interface TrainingLogEntry {
  readonly sequence: number;
  readonly event: TrainingEvent;
}

export interface TrainingLogPage {
  readonly store_id: string | null;
  readonly events: readonly TrainingLogEntry[];
  readonly next_before_sequence: number | null;
}

export type TrainingLogMessage =
  | {
    readonly type: "event";
    readonly sequence: number;
    readonly event: TrainingEvent;
  }
  | { readonly type: "replacement"; readonly store_id: string | null };

export function parseConfig(value: unknown): TrainingConfig {
  const record = requiredRecord(value, "training config");
  return {
    default_run_dir: requiredString(
      record.default_run_dir,
      "default_run_dir",
    ),
    stop_timeout_seconds: nonNegativeNumber(
      record.stop_timeout_seconds,
      "stop_timeout_seconds",
    ),
  };
}

export function parseProcessState(value: unknown): ProcessState {
  const record = requiredRecord(value, "process state");
  rejectUnknownKeys(record, ["process"], "process state");
  return {
    process: record.process === null
      ? null
      : parseProcessSnapshot(record.process),
  };
}

export function parseStopResult(value: unknown): StopResult {
  const record = requiredRecord(value, "stop result");
  rejectUnknownKeys(record, ["forced"], "stop result");
  return {
    forced: requiredBoolean(record.forced, "forced"),
  };
}

export function parseCheckpointCatalog(
  value: unknown,
): CheckpointCatalog {
  const record = requiredRecord(value, "checkpoint catalog");
  return {
    checkpoint_directory: requiredString(
      record.checkpoint_directory,
      "checkpoint_directory",
    ),
    manifests: requiredArray(record.manifests, "manifests").map(
      parseCheckpointManifest,
    ),
    objects: requiredArray(record.objects, "objects").map(
      parseCheckpointObject,
    ),
    total_unique_state_bytes: nonNegativeInteger(
      record.total_unique_state_bytes,
      "total_unique_state_bytes",
    ),
  };
}

export function parseMetrics(value: unknown): TrainingMetrics {
  const record = requiredRecord(value, "training metrics");
  if (record.schema_version !== 2) {
    throw new Error("Unsupported training metrics schema");
  }
  const datasets = requiredRecord(record.datasets, "metric datasets");
  return {
    schema_version: 2,
    store_id: nullableStoreId(record.store_id),
    through_sequence: nonNegativeInteger(
      record.through_sequence,
      "through_sequence",
    ),
    complete: requiredBoolean(record.complete, "complete"),
    dropped_event_count: nonNegativeInteger(
      record.dropped_event_count,
      "dropped_event_count",
    ),
    totals: primitiveRecord(record.totals, "totals"),
    datasets: {
      throughput: metricPoints(datasets.throughput, "throughput"),
      optimization: metricPoints(
        datasets.optimization,
        "optimization",
      ),
      ppo_timing: metricPoints(datasets.ppo_timing, "ppo_timing"),
      rollout: metricPoints(datasets.rollout, "rollout"),
      rewards: metricPoints(datasets.rewards, "rewards"),
      inference: metricPoints(datasets.inference, "inference"),
      processes: metricPoints(datasets.processes, "processes"),
    },
  };
}

export function parseLogPage(value: unknown): TrainingLogPage {
  const record = requiredRecord(value, "training log page");
  if (!Array.isArray(record.events)) {
    throw new Error("Invalid training log page events");
  }
  return {
    store_id: nullableStoreId(record.store_id),
    events: record.events.map(parseLogEntry),
    next_before_sequence: nullablePositiveInteger(
      record.next_before_sequence,
      "next_before_sequence",
    ),
  };
}

export function parseLogMessage(value: unknown): TrainingLogMessage {
  const record = requiredRecord(value, "training log message");
  const type = requiredString(record.type, "type");
  if (type === "replacement") {
    return {
      type,
      store_id: nullableStoreId(record.store_id),
    };
  }
  if (type !== "event") throw new Error("Unknown training log message");
  const entry = parseLogEntry(record);
  return { type, ...entry };
}

export function parseCheckpointStreamMessage(
  value: unknown,
): CheckpointStreamMessage {
  const record = requiredRecord(value, "checkpoint stream message");
  const type = requiredString(record.type, "type");
  if (type !== "invalidation" && type !== "replacement") {
    throw new Error("Unknown checkpoint stream message");
  }
  return {
    type,
    store_id: nullableStoreId(record.store_id),
    through_sequence: nonNegativeInteger(
      record.through_sequence,
      "through_sequence",
    ),
  };
}

function parseProcessSnapshot(value: unknown): ProcessSnapshot {
  const record = requiredRecord(value, "process snapshot");
  rejectUnknownKeys(record, ["pid", "inspection"], "process snapshot");
  return {
    pid: positiveInteger(record.pid, "pid"),
    inspection: parseProcessInspection(record.inspection),
  };
}

function parseProcessInspection(
  value: unknown,
): ProcessDetails | ProcessInspectionError {
  const record = requiredRecord(value, "process inspection");
  const kind = requiredString(record.kind, "inspection.kind");
  if (kind === "error") {
    rejectUnknownKeys(record, ["kind", "error"], "process inspection");
    return {
      kind,
      error: requiredString(record.error, "inspection.error"),
    };
  }
  if (kind !== "details") {
    throw new Error("Invalid process inspection kind");
  }
  rejectUnknownKeys(
    record,
    [
      "kind",
      "started_at_ms",
      "kernel_state",
      "executable",
      "working_directory",
      "argv",
      "process_group_id",
      "unix_session_id",
    ],
    "process inspection",
  );
  return {
    kind,
    started_at_ms: nonNegativeInteger(
      record.started_at_ms,
      "started_at_ms",
    ),
    kernel_state: requiredString(record.kernel_state, "kernel_state"),
    executable: requiredString(record.executable, "executable"),
    working_directory: requiredString(
      record.working_directory,
      "working_directory",
    ),
    argv: stringArray(record.argv, "argv"),
    process_group_id: positiveInteger(
      record.process_group_id,
      "process_group_id",
    ),
    unix_session_id: positiveInteger(
      record.unix_session_id,
      "unix_session_id",
    ),
  };
}

function parseCheckpointManifest(value: unknown): CheckpointManifest {
  const record = requiredRecord(value, "checkpoint manifest");
  const kind = requiredString(record.kind, "kind");
  if (kind !== "latest" && kind !== "archive" && kind !== "invalid") {
    throw new Error("Invalid checkpoint manifest kind");
  }
  return {
    name: requiredString(record.name, "name"),
    kind,
    valid: requiredBoolean(record.valid, "valid"),
    error: nullableString(record.error, "error"),
    checkpoint_id: nullableString(
      record.checkpoint_id,
      "checkpoint_id",
    ),
    state_path: nullableString(record.state_path, "state_path"),
    state_exists: requiredBoolean(record.state_exists, "state_exists"),
    state_size_bytes: nullableNonNegativeInteger(
      record.state_size_bytes,
      "state_size_bytes",
    ),
    modified_at_ms: nullableNonNegativeInteger(
      record.modified_at_ms,
      "modified_at_ms",
    ),
    state_modified_at_ms: nullableNonNegativeInteger(
      record.state_modified_at_ms,
      "state_modified_at_ms",
    ),
    state_sha256: nullableString(
      record.state_sha256,
      "state_sha256",
    ),
    total_rounds: nullableNonNegativeInteger(
      record.total_rounds,
      "total_rounds",
    ),
    total_samples: nullableNonNegativeInteger(
      record.total_samples,
      "total_samples",
    ),
    total_updates: nullableNonNegativeInteger(
      record.total_updates,
      "total_updates",
    ),
    model_config_values: nullablePrimitiveRecord(
      record.model_config_values,
      "model_config_values",
    ),
    train_config_values: nullablePrimitiveRecord(
      record.train_config_values,
      "train_config_values",
    ),
  };
}

function parseCheckpointObject(value: unknown): CheckpointObject {
  const record = requiredRecord(value, "checkpoint object");
  return {
    checkpoint_id: requiredString(
      record.checkpoint_id,
      "checkpoint_id",
    ),
    state_path: requiredString(record.state_path, "state_path"),
    valid: requiredBoolean(record.valid, "valid"),
    error: nullableString(record.error, "error"),
    state_size_bytes: nullableNonNegativeInteger(
      record.state_size_bytes,
      "state_size_bytes",
    ),
    state_modified_at_ms: nullableNonNegativeInteger(
      record.state_modified_at_ms,
      "state_modified_at_ms",
    ),
    referenced_by: stringArray(record.referenced_by, "referenced_by"),
    orphan: requiredBoolean(record.orphan, "orphan"),
  };
}

function metricPoints(
  value: unknown,
  label: string,
): readonly MetricPoint[] {
  return requiredArray(value, label).map((item) => {
    const record = requiredRecord(item, `${label} metric point`);
    return {
      sequence: positiveInteger(record.sequence, "sequence"),
      update: nullableNonNegativeInteger(record.update, "update"),
      elapsed_seconds: nonNegativeNumber(
        record.elapsed_seconds,
        "elapsed_seconds",
      ),
      recorded_at_ms: nonNegativeInteger(
        record.recorded_at_ms,
        "recorded_at_ms",
      ),
      values: primitiveRecord(record.values, "values"),
    };
  });
}

function parseLogEntry(value: unknown): TrainingLogEntry {
  const record = requiredRecord(value, "training log entry");
  return {
    sequence: positiveInteger(record.sequence, "sequence"),
    event: parseEvent(record.event),
  };
}

function parseEvent(value: unknown): TrainingEvent {
  const record = requiredRecord(value, "training event");
  rejectUnknownKeys(
    record,
    [
      "schema_version",
      "event",
      "recorded_at_ms",
      "process",
      "context",
      "fields",
      "error",
    ],
    "training event",
  );
  if (record.schema_version !== 2) {
    throw new Error("Unsupported training event schema");
  }
  const error = record.error === undefined
    ? undefined
    : nonEmptyString(record.error, "error");
  const event = requiredString(record.event, "event");
  if (!isTrainingEventName(event)) {
    throw new Error(`Unknown training event: ${event}`);
  }
  return {
    schema_version: 2,
    event,
    recorded_at_ms: nonNegativeInteger(
      record.recorded_at_ms,
      "recorded_at_ms",
    ),
    process: parseEventProcess(record.process),
    context: parseEventContext(record.context),
    fields: jsonObject(record.fields, "fields"),
    ...(error === undefined ? {} : { error }),
  };
}

function parseEventProcess(value: unknown): EventProcess {
  const record = requiredRecord(value, "event process");
  rejectUnknownKeys(record, ["kind", "index", "pid"], "event process");
  const kind = requiredString(record.kind, "process.kind");
  if (!isProcessKind(kind)) throw new Error("Invalid process.kind");
  return {
    kind,
    index: nullableNonNegativeInteger(record.index, "process.index"),
    pid: positiveInteger(record.pid, "process.pid"),
  };
}

function parseEventContext(value: unknown): EventContext {
  const record = requiredRecord(value, "event context");
  const numberKeys = [
    "policy_version",
    "worker_index",
    "model_rank_index",
    "game_env_index",
    "episode_id",
    "player_index",
    "decision_index",
    "request_id",
    "batch_id",
  ] as const;
  rejectUnknownKeys(
    record,
    [...numberKeys, "rollout_id"],
    "event context",
  );
  const result: Record<string, number | string> = {};
  for (const key of numberKeys) {
    if (record[key] !== undefined) {
      result[key] = nonNegativeInteger(record[key], `context.${key}`);
    }
  }
  if (record.rollout_id !== undefined) {
    result.rollout_id = nonEmptyString(
      record.rollout_id,
      "context.rollout_id",
    );
  }
  return result;
}

function isProcessKind(value: string): value is ProcessKind {
  return ["initializer", "coordinator", "worker", "model_rank"]
    .includes(value);
}

function isTrainingEventName(
  value: string,
): value is TrainingEventName {
  return [
    "initialize",
    "training",
    "process.start",
    "process.stop",
    "rollout",
    "sampling",
    "round",
    "update",
    "update.rank",
    "checkpoint",
    "inference.batch",
    "decision",
    "logging.drop",
  ].includes(value);
}

function requiredRecord(
  value: unknown,
  label: string,
): Record<string, unknown> {
  const record = recordValue(value);
  if (record === null) throw new Error(`Invalid ${label}`);
  return record;
}

function jsonObject(value: unknown, label: string): JsonObject {
  const record = requiredRecord(value, label);
  return Object.fromEntries(
    Object.entries(record).map(([key, item]) => [
      key,
      jsonValue(item, `${label}.${key}`),
    ]),
  );
}

function jsonValue(value: unknown, label: string): JsonValue {
  if (
    value === null || typeof value === "string" ||
    typeof value === "boolean"
  ) return value;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (Array.isArray(value)) {
    return value.map((item, index) =>
      jsonValue(item, `${label}[${index}]`)
    );
  }
  const record = recordValue(value);
  if (record !== null) return jsonObject(record, label);
  throw new Error(`Invalid JSON value: ${label}`);
}

function primitiveRecord(
  value: unknown,
  label: string,
): Readonly<Record<string, JsonPrimitive>> {
  const record = requiredRecord(value, label);
  return Object.fromEntries(
    Object.entries(record).map(([key, item]) => [
      key,
      jsonPrimitive(item, `${label}.${key}`),
    ]),
  );
}

function nullablePrimitiveRecord(
  value: unknown,
  label: string,
): Readonly<Record<string, JsonPrimitive>> | null {
  if (value === null) return null;
  return primitiveRecord(value, label);
}

function jsonPrimitive(value: unknown, label: string): JsonPrimitive {
  if (
    value === null || typeof value === "string" ||
    typeof value === "boolean"
  ) return value;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  throw new Error(`Invalid JSON primitive: ${label}`);
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== "string") throw new Error(`Invalid ${label}`);
  return value;
}

function nonEmptyString(value: unknown, label: string): string {
  const result = requiredString(value, label);
  if (result.trim() === "") throw new Error(`Invalid ${label}`);
  return result;
}

function nullableString(value: unknown, label: string): string | null {
  if (value === null) return null;
  return requiredString(value, label);
}

function nullableStoreId(value: unknown): string | null {
  if (value === null) return null;
  const result = requiredString(value, "store_id");
  if (!/^[0-9a-f]{32}$/.test(result)) {
    throw new Error("Invalid store_id");
  }
  return result;
}

function requiredNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Invalid ${label}`);
  }
  return value;
}

function requiredInteger(value: unknown, label: string): number {
  const result = requiredNumber(value, label);
  if (!Number.isInteger(result)) throw new Error(`Invalid ${label}`);
  return result;
}

function positiveInteger(value: unknown, label: string): number {
  const result = requiredInteger(value, label);
  if (result <= 0) throw new Error(`Invalid ${label}`);
  return result;
}

function nonNegativeInteger(value: unknown, label: string): number {
  const result = requiredInteger(value, label);
  if (result < 0) throw new Error(`Invalid ${label}`);
  return result;
}

function nonNegativeNumber(value: unknown, label: string): number {
  const result = requiredNumber(value, label);
  if (result < 0) throw new Error(`Invalid ${label}`);
  return result;
}

function nullableNonNegativeInteger(
  value: unknown,
  label: string,
): number | null {
  if (value === null) return null;
  return nonNegativeInteger(value, label);
}

function nullablePositiveInteger(
  value: unknown,
  label: string,
): number | null {
  if (value === null) return null;
  return positiveInteger(value, label);
}

function requiredBoolean(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") throw new Error(`Invalid ${label}`);
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

function requiredArray(
  value: unknown,
  label: string,
): readonly unknown[] {
  if (!Array.isArray(value)) throw new Error(`Invalid ${label}`);
  return value;
}

function rejectUnknownKeys(
  record: Readonly<Record<string, unknown>>,
  allowed: readonly string[],
  label: string,
): void {
  const expected = new Set(allowed);
  const unknown = Object.keys(record).find((key) => !expected.has(key));
  if (unknown !== undefined) {
    throw new Error(`Unknown ${label} field: ${unknown}`);
  }
}
