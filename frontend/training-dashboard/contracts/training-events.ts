import {
  TRAINING_EVENT_NAMES,
  TRAINING_EVENT_SCHEMA_VERSION,
  TRAINING_PROCESS_KINDS,
} from "./generated.ts";
import { nullableStoreId } from "./event-store.ts";
import { type JsonObject, jsonObject } from "./json.ts";
import {
  nonEmptyString,
  nonNegativeInteger,
  nullableNonNegativeInteger,
  nullablePositiveInteger,
  positiveInteger,
  rejectUnknownKeys,
  requiredArray,
  requiredRecord,
  requiredString,
} from "./wire.ts";

export type ProcessKind = (typeof TRAINING_PROCESS_KINDS)[number];
export type TrainingEventName = (typeof TRAINING_EVENT_NAMES)[number];

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
  readonly round_id?: number;
  readonly batch_id?: number;
}

export interface TrainingEvent {
  readonly schema_version: typeof TRAINING_EVENT_SCHEMA_VERSION;
  readonly event: TrainingEventName;
  readonly recorded_at_ms: number;
  readonly process: EventProcess;
  readonly context: EventContext;
  readonly fields: JsonObject;
  readonly error?: string;
}

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

export function parseLogPage(value: unknown): TrainingLogPage {
  const record = requiredRecord(value, "training log page");
  rejectUnknownKeys(
    record,
    ["store_id", "events", "next_before_sequence"],
    "training log page",
  );
  return {
    store_id: nullableStoreId(record.store_id),
    events: requiredArray(record.events, "events").map(parseLogEntry),
    next_before_sequence: nullablePositiveInteger(
      record.next_before_sequence,
      "next_before_sequence",
    ),
  };
}

export function parseLogEntry(value: unknown): TrainingLogEntry {
  const record = requiredRecord(value, "training log entry");
  rejectUnknownKeys(
    record,
    ["sequence", "event"],
    "training log entry",
  );
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
  if (record.schema_version !== TRAINING_EVENT_SCHEMA_VERSION) {
    throw new Error("Unsupported training event schema");
  }
  const error = record.error === undefined
    ? undefined
    : nonEmptyString(record.error, "error");
  const event = requiredString(record.event, "event");
  if (!isTrainingEventName(event)) {
    throw new Error(`Unknown training event: ${event}`);
  }
  const process = parseEventProcess(record.process);
  const context = parseEventContext(record.context);
  validateRoundIdentity(event, process, context);
  return {
    schema_version: TRAINING_EVENT_SCHEMA_VERSION,
    event,
    recorded_at_ms: nonNegativeInteger(
      record.recorded_at_ms,
      "recorded_at_ms",
    ),
    process,
    context,
    fields: jsonObject(record.fields, "fields"),
    ...(error === undefined ? {} : { error }),
  };
}

function validateRoundIdentity(
  event: TrainingEventName,
  process: EventProcess,
  context: EventContext,
): void {
  if (event !== "round") {
    if (context.round_id !== undefined) {
      throw new Error("Only round events may carry round_id");
    }
    return;
  }
  if (
    process.kind !== "worker" ||
    process.index === null ||
    context.policy_version === undefined ||
    context.rollout_id === undefined ||
    context.worker_index !== process.index ||
    context.game_env_index === undefined ||
    context.round_id === undefined
  ) {
    throw new Error("Round event identity is incomplete");
  }
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
    "round_id",
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
  return TRAINING_PROCESS_KINDS.some((kind) => kind === value);
}

function isTrainingEventName(
  value: string,
): value is TrainingEventName {
  return TRAINING_EVENT_NAMES.some((event) => event === value);
}
