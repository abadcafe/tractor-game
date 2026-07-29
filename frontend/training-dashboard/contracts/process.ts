import {
  nonNegativeInteger,
  positiveInteger,
  rejectUnknownKeys,
  requiredRecord,
  requiredString,
  stringArray,
} from "./wire.ts";

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

export function parseProcessState(value: unknown): ProcessState {
  const record = requiredRecord(value, "process state");
  rejectUnknownKeys(record, ["process"], "process state");
  return {
    process: record.process === null
      ? null
      : parseProcessSnapshot(record.process),
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
