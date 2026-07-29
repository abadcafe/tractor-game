import { type JsonPrimitive, nullablePrimitiveRecord } from "./json.ts";
import {
  nonNegativeInteger,
  nullableNonNegativeInteger,
  nullableString,
  rejectUnknownKeys,
  requiredArray,
  requiredBoolean,
  requiredRecord,
  requiredString,
  stringArray,
} from "./wire.ts";

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

export function parseCheckpointCatalog(
  value: unknown,
): CheckpointCatalog {
  const record = requiredRecord(value, "checkpoint catalog");
  rejectUnknownKeys(
    record,
    [
      "checkpoint_directory",
      "manifests",
      "objects",
      "total_unique_state_bytes",
    ],
    "checkpoint catalog",
  );
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

function parseCheckpointManifest(value: unknown): CheckpointManifest {
  const record = requiredRecord(value, "checkpoint manifest");
  rejectUnknownKeys(
    record,
    [
      "name",
      "kind",
      "valid",
      "error",
      "checkpoint_id",
      "state_path",
      "state_exists",
      "state_size_bytes",
      "modified_at_ms",
      "state_modified_at_ms",
      "state_sha256",
      "total_rounds",
      "total_samples",
      "total_updates",
      "model_config_values",
      "train_config_values",
    ],
    "checkpoint manifest",
  );
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
  rejectUnknownKeys(
    record,
    [
      "checkpoint_id",
      "state_path",
      "valid",
      "error",
      "state_size_bytes",
      "state_modified_at_ms",
      "referenced_by",
      "orphan",
    ],
    "checkpoint object",
  );
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
