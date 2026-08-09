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
  readonly policy_path: string | null;
  readonly policy_exists: boolean;
  readonly policy_size_bytes: number | null;
  readonly trainer_path: string | null;
  readonly trainer_exists: boolean;
  readonly trainer_size_bytes: number | null;
  readonly modified_at_ms: number | null;
  readonly policy_modified_at_ms: number | null;
  readonly trainer_modified_at_ms: number | null;
  readonly policy_sha256: string | null;
  readonly trainer_sha256: string | null;
  readonly total_rounds: number | null;
  readonly total_trainable_decisions: number | null;
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
  readonly policy_path: string;
  readonly trainer_path: string;
  readonly valid: boolean;
  readonly error: string | null;
  readonly policy_size_bytes: number | null;
  readonly trainer_size_bytes: number | null;
  readonly policy_modified_at_ms: number | null;
  readonly trainer_modified_at_ms: number | null;
  readonly referenced_by: readonly string[];
  readonly orphan: boolean;
}

export interface CheckpointCatalog {
  readonly checkpoint_directory: string;
  readonly manifests: readonly CheckpointManifest[];
  readonly objects: readonly CheckpointObject[];
  readonly total_unique_payload_bytes: number;
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
      "total_unique_payload_bytes",
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
    total_unique_payload_bytes: nonNegativeInteger(
      record.total_unique_payload_bytes,
      "total_unique_payload_bytes",
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
      "policy_path",
      "policy_exists",
      "policy_size_bytes",
      "trainer_path",
      "trainer_exists",
      "trainer_size_bytes",
      "modified_at_ms",
      "policy_modified_at_ms",
      "trainer_modified_at_ms",
      "policy_sha256",
      "trainer_sha256",
      "total_rounds",
      "total_trainable_decisions",
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
    policy_path: nullableString(record.policy_path, "policy_path"),
    policy_exists: requiredBoolean(
      record.policy_exists,
      "policy_exists",
    ),
    policy_size_bytes: nullableNonNegativeInteger(
      record.policy_size_bytes,
      "policy_size_bytes",
    ),
    trainer_path: nullableString(record.trainer_path, "trainer_path"),
    trainer_exists: requiredBoolean(
      record.trainer_exists,
      "trainer_exists",
    ),
    trainer_size_bytes: nullableNonNegativeInteger(
      record.trainer_size_bytes,
      "trainer_size_bytes",
    ),
    modified_at_ms: nullableNonNegativeInteger(
      record.modified_at_ms,
      "modified_at_ms",
    ),
    policy_modified_at_ms: nullableNonNegativeInteger(
      record.policy_modified_at_ms,
      "policy_modified_at_ms",
    ),
    trainer_modified_at_ms: nullableNonNegativeInteger(
      record.trainer_modified_at_ms,
      "trainer_modified_at_ms",
    ),
    policy_sha256: nullableString(
      record.policy_sha256,
      "policy_sha256",
    ),
    trainer_sha256: nullableString(
      record.trainer_sha256,
      "trainer_sha256",
    ),
    total_rounds: nullableNonNegativeInteger(
      record.total_rounds,
      "total_rounds",
    ),
    total_trainable_decisions: nullableNonNegativeInteger(
      record.total_trainable_decisions,
      "total_trainable_decisions",
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
      "policy_path",
      "trainer_path",
      "valid",
      "error",
      "policy_size_bytes",
      "trainer_size_bytes",
      "policy_modified_at_ms",
      "trainer_modified_at_ms",
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
    policy_path: requiredString(record.policy_path, "policy_path"),
    trainer_path: requiredString(record.trainer_path, "trainer_path"),
    valid: requiredBoolean(record.valid, "valid"),
    error: nullableString(record.error, "error"),
    policy_size_bytes: nullableNonNegativeInteger(
      record.policy_size_bytes,
      "policy_size_bytes",
    ),
    trainer_size_bytes: nullableNonNegativeInteger(
      record.trainer_size_bytes,
      "trainer_size_bytes",
    ),
    policy_modified_at_ms: nullableNonNegativeInteger(
      record.policy_modified_at_ms,
      "policy_modified_at_ms",
    ),
    trainer_modified_at_ms: nullableNonNegativeInteger(
      record.trainer_modified_at_ms,
      "trainer_modified_at_ms",
    ),
    referenced_by: stringArray(record.referenced_by, "referenced_by"),
    orphan: requiredBoolean(record.orphan, "orphan"),
  };
}
