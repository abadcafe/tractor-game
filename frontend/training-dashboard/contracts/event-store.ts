import {
  nonNegativeInteger,
  rejectUnknownKeys,
  requiredRecord,
  requiredString,
} from "./wire.ts";

export interface CheckpointStreamMessage {
  readonly type: "invalidation" | "replacement";
  readonly store_id: string | null;
  readonly through_sequence: number;
}

export interface CheckpointCursor {
  readonly store_id: string | null;
  readonly through_sequence: number;
}

export interface StoreReplacement {
  readonly store_id: string | null;
}

export function parseStoreReplacement(
  value: unknown,
): StoreReplacement {
  const record = requiredRecord(value, "store replacement");
  rejectUnknownKeys(record, ["store_id"], "store replacement");
  return { store_id: nullableStoreId(record.store_id) };
}

export function parseCheckpointCursor(
  value: unknown,
): CheckpointCursor {
  const record = requiredRecord(value, "checkpoint cursor");
  rejectUnknownKeys(
    record,
    ["store_id", "through_sequence"],
    "checkpoint cursor",
  );
  return {
    store_id: nullableStoreId(record.store_id),
    through_sequence: nonNegativeInteger(
      record.through_sequence,
      "through_sequence",
    ),
  };
}

export function nullableStoreId(value: unknown): string | null {
  if (value === null) return null;
  const result = requiredString(value, "store_id");
  if (!/^[0-9a-f]{32}$/.test(result)) {
    throw new Error("Invalid store_id");
  }
  return result;
}
