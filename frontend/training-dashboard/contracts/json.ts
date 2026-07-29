import { recordValue } from "../../browser/json.ts";
import { requiredRecord } from "./wire.ts";

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonArray | JsonObject;

export interface JsonArray extends ReadonlyArray<JsonValue> {}

export interface JsonObject {
  readonly [key: string]: JsonValue;
}

export function jsonObject(
  value: unknown,
  label: string,
): JsonObject {
  const record = requiredRecord(value, label);
  return Object.fromEntries(
    Object.entries(record).map(([key, item]) => [
      key,
      jsonValue(item, `${label}.${key}`),
    ]),
  );
}

export function primitiveRecord(
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

export function nullablePrimitiveRecord(
  value: unknown,
  label: string,
): Readonly<Record<string, JsonPrimitive>> | null {
  if (value === null) return null;
  return primitiveRecord(value, label);
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

function jsonPrimitive(
  value: unknown,
  label: string,
): JsonPrimitive {
  if (
    value === null || typeof value === "string" ||
    typeof value === "boolean"
  ) return value;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  throw new Error(`Invalid JSON primitive: ${label}`);
}
