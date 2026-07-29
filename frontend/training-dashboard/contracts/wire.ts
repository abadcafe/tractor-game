import { recordValue } from "../../browser/json.ts";

export function requiredRecord(
  value: unknown,
  label: string,
): Record<string, unknown> {
  const record = recordValue(value);
  if (record === null) throw new Error(`Invalid ${label}`);
  return record;
}

export function requiredString(
  value: unknown,
  label: string,
): string {
  if (typeof value !== "string") throw new Error(`Invalid ${label}`);
  return value;
}

export function nonEmptyString(
  value: unknown,
  label: string,
): string {
  const result = requiredString(value, label);
  if (result.trim() === "") throw new Error(`Invalid ${label}`);
  return result;
}

export function nullableString(
  value: unknown,
  label: string,
): string | null {
  if (value === null) return null;
  return requiredString(value, label);
}

export function positiveInteger(
  value: unknown,
  label: string,
): number {
  const result = requiredInteger(value, label);
  if (result <= 0) throw new Error(`Invalid ${label}`);
  return result;
}

export function nonNegativeInteger(
  value: unknown,
  label: string,
): number {
  const result = requiredInteger(value, label);
  if (result < 0) throw new Error(`Invalid ${label}`);
  return result;
}

export function nonNegativeNumber(
  value: unknown,
  label: string,
): number {
  const result = requiredNumber(value, label);
  if (result < 0) throw new Error(`Invalid ${label}`);
  return result;
}

export function nullableNonNegativeInteger(
  value: unknown,
  label: string,
): number | null {
  if (value === null) return null;
  return nonNegativeInteger(value, label);
}

export function nullablePositiveInteger(
  value: unknown,
  label: string,
): number | null {
  if (value === null) return null;
  return positiveInteger(value, label);
}

export function requiredBoolean(
  value: unknown,
  label: string,
): boolean {
  if (typeof value !== "boolean") throw new Error(`Invalid ${label}`);
  return value;
}

export function stringArray(
  value: unknown,
  label: string,
): readonly string[] {
  if (
    !Array.isArray(value) ||
    !value.every((item) => typeof item === "string")
  ) {
    throw new Error(`Invalid ${label}`);
  }
  return value;
}

export function requiredArray(
  value: unknown,
  label: string,
): readonly unknown[] {
  if (!Array.isArray(value)) throw new Error(`Invalid ${label}`);
  return value;
}

export function rejectUnknownKeys(
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
