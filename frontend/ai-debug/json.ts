import type {
  JsonParseResult,
  SeatId,
  TranscriptRecord,
} from "./types.ts";
import { recordValue } from "../browser/json.ts";

export { recordValue } from "../browser/json.ts";

export function firstRecord(
  value: unknown,
): Record<string, unknown> | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  return recordValue(value[0]);
}

export function valueAt(value: unknown, key: string): unknown {
  const record = recordValue(value);
  return record === null ? undefined : record[key];
}

export function textValue(value: unknown): string {
  if (value === undefined || value === null) return "<empty>";
  if (typeof value === "string") {
    return value === "" ? "<empty>" : value;
  }
  return String(value);
}

export function stringify(value: unknown): string {
  if (value === undefined) return "<missing>";
  if (typeof value === "string") {
    return value === "" ? "<empty>" : value;
  }
  return JSON.stringify(value, null, 2);
}

export function compactJson(value: unknown): string {
  if (value === undefined) return "<missing>";
  if (typeof value === "string") {
    return value === "" ? "<empty>" : value;
  }
  return JSON.stringify(value);
}

export function parseJson(raw: string): JsonParseResult {
  try {
    return { ok: true, value: JSON.parse(raw) as unknown };
  } catch (_error) {
    return { ok: false };
  }
}

export function transcriptRecord(
  value: unknown,
): TranscriptRecord | null {
  const record = recordValue(value);
  if (record === null) return null;
  const id = record.id;
  const eventId = record.event_id;
  const createdAt = record.created_at;
  const seat = record.seat;
  const seq = record.seq;
  const attempt = record.attempt;
  if (
    typeof id !== "number" || typeof eventId !== "number" ||
    typeof createdAt !== "string" || !isSeatId(seat) ||
    typeof seq !== "number" || typeof attempt !== "number"
  ) {
    return null;
  }
  const apiRequest = nullableString(record.api_request);
  const apiResponse = nullableString(record.api_response);
  const apiError = nullableString(record.api_error);
  const toolResult = nullableString(record.tool_result);
  if (
    apiRequest === undefined || apiResponse === undefined ||
    apiError === undefined || toolResult === undefined
  ) {
    return null;
  }
  return {
    id,
    event_id: eventId,
    created_at: createdAt,
    seat,
    seq,
    attempt,
    api_request: apiRequest,
    api_response: apiResponse,
    api_error: apiError,
    tool_result: toolResult,
  };
}

function isSeatId(value: unknown): value is SeatId {
  return value === "a" || value === "b" || value === "c" ||
    value === "d";
}

function nullableString(value: unknown): string | null | undefined {
  if (value === null) return null;
  if (typeof value === "string") return value;
  return undefined;
}
