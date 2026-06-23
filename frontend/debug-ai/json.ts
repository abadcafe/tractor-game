import type { JsonParseResult, TranscriptRecord } from "./types.ts";

export function firstRecord(
  value: unknown,
): Record<string, unknown> | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  return recordValue(value[0]);
}

export function recordValue(
  value: unknown,
): Record<string, unknown> | null {
  if (
    value !== null && typeof value === "object" &&
    !Array.isArray(value)
  ) {
    return value as Record<string, unknown>;
  }
  return null;
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
  const playerIndex = record.player_index;
  const seq = record.seq;
  const attempt = record.attempt;
  if (
    typeof id !== "number" || typeof eventId !== "number" ||
    typeof createdAt !== "string" || typeof playerIndex !== "number" ||
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
    player_index: playerIndex,
    seq,
    attempt,
    api_request: apiRequest,
    api_response: apiResponse,
    api_error: apiError,
    tool_result: toolResult,
  };
}

function nullableString(value: unknown): string | null | undefined {
  if (value === null) return null;
  if (typeof value === "string") return value;
  return undefined;
}
