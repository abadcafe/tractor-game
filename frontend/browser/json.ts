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
