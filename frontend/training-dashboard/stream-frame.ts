import { recordValue } from "../browser/json.ts";

export type TrainingStreamFrame =
  | { readonly type: "message"; readonly value: unknown }
  | { readonly type: "rejected"; readonly error: string };

export function parseTrainingStreamFrame(
  value: unknown,
): TrainingStreamFrame {
  const record = recordValue(value);
  if (record?.type !== "rejected") {
    return { type: "message", value };
  }
  const keys = Object.keys(record).sort();
  if (keys.length !== 2 || keys[0] !== "error" || keys[1] !== "type") {
    throw new Error("Invalid training stream rejection");
  }
  if (typeof record.error !== "string" || record.error.trim() === "") {
    throw new Error("Invalid training stream rejection error");
  }
  return { type: "rejected", error: record.error };
}
