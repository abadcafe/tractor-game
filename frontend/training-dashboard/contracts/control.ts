import {
  nonNegativeNumber,
  rejectUnknownKeys,
  requiredBoolean,
  requiredRecord,
  requiredString,
} from "./wire.ts";

export interface TrainingConfig {
  readonly default_run_dir: string;
  readonly stop_timeout_seconds: number;
}

export interface StopResult {
  readonly forced: boolean;
}

export function parseConfig(value: unknown): TrainingConfig {
  const record = requiredRecord(value, "training config");
  rejectUnknownKeys(
    record,
    ["default_run_dir", "stop_timeout_seconds"],
    "training config",
  );
  return {
    default_run_dir: requiredString(
      record.default_run_dir,
      "default_run_dir",
    ),
    stop_timeout_seconds: nonNegativeNumber(
      record.stop_timeout_seconds,
      "stop_timeout_seconds",
    ),
  };
}

export function parseStopResult(value: unknown): StopResult {
  const record = requiredRecord(value, "stop result");
  rejectUnknownKeys(record, ["forced"], "stop result");
  return {
    forced: requiredBoolean(record.forced, "forced"),
  };
}
