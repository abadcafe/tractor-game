import { recordValue } from "../browser/json.ts";
import type { InitRequest, ResumeRequest } from "./fields.ts";
import {
  parseConfig,
  parseMetrics,
  parseProcess,
  parseSummary,
  type TrainingConfig,
  type TrainingMetrics,
  type TrainingProcess,
  type TrainingSummary,
} from "./types.ts";

export async function fetchConfig(): Promise<TrainingConfig> {
  return parseConfig(await requestJson("/api/training/config", "GET"));
}

export async function initializeTraining(
  request: InitRequest,
): Promise<void> {
  await requestJson("/api/training/init", "POST", request);
}

export async function resumeTraining(
  request: ResumeRequest,
): Promise<TrainingProcess> {
  const value = await requestJson(
    "/api/training/resume",
    "POST",
    request,
  );
  const record = recordValue(value);
  if (record === null) throw new Error("Invalid resume response");
  return parseProcess({ process: record }) ??
    fail("Resume returned no process");
}

export async function stopTraining(runDir: string): Promise<boolean> {
  const record = recordValue(
    await requestJson("/api/training/stop", "POST", {
      run_dir: runDir,
    }),
  );
  if (record === null || typeof record.forced !== "boolean") {
    throw new Error("Invalid stop response");
  }
  return record.forced;
}

export async function fetchSummary(
  runDir: string,
): Promise<TrainingSummary> {
  return parseSummary(
    await requestJson(
      `/api/training/summary?${query({ run_dir: runDir })}`,
      "GET",
    ),
  );
}

export async function fetchMetrics(
  runDir: string,
  updateLimit: number,
  seriesPoints: number,
  sessionId: string | null = null,
): Promise<TrainingMetrics> {
  return parseMetrics(
    await requestJson(
      `/api/training/metrics?${
        query({
          run_dir: runDir,
          update_limit: String(updateLimit),
          series_points: String(seriesPoints),
          ...(sessionId === null ? {} : { session_id: sessionId }),
        })
      }`,
      "GET",
    ),
  );
}

async function requestJson(
  path: string,
  method: "GET" | "POST",
  body?: unknown,
): Promise<unknown> {
  const response = await fetch(path, {
    method,
    headers: body === undefined
      ? undefined
      : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const value: unknown = await response.json();
  if (!response.ok) {
    const detail = recordValue(value)?.detail;
    throw new Error(
      typeof detail === "string" ? detail : `HTTP ${response.status}`,
    );
  }
  return value;
}

function fail(message: string): never {
  throw new Error(message);
}

function query(values: Readonly<Record<string, string>>): string {
  return new URLSearchParams(values).toString();
}
