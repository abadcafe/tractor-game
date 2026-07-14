import { recordValue } from "../browser/json.ts";
import type { InitRequest, ResumeRequest } from "./fields.ts";
import {
  type CheckpointCatalog,
  parseCheckpointCatalog,
  parseConfig,
  parseLogPage,
  parseMetrics,
  parseProcessEnvelope,
  parseStopEnvelope,
  type ProcessEnvelope,
  type StopEnvelope,
  type TrainingConfig,
  type TrainingLogPage,
  type TrainingMetrics,
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
): Promise<ProcessEnvelope> {
  return parseProcessEnvelope(
    await requestJson("/api/training/resume", "POST", request),
  );
}

export async function stopTraining(
  runDir: string,
): Promise<StopEnvelope> {
  return parseStopEnvelope(
    await requestJson("/api/training/stop", "POST", {
      run_dir: runDir,
    }),
  );
}

export async function fetchProcess(
  runDir: string,
): Promise<ProcessEnvelope> {
  return parseProcessEnvelope(
    await requestJson(processRequestPath(runDir), "GET"),
  );
}

export async function fetchMetrics(
  runDir: string,
  updateLimit: number,
  seriesPoints: number,
): Promise<TrainingMetrics> {
  return parseMetrics(
    await requestJson(
      metricsRequestPath(runDir, updateLimit, seriesPoints),
      "GET",
    ),
  );
}

export async function fetchCheckpoints(
  runDir: string,
): Promise<CheckpointCatalog> {
  return parseCheckpointCatalog(
    await requestJson(checkpointRequestPath(runDir), "GET"),
  );
}

export async function fetchLogPage(
  runDir: string,
  beforeSequence: number | null,
  limit: number,
): Promise<TrainingLogPage> {
  return parseLogPage(
    await requestJson(
      logPageRequestPath(runDir, beforeSequence, limit),
      "GET",
    ),
  );
}

export function processRequestPath(runDir: string): string {
  return `/api/training/process?${query({ run_dir: runDir })}`;
}

export function metricsRequestPath(
  runDir: string,
  updateLimit: number,
  seriesPoints: number,
): string {
  return `/api/training/metrics?${
    query({
      run_dir: runDir,
      update_limit: String(updateLimit),
      series_points: String(seriesPoints),
    })
  }`;
}

export function checkpointRequestPath(runDir: string): string {
  return `/api/training/checkpoints?${query({ run_dir: runDir })}`;
}

export function logPageRequestPath(
  runDir: string,
  beforeSequence: number | null,
  limit: number,
): string {
  return `/api/training/logs?${
    query({
      run_dir: runDir,
      ...(beforeSequence === null
        ? {}
        : { before_sequence: String(beforeSequence) }),
      limit: String(limit),
    })
  }`;
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
    const record = recordValue(value);
    const detail = record?.detail;
    throw new Error(
      typeof detail === "string" ? detail : `HTTP ${response.status}`,
    );
  }
  return value;
}

function query(values: Readonly<Record<string, string>>): string {
  return new URLSearchParams(values).toString();
}
