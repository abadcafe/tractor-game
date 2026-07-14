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

const REPLACEMENT_REQUIRED_STATUS = 412;

export interface ReplacementRequired {
  readonly error: string;
}

export async function fetchConfig(): Promise<TrainingConfig> {
  return parseConfig(await requestJson("/api/training/config", "GET"));
}

export async function initializeTraining(
  request: InitRequest,
): Promise<ReplacementRequired | null> {
  const response = await requestJsonResponse(
    "/api/training/init",
    "POST",
    request,
  );
  if (response.status === REPLACEMENT_REQUIRED_STATUS) {
    return { error: responseError(response) };
  }
  responseValue(response);
  return null;
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
  return responseValue(await requestJsonResponse(path, method, body));
}

interface JsonResponse {
  readonly ok: boolean;
  readonly status: number;
  readonly value: unknown;
}

async function requestJsonResponse(
  path: string,
  method: "GET" | "POST",
  body?: unknown,
): Promise<JsonResponse> {
  const response = await fetch(path, {
    method,
    headers: body === undefined
      ? undefined
      : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const value: unknown = await response.json();
  return { ok: response.ok, status: response.status, value };
}

function responseValue(response: JsonResponse): unknown {
  if (!response.ok) throw new Error(responseError(response));
  return response.value;
}

function responseError(response: JsonResponse): string {
  const record = recordValue(response.value);
  const detail = record?.detail;
  return typeof detail === "string"
    ? detail
    : `HTTP ${response.status}`;
}

function query(values: Readonly<Record<string, string>>): string {
  return new URLSearchParams(values).toString();
}
