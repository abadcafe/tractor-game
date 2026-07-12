import { recordValue } from "../browser/json.ts";
import type { InitRequest, ResumeRequest } from "./fields.ts";
import {
  parseConfig,
  parseProcess,
  type TrainingConfig,
  type TrainingProcess,
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
