import { ProcessController, processStreamUrl } from "../process.ts";
import {
  parseProcessEnvelope,
  type ProcessEnvelope,
} from "../types.ts";

Deno.test("process snapshots reject older server revisions", () => {
  const applied: ProcessEnvelope[] = [];
  const controller = new ProcessController((value) =>
    applied.push(value)
  );
  controller.apply({ revision: 12, process: null });
  controller.apply({ revision: 11, process: processSnapshot(7) });
  if (applied.length !== 1 || applied[0]?.revision !== 12) {
    throw new Error("An older process snapshot was applied");
  }
});

Deno.test("process stream is scoped only by canonical run directory", () => {
  const value = processStreamUrl(
    "/tmp/run with spaces",
    { protocol: "https:", host: "training.example:8443" },
  );
  const parsed = new URL(value);
  if (
    parsed.protocol !== "wss:" ||
    parsed.pathname !== "/ws/training/process" ||
    parsed.searchParams.get("run_dir") !== "/tmp/run with spaces" ||
    [...parsed.searchParams].length !== 1
  ) throw new Error(value);
});

Deno.test("legacy process readiness snapshots are rejected", () => {
  let rejected = false;
  try {
    parseProcessEnvelope({
      revision: 1,
      process: { ...processSnapshot(7), ready: true },
    });
  } catch (error: unknown) {
    if (!(error instanceof Error)) throw error;
    if (error.message !== "Legacy process readiness is not supported") {
      throw error;
    }
    rejected = true;
  }
  if (!rejected) throw new Error("Legacy readiness was accepted");
});

function processSnapshot(pid: number) {
  return {
    pid,
    start_ticks: 44,
    started_at_ms: 1_700_000_000_000,
    kernel_state: "S",
    executable: "/usr/bin/python3",
    working_directory: "/workspace",
    run_dir: "/tmp/run",
    argv: ["python", "-m", "server.training_cli", "resume"],
    process_group_id: pid,
    unix_session_id: pid,
    command: "resume" as const,
  };
}
