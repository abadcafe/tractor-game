import { processStreamUrl } from "../process.ts";
import { parseProcessState } from "../types.ts";

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

Deno.test("process state parses PID diagnostics without ownership", () => {
  const parsed = parseProcessState({ process: processSnapshot(7) });
  if (parsed.process?.pid !== 7) {
    throw new Error("Process PID was not parsed");
  }
  if (parsed.process.inspection.kind !== "details") {
    throw new Error("Process details were not parsed");
  }
  if (parsed.process.inspection.argv.at(-1) !== "resume") {
    throw new Error("Process argv was not preserved");
  }
});

Deno.test("process state parses a live PID with unreadable details", () => {
  const parsed = parseProcessState({
    process: {
      pid: 9,
      inspection: { kind: "error", error: "permission denied" },
    },
  });
  if (parsed.process?.inspection.kind !== "error") {
    throw new Error("Inspection error was not parsed");
  }
});

Deno.test("revisioned process envelopes are rejected", () => {
  let rejected = false;
  try {
    parseProcessState({ revision: 1, process: processSnapshot(7) });
  } catch (error: unknown) {
    if (!(error instanceof Error)) throw error;
    rejected = true;
  }
  if (!rejected) throw new Error("Legacy revision was accepted");
});

function processSnapshot(pid: number) {
  return {
    pid,
    inspection: {
      kind: "details" as const,
      started_at_ms: 1_700_000_000_000,
      kernel_state: "S",
      executable: "/usr/bin/python3",
      working_directory: "/workspace",
      argv: ["python", "-m", "server.training_cli", "resume"],
      process_group_id: pid,
      unix_session_id: pid,
    },
  };
}
