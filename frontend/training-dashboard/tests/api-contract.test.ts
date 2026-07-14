import {
  checkpointRequestPath,
  logPageRequestPath,
  metricsRequestPath,
  processRequestPath,
} from "../api.ts";

Deno.test("read APIs have disjoint strict paths", () => {
  const runDir = "/tmp/run with spaces";
  const paths = [
    processRequestPath(runDir),
    metricsRequestPath(runDir, 200, 500),
    checkpointRequestPath(runDir),
    logPageRequestPath(runDir, 41, 200),
  ];
  if (paths.some((path) => path.includes("summary"))) {
    throw new Error("Summary must not exist in the frontend contract");
  }
  if (paths[1]?.includes("session")) {
    throw new Error("Metrics must not carry a session selector");
  }
  const logs = new URL(paths[3] ?? "", "https://example.test");
  if (
    logs.searchParams.get("before_sequence") !== "41" ||
    logs.searchParams.get("limit") !== "200"
  ) throw new Error(logs.toString());
});
