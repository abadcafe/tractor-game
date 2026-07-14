import { DashboardStatus } from "../status.ts";

Deno.test("process success preserves websocket reconnecting state", () => {
  const status = new DashboardStatus();
  status.setStreamConnection("reconnecting");
  status.setRefreshPending();
  status.clearError("process");
  status.setRefreshIdle();

  const snapshot = status.snapshot();
  if (snapshot.label !== "RECONNECTING") {
    throw new Error(
      "Process success overwrote WebSocket reconnecting state",
    );
  }
});

Deno.test("clearing process error preserves metrics error", () => {
  const status = new DashboardStatus();
  status.reportError("metrics", "metrics unavailable");
  status.reportError("process", "process unavailable");
  status.clearError("process");

  const snapshot = status.snapshot();
  if (snapshot.message !== "metrics unavailable") {
    throw new Error(
      "Summary success cleared an unrelated metrics error",
    );
  }
  if (snapshot.label !== "ERROR") {
    throw new Error("Active metrics error must remain visible");
  }
});
