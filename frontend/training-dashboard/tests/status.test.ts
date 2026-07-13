import { DashboardStatus } from "../status.ts";

Deno.test("summary success preserves websocket reconnecting state", () => {
  const status = new DashboardStatus();
  status.setStreamConnection("reconnecting");
  status.setRefreshPending();
  status.clearError("summary");
  status.setRefreshIdle();

  const snapshot = status.snapshot();
  if (snapshot.label !== "RECONNECTING") {
    throw new Error(
      "Summary success overwrote WebSocket reconnecting state",
    );
  }
});

Deno.test("clearing summary error preserves metrics error", () => {
  const status = new DashboardStatus();
  status.reportError("metrics", "metrics unavailable");
  status.reportError("summary", "summary unavailable");
  status.clearError("summary");

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
