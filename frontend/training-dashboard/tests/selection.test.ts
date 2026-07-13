import { DashboardSelection } from "../selection.ts";

Deno.test("run snapshots expire when the run directory changes", () => {
  const selection = new DashboardSelection();
  selection.setRunDirectory("/runs/first");
  const snapshot = selection.captureRun();

  selection.setRunDirectory("/runs/second");

  if (selection.ownsRun(snapshot)) {
    throw new Error(
      "Old run request must not update the new directory",
    );
  }
  if (snapshot.runDir !== "/runs/first") {
    throw new Error("Request must retain its originating directory");
  }
});

Deno.test("returning to a directory does not revive old snapshots", () => {
  const selection = new DashboardSelection();
  selection.setRunDirectory("/runs/first");
  const snapshot = selection.captureRun();

  selection.setRunDirectory("/runs/second");
  selection.setRunDirectory("/runs/first");

  if (selection.ownsRun(snapshot)) {
    throw new Error("Run ownership must include a monotonic revision");
  }
});

Deno.test("metric snapshots bind the selected historical session", () => {
  const selection = new DashboardSelection();
  selection.setRunDirectory("/runs/first");
  selection.setMetricSession("old-session");
  const snapshot = selection.captureMetrics();

  selection.setMetricSession(null);

  if (selection.ownsMetrics(snapshot)) {
    throw new Error(
      "Old session metrics must not replace latest metrics",
    );
  }
  if (snapshot.sessionId !== "old-session") {
    throw new Error("Metric request must retain its selected session");
  }
});

Deno.test("run replacement invalidates requests and clears session", () => {
  const selection = new DashboardSelection();
  selection.setRunDirectory("/runs/first");
  selection.setMetricSession("deleted-session");
  const runSnapshot = selection.captureRun();
  const metricSnapshot = selection.captureMetrics();

  selection.markRunReplaced();

  if (selection.metricSession !== null) {
    throw new Error("Replacement must select the latest session");
  }
  if (
    selection.ownsRun(runSnapshot) ||
    selection.ownsMetrics(metricSnapshot)
  ) {
    throw new Error(
      "Replacement must invalidate old database requests",
    );
  }
});
