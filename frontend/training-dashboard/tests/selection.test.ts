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
    throw new Error(
      "Run ownership must include a monotonic generation",
    );
  }
});

Deno.test("run replacement invalidates every domain request", () => {
  const selection = new DashboardSelection();
  selection.setRunDirectory("/runs/first");
  const runSnapshot = selection.captureRun();

  selection.markRunReplaced();

  if (selection.ownsRun(runSnapshot)) {
    throw new Error(
      "Replacement must invalidate old domain requests",
    );
  }
});
