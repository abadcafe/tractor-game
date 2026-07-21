import { ProcessStatus } from "../process-status.ts";

Deno.test("process success preserves event-stream reconnecting state", () => {
  const status = new ProcessStatus();
  status.setConnection("reconnecting");
  status.clearError("process");

  const snapshot = status.snapshot();
  if (snapshot.label !== "RECONNECTING") {
    throw new Error(
      "Process success overwrote event-stream reconnecting state",
    );
  }
});

Deno.test("clearing process error preserves process control error", () => {
  const status = new ProcessStatus();
  status.reportError("control", "stop failed");
  status.reportError("process", "process unavailable");
  status.clearError("process");

  const snapshot = status.snapshot();
  if (snapshot.message !== "stop failed") {
    throw new Error("Process success cleared a control error");
  }
  if (snapshot.label !== "ERROR") {
    throw new Error("Active process control error must remain visible");
  }
});
