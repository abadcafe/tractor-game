import { trainingStreamUrl } from "../stream.ts";
import { parseLogMessage } from "../types.ts";

Deno.test("training stream URL carries structured log options", () => {
  const url = trainingStreamUrl(
    {
      runDir: "/tmp/run with spaces",
      window: 5000,
      eventTypes: ["update.completed", "session.failed"],
      sessionId: "session-1",
    },
    { protocol: "https:", host: "training.example:8443" },
  );
  const parsed = new URL(url);
  if (parsed.protocol !== "wss:") throw new Error(url);
  if (parsed.pathname !== "/ws/training/logs") throw new Error(url);
  if (parsed.searchParams.get("window") !== "5000") {
    throw new Error(url);
  }
  if (parsed.searchParams.getAll("event").length !== 2) {
    throw new Error(url);
  }
  if (parsed.searchParams.get("session_id") !== "session-1") {
    throw new Error(url);
  }
});

Deno.test("structured log parser accepts reset and event messages", () => {
  const reset = parseLogMessage({ type: "reset", window: 5000 });
  if (reset.type !== "reset" || reset.window !== 5000) {
    throw new Error("reset");
  }
  const event = parseLogMessage({
    type: "event",
    sequence: 7,
    event: {
      schema_version: 1,
      event: "update.completed",
      level: "INFO",
      recorded_at_ms: 1,
      session_id: "session-1",
      process: {},
      context: {},
      fields: {},
    },
  });
  if (event.type !== "event" || event.sequence !== 7) {
    throw new Error("event");
  }
});
