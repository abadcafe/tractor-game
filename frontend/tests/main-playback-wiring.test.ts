import { assert } from "https://deno.land/std@0.224.0/assert/mod.ts";

Deno.test("main wires server messages through phase-aware state playback queue", async () => {
  const source = await Deno.readTextFile(
    new URL("../main.ts", import.meta.url),
  );

  assert(/const DEAL_BID_PLAYBACK_INTERVAL_MS = 125;/.test(source));
  assert(/const DEFAULT_PLAYBACK_INTERVAL_MS = 500;/.test(source));
  assert(/new StatePlaybackQueue<ServerMessage>\(/.test(source));
  assert(/minFrameMsForMessage\(msg\)/.test(source));
  assert(
    /msg\.state\.phase === "DEAL_BID"\s*\?\s*DEAL_BID_PLAYBACK_INTERVAL_MS\s*:\s*DEFAULT_PLAYBACK_INTERVAL_MS/s
      .test(source),
  );
  assert(
    /wsClient\.onMessage\(\(msg: ServerMessage\) => \{\s*playbackQueue\?\.enqueue\(msg\);\s*\}\);/s
      .test(source),
  );
  assert(
    !/wsClient\.onMessage\(\(msg: ServerMessage\) => \{\s*gameLoop\.handleMessage\(msg\);\s*\}\);/s
      .test(source),
  );
});
