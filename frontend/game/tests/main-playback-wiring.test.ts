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
    /wsClient\.onMessage\(\(msg: ServerMessage\) => \{\s*updateConnectionStatus\("connected"\);\s*playbackQueue\?\.enqueue\(msg\);\s*\}\);/s
      .test(source),
  );
  assert(
    !/wsClient\.onMessage\(\(msg: ServerMessage\) => \{\s*gameLoop\.handleMessage\(msg\);\s*\}\);/s
      .test(source),
  );
});

Deno.test("main opens entered lobby player in a new page", async () => {
  const source = await Deno.readTextFile(
    new URL("../main.ts", import.meta.url),
  );

  assert(/onTogglePlayer\(gameId, playerIndex\) \{/.test(source));
  assert(
    /void handleTogglePlayer\(gameId, playerIndex\);/.test(source),
  );
  assert(/onDeleteGame\(gameId: string\) \{/.test(source));
  assert(/void handleDeleteGame\(gameId\);/.test(source));
  assert(/onEnterPlayer\(gameId, playerIndex\) \{/.test(source));
  assert(/void handleEnterPlayer\(gameId, playerIndex\);/.test(source));
  assert(/enterPlayerHref\(gameId, playerIndex\) \{/.test(source));
  assert(
    /await joinPlayer\(gameId, playerIndex, userId\);/.test(source),
  );
  assert(
    /await leavePlayer\(gameId, playerIndex, userId\);/.test(source),
  );
  assert(
    /return gamePlayerHref\(gameId, playerIndex, ensureUserId\(\)\);/
      .test(
        source,
      ),
  );
  assert(
    /void handleChangeBotFillMode\(mode\);/.test(source),
  );
  assert(
    /await fillBotPlayers\(selectedGame\.gameId, mode, ensureUserId\(\)\);/
      .test(source),
  );
  assert(/await deleteGame\(gameId\);/.test(source));
  assert(!/globalThis\.open\(/.test(source));
  assert(!/globalThis\.location\.assign\(/.test(source));
});
