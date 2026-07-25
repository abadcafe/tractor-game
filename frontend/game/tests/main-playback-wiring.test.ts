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

Deno.test("main opens entered lobby seat in a new page", async () => {
  const source = await Deno.readTextFile(
    new URL("../main.ts", import.meta.url),
  );

  assert(/onToggleSeat\(gameId, seatId\) \{/.test(source));
  assert(
    /void handleToggleSeat\(gameId, seatId\);/.test(source),
  );
  assert(/onDeleteGame\(gameId: string\) \{/.test(source));
  assert(/void handleDeleteGame\(gameId\);/.test(source));
  assert(/onEnterSeat\(gameId, seatId\) \{/.test(source));
  assert(/void handleEnterSeat\(gameId, seatId\);/.test(source));
  assert(/enterSeatHref\(gameId, seatId\) \{/.test(source));
  assert(
    /await occupySeat\(gameId, seatId, userId\);/.test(source),
  );
  assert(
    /await vacateSeat\(gameId, seatId, userId\);/.test(source),
  );
  assert(
    /return gameSeatHref\(gameId, seatId, ensureUserId\(\)\);/
      .test(
        source,
      ),
  );
  assert(
    /void handleChangeBotFillMode\(mode\);/.test(source),
  );
  assert(
    /await fillBotSeats\(selectedGame\.gameId, mode, ensureUserId\(\)\);/
      .test(source),
  );
  assert(/await deleteGame\(gameId\);/.test(source));
  assert(!/globalThis\.open\(/.test(source));
  assert(!/globalThis\.location\.assign\(/.test(source));
});
