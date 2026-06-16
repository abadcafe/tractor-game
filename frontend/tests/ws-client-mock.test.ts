import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { WsClient } from "../net/ws-client.ts";
import type { ServerMessage, ClientAction } from "../core/protocol.ts";

function makeStateMessage(): ServerMessage {
  return {
    type: "state",
    awaiting: null,
    state: {
      phase: "DEAL_BID",
      player_hand: [],
      bottom_cards: [],
      trump_rank: "2",
      trump_suit: null,
      declarer_team: null,
      declarer_player: null,
      defender_points: 0,
      legal_actions: [],
      trick: null,
      trick_history: [],
      bid_events: [],
      bid_winner: null,
      awaiting_action: null,
      stirring_state: null,
      exchange_state: null,
      scoring: null,
      winning_team: null,
      team0_level: "2",
      team1_level: "2",
      player_hand_counts: [13, 13, 13, 13],
      next_round_confirmed: [],
    },
  };
}

/** Helper: wait for a condition with polling, up to maxMs. */
async function waitFor(
  check: () => boolean,
  maxMs = 2000,
  pollMs = 10,
): Promise<void> {
  const start = Date.now();
  while (!check()) {
    if (Date.now() - start > maxMs) {
      throw new Error("waitFor timed out");
    }
    await new Promise((r) => setTimeout(r, pollMs));
  }
}

Deno.test("test_connect_success", async () => {
  const server = Deno.serve({ port: 0 }, (req) => {
    const upgrade = req.headers.get("upgrade") || "";
    if (upgrade.toLowerCase() === "websocket") {
      const { socket, response } = Deno.upgradeWebSocket(req);
      socket.addEventListener("open", () => {
        socket.send(JSON.stringify(makeStateMessage()));
      });
      return response;
    }
    return new Response("Not Found", { status: 404 });
  });

  const addr = server.addr;
  const port = typeof addr === "object" && "port" in addr ? addr.port : 0;
  const client = new WsClient();
  let received: ServerMessage | null = null;
  client.onMessage((msg) => { received = msg; });

  // connect takes gameId + wsHost, constructs URL internally
  await client.connect("test-id", `ws://localhost:${port}`);

  // Wait for message
  await waitFor(() => received !== null);
  assertEquals(received!.type, "state");

  await server.shutdown();
  client.disconnect();
});

Deno.test("test_send_action", async () => {
  let receivedAction: string | null = null;
  let serverReady = false;
  const server = Deno.serve({ port: 0 }, (req) => {
    const upgrade = req.headers.get("upgrade") || "";
    if (upgrade.toLowerCase() === "websocket") {
      const { socket, response } = Deno.upgradeWebSocket(req);
      socket.addEventListener("open", () => {
        serverReady = true;
        socket.send(JSON.stringify(makeStateMessage()));
      });
      socket.addEventListener("message", (e) => {
        receivedAction = e.data as string;
      });
      return response;
    }
    return new Response("Not Found", { status: 404 });
  });

  const addr = server.addr;
  const port = typeof addr === "object" && "port" in addr ? addr.port : 0;
  const client = new WsClient();
  client.onMessage(() => {});

  await client.connect("test-id", `ws://localhost:${port}`);
  await waitFor(() => serverReady);

  const action: ClientAction = { type: "bid", cards: ["D1-hearts-2"] };
  client.send(action);

  await waitFor(() => receivedAction !== null);
  const parsed = JSON.parse(receivedAction!);
  assertEquals(parsed.type, "bid");
  assertEquals(parsed.cards, ["D1-hearts-2"]);

  await server.shutdown();
  client.disconnect();
});

Deno.test("test_onMessage_receives_state", async () => {
  const msg = makeStateMessage();
  const server = Deno.serve({ port: 0 }, (req) => {
    const upgrade = req.headers.get("upgrade") || "";
    if (upgrade.toLowerCase() === "websocket") {
      const { socket, response } = Deno.upgradeWebSocket(req);
      socket.addEventListener("open", () => {
        socket.send(JSON.stringify(msg));
      });
      return response;
    }
    return new Response("Not Found", { status: 404 });
  });

  const addr = server.addr;
  const port = typeof addr === "object" && "port" in addr ? addr.port : 0;

  const client = new WsClient();
  let received: ServerMessage | null = null;
  client.onMessage((m) => { received = m; });

  await client.connect("test-id", `ws://localhost:${port}`);
  await waitFor(() => received !== null);

  const receivedMsg = received!;
  assertEquals(receivedMsg.type, "state");
  if (receivedMsg.type === "state") {
    assertEquals(receivedMsg.state.phase, "DEAL_BID");
  }

  await server.shutdown();
  client.disconnect();
});

Deno.test("test_onDisconnect_called", async () => {
  const server = Deno.serve({ port: 0 }, (req) => {
    const upgrade = req.headers.get("upgrade") || "";
    if (upgrade.toLowerCase() === "websocket") {
      const { socket, response } = Deno.upgradeWebSocket(req);
      socket.addEventListener("open", () => {
        socket.send(JSON.stringify(makeStateMessage()));
        setTimeout(() => socket.close(), 50);
      });
      return response;
    }
    return new Response("Not Found", { status: 404 });
  });

  const addr = server.addr;
  const port = typeof addr === "object" && "port" in addr ? addr.port : 0;

  const client = new WsClient();
  let disconnected = false;
  client.onMessage(() => {});
  client.onDisconnect(() => { disconnected = true; });

  await client.connect("test-id", `ws://localhost:${port}`);
  await waitFor(() => disconnected);

  assertEquals(disconnected, true);

  client.disconnect();
  await server.shutdown();
});

Deno.test("test_connect_constructs_ws_url_from_game_id", async () => {
  let requestedPath = "";
  const server = Deno.serve({ port: 0 }, (req) => {
    const upgrade = req.headers.get("upgrade") || "";
    if (upgrade.toLowerCase() === "websocket") {
      requestedPath = new URL(req.url).pathname;
      const { socket, response } = Deno.upgradeWebSocket(req);
      socket.addEventListener("open", () => {
        socket.send(JSON.stringify(makeStateMessage()));
      });
      return response;
    }
    return new Response("Not Found", { status: 404 });
  });

  const addr = server.addr;
  const port = typeof addr === "object" && "port" in addr ? addr.port : 0;
  const client = new WsClient();
  client.onMessage(() => {});

  await client.connect("my-game-42", `ws://localhost:${port}`);
  await waitFor(() => requestedPath !== "");

  // WsClient should have connected to /game/my-game-42
  assertEquals(requestedPath, "/game/my-game-42");

  await server.shutdown();
  client.disconnect();
});

Deno.test("test_reconnects_after_server_disconnect", async () => {
  let connectionCount = 0;
  let latestReceived: ServerMessage | null = null;

  const server = Deno.serve({ port: 0 }, (req) => {
    const upgrade = req.headers.get("upgrade") || "";
    if (upgrade.toLowerCase() === "websocket") {
      const { socket, response } = Deno.upgradeWebSocket(req);
      socket.addEventListener("open", () => {
        connectionCount++;
        socket.send(JSON.stringify(makeStateMessage()));
        // Close after first connection to trigger reconnect
        if (connectionCount === 1) {
          setTimeout(() => socket.close(), 50);
        }
      });
      return response;
    }
    return new Response("Not Found", { status: 404 });
  });

  const addr = server.addr;
  const port = typeof addr === "object" && "port" in addr ? addr.port : 0;

  const client = new WsClient();
  client.onMessage((msg) => { latestReceived = msg; });
  client.onDisconnect(() => {});

  await client.connect("test-id", `ws://localhost:${port}`);

  // Wait for reconnection: connectionCount should become 2
  // With 1s backoff (first retry), this should happen within ~1.5s
  await waitFor(() => connectionCount >= 2, 3000);

  assertEquals(connectionCount >= 2, true);
  assertEquals(latestReceived !== null, true);
  assertEquals(latestReceived!.type, "state");

  client.disconnect();
  await server.shutdown();
});

Deno.test("test_reconnect_respects_max_retries", async () => {
  let connectionCount = 0;

  const server = Deno.serve({ port: 0 }, (req) => {
    const upgrade = req.headers.get("upgrade") || "";
    if (upgrade.toLowerCase() === "websocket") {
      const { socket, response } = Deno.upgradeWebSocket(req);
      socket.addEventListener("open", () => {
        connectionCount++;
        // Always close immediately to force reconnection attempts
        setTimeout(() => socket.close(), 10);
      });
      return response;
    }
    return new Response("Not Found", { status: 404 });
  });

  const addr = server.addr;
  const port = typeof addr === "object" && "port" in addr ? addr.port : 0;

  const client = new WsClient();
  client.onMessage(() => {});
  client.onDisconnect(() => {});

  await client.connect("test-id", `ws://localhost:${port}`);

  // Wait for all 3 retries: 1s + 2s + 4s = 7s, use 10s timeout to be safe
  await waitFor(() => connectionCount >= 4, 10000);

  // Should have 1 initial + 3 retries = 4 connections max
  assertEquals(connectionCount, 4);

  client.disconnect();
  await server.shutdown();
});

Deno.test("test_connect_rejects_without_ws_host", async () => {
  const client = new WsClient();
  client.onMessage(() => {});

  let threw = false;
  try {
    await client.connect("test-id");
  } catch {
    threw = true;
  }
  assertEquals(threw, true);
});
