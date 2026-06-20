import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { WsClient } from "../net/ws-client.ts";
import type { ClientAction, ServerMessage } from "../core/protocol.ts";

function makeStateMessage(): ServerMessage {
  return {
    type: "state",
    seq: 1,
    state: {
      phase: "DEAL_BID",
      player_hand: [],
      bottom_cards: [],
      trump_rank: "2",
      trump_suit: null,
      declarer_team: null,
      declarer_player: null,
      defender_points: 0,
      action_hints: [],
      trick: null,
      last_completed_trick: null,
      defender_point_cards: [],
      failed_throw: null,
      bid_events: [],
      bid_winner: null,
      awaiting_action: null,
      stirring_state: null,
      scoring: null,
      winning_team: null,
      team0_level: "2",
      team1_level: "2",
      player_hand_counts: [13, 13, 13, 13],
      next_round_confirmed: [],
    },
  };
}

function sendStateOnRequest(
  socket: WebSocket,
  msg: ServerMessage = makeStateMessage(),
): void {
  socket.addEventListener("message", (event) => {
    if (event.data === JSON.stringify({ seq: 0 })) {
      socket.send(JSON.stringify(msg));
    }
  });
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
        sendStateOnRequest(socket);
      });
      return response;
    }
    return new Response("Not Found", { status: 404 });
  });

  const addr = server.addr;
  const port = typeof addr === "object" && "port" in addr
    ? addr.port
    : 0;
  const client = new WsClient();
  let received: ServerMessage | null = null;
  client.onMessage((msg) => {
    received = msg;
  });

  // connect takes gameId + wsHost, constructs URL internally
  await client.connect("test-id", `ws://localhost:${port}`);

  // Wait for message
  await waitFor(() => received !== null);
  assertEquals(received!.type, "state");

  await server.shutdown();
  client.disconnect();
});

Deno.test("test_send_returns_false_when_socket_not_open", () => {
  const client = new WsClient();
  const action: ClientAction = {
    type: "play",
    seq: 1,
    cards: ["D1-hearts-5"],
  };

  assertEquals(client.send(action), false);
});

Deno.test("test_send_action", async () => {
  const receivedActions: string[] = [];
  let serverReady = false;
  const server = Deno.serve({ port: 0 }, (req) => {
    const upgrade = req.headers.get("upgrade") || "";
    if (upgrade.toLowerCase() === "websocket") {
      const { socket, response } = Deno.upgradeWebSocket(req);
      socket.addEventListener("open", () => {
        serverReady = true;
      });
      socket.addEventListener("message", (e) => {
        if (typeof e.data === "string") {
          receivedActions.push(e.data);
        }
        if (e.data === JSON.stringify({ seq: 0 })) {
          socket.send(JSON.stringify(makeStateMessage()));
        }
      });
      return response;
    }
    return new Response("Not Found", { status: 404 });
  });

  const addr = server.addr;
  const port = typeof addr === "object" && "port" in addr
    ? addr.port
    : 0;
  const client = new WsClient();
  client.onMessage(() => {});

  await client.connect("test-id", `ws://localhost:${port}`);
  await waitFor(() => serverReady);

  const action: ClientAction = {
    type: "bid",
    seq: 1,
    cards: ["D1-hearts-2"],
  };
  assertEquals(client.send(action), true);

  await waitFor(() => receivedActions.length >= 2);
  const parsed = JSON.parse(
    receivedActions[receivedActions.length - 1],
  );
  assertEquals(parsed.type, "bid");
  assertEquals(parsed.cards, ["D1-hearts-2"]);

  await server.shutdown();
  client.disconnect();
});

Deno.test("stale socket close does not clear the current connection", async () => {
  const sockets: WebSocket[] = [];
  const receivedByPath = new Map<string, string[]>();

  const server = Deno.serve({ port: 0 }, (req) => {
    const upgrade = req.headers.get("upgrade") || "";
    if (upgrade.toLowerCase() === "websocket") {
      const path = new URL(req.url).pathname;
      const { socket, response } = Deno.upgradeWebSocket(req);
      socket.addEventListener("open", () => {
        sockets.push(socket);
        sendStateOnRequest(socket);
      });
      socket.addEventListener("message", (event) => {
        if (typeof event.data !== "string") return;
        const existing = receivedByPath.get(path) ?? [];
        existing.push(event.data);
        receivedByPath.set(path, existing);
        if (event.data === JSON.stringify({ seq: 0 })) {
          socket.send(JSON.stringify(makeStateMessage()));
        }
      });
      return response;
    }
    return new Response("Not Found", { status: 404 });
  });

  const addr = server.addr;
  const port = typeof addr === "object" && "port" in addr
    ? addr.port
    : 0;

  const client = new WsClient();
  client.onMessage(() => {});

  await client.connect("old-game", `ws://localhost:${port}`);
  await waitFor(() => sockets.length === 1);
  await client.connect("new-game", `ws://localhost:${port}`);
  await waitFor(() => sockets.length === 2);

  sockets[0].close();
  await new Promise((resolve) => setTimeout(resolve, 50));

  const action: ClientAction = {
    type: "next_round",
    seq: 1,
  };
  assertEquals(client.send(action), true);

  await waitFor(() =>
    (receivedByPath.get("/game/new-game") ?? []).some((raw) =>
      raw === JSON.stringify(action)
    )
  );

  client.disconnect();
  await server.shutdown();
});

Deno.test("failed initial connect does not schedule reconnects", async () => {
  let requestCount = 0;

  const server = Deno.serve({ port: 0 }, () => {
    requestCount++;
    return new Response("Not Found", { status: 404 });
  });

  const addr = server.addr;
  const port = typeof addr === "object" && "port" in addr
    ? addr.port
    : 0;

  const client = new WsClient();
  client.onMessage(() => {});

  let rejected = false;
  try {
    await client.connect("missing-game", `ws://localhost:${port}`);
  } catch {
    rejected = true;
  }

  assertEquals(rejected, true);
  await new Promise((resolve) => setTimeout(resolve, 1200));
  assertEquals(requestCount, 1);

  client.disconnect();
  await server.shutdown();
});

Deno.test("test_onMessage_receives_state", async () => {
  const msg = makeStateMessage();
  const server = Deno.serve({ port: 0 }, (req) => {
    const upgrade = req.headers.get("upgrade") || "";
    if (upgrade.toLowerCase() === "websocket") {
      const { socket, response } = Deno.upgradeWebSocket(req);
      socket.addEventListener("open", () => {
        sendStateOnRequest(socket, msg);
      });
      return response;
    }
    return new Response("Not Found", { status: 404 });
  });

  const addr = server.addr;
  const port = typeof addr === "object" && "port" in addr
    ? addr.port
    : 0;

  const client = new WsClient();
  let received: ServerMessage | null = null;
  client.onMessage((m) => {
    received = m;
  });

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
        sendStateOnRequest(socket);
        socket.addEventListener("message", () => {
          setTimeout(() => socket.close(), 50);
        });
      });
      return response;
    }
    return new Response("Not Found", { status: 404 });
  });

  const addr = server.addr;
  const port = typeof addr === "object" && "port" in addr
    ? addr.port
    : 0;

  const client = new WsClient();
  let disconnected = false;
  client.onMessage(() => {});
  client.onDisconnect(() => {
    disconnected = true;
  });

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
        sendStateOnRequest(socket);
      });
      return response;
    }
    return new Response("Not Found", { status: 404 });
  });

  const addr = server.addr;
  const port = typeof addr === "object" && "port" in addr
    ? addr.port
    : 0;
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
        sendStateOnRequest(socket);
        // Close after first connection to trigger reconnect
        if (connectionCount === 1) {
          socket.addEventListener("message", () => {
            setTimeout(() => socket.close(), 50);
          });
        }
      });
      return response;
    }
    return new Response("Not Found", { status: 404 });
  });

  const addr = server.addr;
  const port = typeof addr === "object" && "port" in addr
    ? addr.port
    : 0;

  const client = new WsClient();
  client.onMessage((msg) => {
    latestReceived = msg;
  });
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
  const port = typeof addr === "object" && "port" in addr
    ? addr.port
    : 0;

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
