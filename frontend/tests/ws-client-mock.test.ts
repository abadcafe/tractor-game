import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { WsClient } from "../net/ws-client.ts";
import type { ServerMessage, ClientAction } from "../core/types.ts";

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
      current_player: 0,
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
    },
  };
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
  await new Promise((r) => setTimeout(r, 100));
  assertEquals(received !== null, true);
  assertEquals(received!.type, "state");

  await server.shutdown();
  client.disconnect();
});

Deno.test("test_send_action", async () => {
  let receivedAction: string | null = null;
  const server = Deno.serve({ port: 0 }, (req) => {
    const upgrade = req.headers.get("upgrade") || "";
    if (upgrade.toLowerCase() === "websocket") {
      const { socket, response } = Deno.upgradeWebSocket(req);
      socket.addEventListener("open", () => {
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
  await new Promise((r) => setTimeout(r, 50));

  const action: ClientAction = { type: "bid", cards: ["D1-hearts-2"] };
  client.send(action);

  await new Promise((r) => setTimeout(r, 100));
  assertEquals(receivedAction !== null, true);
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
  await new Promise((r) => setTimeout(r, 100));

  assertEquals(received!.type, "state");
  if (received!.type === "state") {
    assertEquals(received!.state.phase, "DEAL_BID");
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
  await new Promise((r) => setTimeout(r, 200));

  assertEquals(disconnected, true);

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
  await new Promise((r) => setTimeout(r, 100));

  // WsClient should have connected to /game/my-game-42
  assertEquals(requestedPath, "/game/my-game-42");

  await server.shutdown();
  client.disconnect();
});
