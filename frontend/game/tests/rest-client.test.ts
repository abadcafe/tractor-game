import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  createGame,
  deleteGame,
  fillBotSeats,
  listGames,
  occupySeat,
  vacateSeat,
} from "../net/rest-client.ts";

// Helper to start a mock HTTP server
async function withMockServer(
  handler: (req: Request) => Response,
  fn: (baseUrl: string) => Promise<void>,
): Promise<void> {
  const server = Deno.serve(
    { hostname: "127.0.0.1", port: 0 },
    handler,
  );
  const addr = server.addr;
  const port = typeof addr === "object" && "port" in addr
    ? addr.port
    : 0;
  const baseUrl = `http://127.0.0.1:${port}`;
  try {
    await fn(baseUrl);
  } finally {
    await server.shutdown();
  }
}

Deno.test("test_createGame_success", async () => {
  await withMockServer(
    (req) => {
      if (
        req.method === "POST" &&
        new URL(req.url).pathname === "/api/game"
      ) {
        return new Response(JSON.stringify({ game_id: "test-123" }), {
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response("Not Found", { status: 404 });
    },
    async (baseUrl) => {
      const id = await createGame(baseUrl);
      assertEquals(id, "test-123");
    },
  );
});

Deno.test("test_createGame_error", async () => {
  await withMockServer(
    (_req) => new Response("Internal Server Error", { status: 500 }),
    async (baseUrl) => {
      try {
        await createGame(baseUrl);
        // Should have thrown
        assertEquals(true, false);
      } catch (e) {
        assertEquals(e instanceof Error, true);
      }
    },
  );
});

Deno.test("test_deleteGame_sends_delete_request", async () => {
  let observedMethod = "";
  let observedPath = "";
  await withMockServer(
    (req) => {
      const url = new URL(req.url);
      observedMethod = req.method;
      observedPath = url.pathname;
      if (
        req.method === "DELETE" &&
        url.pathname === "/api/game/game%201"
      ) {
        return new Response(JSON.stringify({ ok: true }), {
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response("Not Found", { status: 404 });
    },
    async (baseUrl) => {
      const ok = await deleteGame("game 1", baseUrl);
      assertEquals(ok, true);
      assertEquals(observedMethod, "DELETE");
      assertEquals(observedPath, "/api/game/game%201");
    },
  );
});

Deno.test("test_listGames_success", async () => {
  await withMockServer(
    (req) => {
      if (
        req.method === "GET" &&
        new URL(req.url).pathname === "/api/game"
      ) {
        return new Response(
          JSON.stringify({
            games: [{
              game_id: "game-123",
              user_count: 2,
              capacity: 4,
              user_seats: ["b", "d"],
              seats: [
                {
                  seat: "a",
                  occupied: false,
                  connected: false,
                  kind: "empty",
                  mine: false,
                  ready: false,
                },
                {
                  seat: "b",
                  occupied: true,
                  connected: true,
                  kind: "user",
                  mine: true,
                  ready: false,
                },
                {
                  seat: "c",
                  occupied: false,
                  connected: false,
                  kind: "empty",
                  mine: false,
                  ready: false,
                },
                {
                  seat: "d",
                  occupied: true,
                  connected: false,
                  kind: "auto",
                  mine: false,
                  ready: true,
                },
              ],
            }],
          }),
          { headers: { "Content-Type": "application/json" } },
        );
      }
      return new Response("Not Found", { status: 404 });
    },
    async (baseUrl) => {
      const games = await listGames(baseUrl);
      assertEquals(games, [{
        gameId: "game-123",
        userCount: 2,
        capacity: 4,
        userSeats: ["b", "d"],
        seats: [
          {
            seat: "a",
            occupied: false,
            connected: false,
            kind: "empty",
            mine: false,
            ready: false,
          },
          {
            seat: "b",
            occupied: true,
            connected: true,
            kind: "user",
            mine: true,
            ready: false,
          },
          {
            seat: "c",
            occupied: false,
            connected: false,
            kind: "empty",
            mine: false,
            ready: false,
          },
          {
            seat: "d",
            occupied: true,
            connected: false,
            kind: "auto",
            mine: false,
            ready: true,
          },
        ],
      }]);
    },
  );
});

Deno.test("test_listGames_sends_user_id_query", async () => {
  let observedSearch = "";
  await withMockServer(
    (req) => {
      const url = new URL(req.url);
      observedSearch = url.search;
      if (req.method === "GET" && url.pathname === "/api/game") {
        return new Response(JSON.stringify({ games: [] }), {
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response("Not Found", { status: 404 });
    },
    async (baseUrl) => {
      await listGames(baseUrl, "user 1");
      assertEquals(observedSearch, "?user_id=user%201");
    },
  );
});

Deno.test("test_occupySeat_sends_seat_request", async () => {
  let observedMethod = "";
  let observedPath = "";
  let observedSearch = "";
  await withMockServer(
    (req) => {
      const url = new URL(req.url);
      observedMethod = req.method;
      observedPath = url.pathname;
      observedSearch = url.search;
      if (
        req.method === "POST" &&
        url.pathname === "/api/game/game%201/seat/d"
      ) {
        return new Response(JSON.stringify({ ok: true }), {
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response("Not Found", { status: 404 });
    },
    async (baseUrl) => {
      const ok = await occupySeat("game 1", "d", "user 1", baseUrl);
      assertEquals(ok, true);
      assertEquals(observedMethod, "POST");
      assertEquals(observedPath, "/api/game/game%201/seat/d");
      assertEquals(observedSearch, "?user_id=user%201");
    },
  );
});

Deno.test("test_vacateSeat_sends_seat_request", async () => {
  let observedMethod = "";
  let observedPath = "";
  let observedSearch = "";
  await withMockServer(
    (req) => {
      const url = new URL(req.url);
      observedMethod = req.method;
      observedPath = url.pathname;
      observedSearch = url.search;
      if (
        req.method === "DELETE" &&
        url.pathname === "/api/game/game-1/seat/c"
      ) {
        return new Response(JSON.stringify({ ok: true }), {
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response("Not Found", { status: 404 });
    },
    async (baseUrl) => {
      const ok = await vacateSeat("game-1", "c", "user 2", baseUrl);
      assertEquals(ok, true);
      assertEquals(observedMethod, "DELETE");
      assertEquals(observedPath, "/api/game/game-1/seat/c");
      assertEquals(observedSearch, "?user_id=user%202");
    },
  );
});

Deno.test("test_fillBotSeats_sends_bot_fill_request", async () => {
  let observedMethod = "";
  let observedPath = "";
  let observedSearch = "";
  await withMockServer(
    (req) => {
      const url = new URL(req.url);
      observedMethod = req.method;
      observedPath = url.pathname;
      observedSearch = url.search;
      if (
        req.method === "POST" &&
        url.pathname === "/api/game/game-1/bots"
      ) {
        return new Response(JSON.stringify({ ok: true }), {
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response("Not Found", { status: 404 });
    },
    async (baseUrl) => {
      const ok = await fillBotSeats(
        "game-1",
        "ai",
        "user 1",
        baseUrl,
      );
      assertEquals(ok, true);
      assertEquals(observedMethod, "POST");
      assertEquals(observedPath, "/api/game/game-1/bots");
      assertEquals(observedSearch, "?kind=ai&user_id=user%201");
    },
  );
});
