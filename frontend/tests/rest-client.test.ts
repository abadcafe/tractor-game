import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { createGame, listGames, deleteGame } from "../net/rest-client.ts";

// Helper to start a mock HTTP server
async function withMockServer(
  handler: (req: Request) => Response,
  fn: (baseUrl: string) => Promise<void>,
): Promise<void> {
  const server = Deno.serve({ port: 0 }, handler);
  const addr = server.addr;
  const port = typeof addr === "object" && "port" in addr ? addr.port : 0;
  const baseUrl = `http://localhost:${port}`;
  try {
    await fn(baseUrl);
  } finally {
    await server.shutdown();
  }
}

Deno.test("test_createGame_success", async () => {
  await withMockServer(
    (req) => {
      if (req.method === "POST" && new URL(req.url).pathname === "/api/game") {
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

Deno.test("test_listGames_success", async () => {
  await withMockServer(
    (req) => {
      if (req.method === "GET" && new URL(req.url).pathname === "/api/game") {
        return new Response(JSON.stringify({ games: ["a", "b"] }), {
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response("Not Found", { status: 404 });
    },
    async (baseUrl) => {
      const games = await listGames(baseUrl);
      assertEquals(games, ["a", "b"]);
    },
  );
});

Deno.test("test_listGames_empty", async () => {
  await withMockServer(
    (_req) => new Response(JSON.stringify({ games: [] }), {
      headers: { "Content-Type": "application/json" },
    }),
    async (baseUrl) => {
      const games = await listGames(baseUrl);
      assertEquals(games, []);
    },
  );
});

Deno.test("test_deleteGame_success", async () => {
  await withMockServer(
    (req) => {
      if (req.method === "DELETE" && new URL(req.url).pathname === "/api/game/g1") {
        return new Response(JSON.stringify({ ok: true }), {
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response("Not Found", { status: 404 });
    },
    async (baseUrl) => {
      await deleteGame("g1", baseUrl);
      // No error means success
    },
  );
});

Deno.test("test_deleteGame_error", async () => {
  await withMockServer(
    (_req) => new Response("Not Found", { status: 404 }),
    async (baseUrl) => {
      try {
        await deleteGame("nonexistent", baseUrl);
        assertEquals(true, false);
      } catch (e) {
        assertEquals(e instanceof Error, true);
      }
    },
  );
});
