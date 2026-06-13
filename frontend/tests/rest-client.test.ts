import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { createGame } from "../net/rest-client.ts";

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
