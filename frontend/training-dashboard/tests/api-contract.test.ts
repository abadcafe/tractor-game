import {
  checkpointRequestPath,
  initializeTraining,
  logPageRequestPath,
} from "../api.ts";
import type { InitRequest } from "../fields.ts";

const INIT_REQUEST: InitRequest = {
  run_dir: "/tmp/run",
  replace_existing: null,
  d_model: 8,
  layers: 1,
  heads: 1,
  seed: 0,
  learning_rate: 0.0003,
  ppo_clip: 0.2,
  value_clip: 0.2,
  entropy_coef: 0.01,
  value_coef: 0.5,
  max_grad_norm: 0.5,
  ppo_epochs: 4,
  minibatch_size: 64,
  adam_beta1: 0.9,
  adam_beta2: 0.999,
  weight_decay: 0,
};

Deno.test("REST artifact reads expose checkpoints and cursor logs", () => {
  const runDir = "/tmp/run with spaces";
  const checkpoints = new URL(
    checkpointRequestPath(runDir),
    "https://example.test",
  );
  const logs = new URL(
    logPageRequestPath(runDir, 41, 200),
    "https://example.test",
  );
  if (
    checkpoints.pathname !== "/api/training/checkpoints" ||
    checkpoints.searchParams.get("run_dir") !== runDir ||
    logs.pathname !== "/api/training/logs" ||
    logs.searchParams.get("run_dir") !== runDir ||
    logs.searchParams.get("before_sequence") !== "41" ||
    logs.searchParams.get("limit") !== "200"
  ) throw new Error(`${checkpoints.toString()} ${logs.toString()}`);
});

Deno.test("initialize reports replacement only after server precondition", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = () =>
    Promise.resolve(
      Response.json(
        { detail: "type yes to replace existing training artifacts" },
        { status: 412 },
      ),
    );
  try {
    const result = await initializeTraining(INIT_REQUEST);
    if (
      result === null ||
      result.error !==
        "type yes to replace existing training artifacts"
    ) {
      throw new Error("Expected a typed replacement requirement");
    }
  } finally {
    globalThis.fetch = originalFetch;
  }
});
