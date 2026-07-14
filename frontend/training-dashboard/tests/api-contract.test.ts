import {
  checkpointRequestPath,
  initializeTraining,
  logPageRequestPath,
  metricsRequestPath,
} from "../api.ts";
import type { InitRequest } from "../fields.ts";

const INIT_REQUEST: InitRequest = {
  run_dir: "/tmp/run",
  replace_existing: null,
  d_model: 2,
  layers: 1,
  heads: 1,
  max_tokens: 512,
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

Deno.test("artifact read APIs have disjoint strict paths", () => {
  const runDir = "/tmp/run with spaces";
  const paths = [
    metricsRequestPath(runDir, 200, 500),
    checkpointRequestPath(runDir),
    logPageRequestPath(runDir, 41, 200),
  ];
  if (paths.some((path) => path.includes("summary"))) {
    throw new Error("Summary must not exist in the frontend contract");
  }
  if (paths[0]?.includes("session")) {
    throw new Error("Metrics must not carry a session selector");
  }
  const logs = new URL(paths[2] ?? "", "https://example.test");
  if (
    logs.searchParams.get("before_sequence") !== "41" ||
    logs.searchParams.get("limit") !== "200"
  ) throw new Error(logs.toString());
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
