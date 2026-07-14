import {
  INIT_FIELDS,
  initCommandPreview,
  type InitRequest,
  RESUME_FIELDS,
  resumeCommandPreview,
  type ResumeRequest,
} from "../fields.ts";

Deno.test("init and resume forms expose disjoint command boundaries", () => {
  const initFlags = new Set(INIT_FIELDS.map((field) => field.flag));
  const resumeFlags = new Set(RESUME_FIELDS.map((field) => field.flag));
  if (!initFlags.has("--d-model") || resumeFlags.has("--d-model")) {
    throw new Error("Model shape must be init-only");
  }
  if (
    !resumeFlags.has("--checkpoint") || initFlags.has("--checkpoint")
  ) {
    throw new Error("Checkpoint selection must be resume-only");
  }
  if (initFlags.has("--replace-existing")) {
    throw new Error("Replace confirmation is rendered separately");
  }
});

Deno.test("command previews include explicit subcommands", () => {
  const init = {
    run_dir: "run with spaces",
    replace_existing: null,
    d_model: 128,
    layers: 3,
    heads: 4,
    max_tokens: 768,
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
  } satisfies InitRequest;
  const resume = {
    run_dir: "run with spaces",
    checkpoint: "latest.json",
    worker_cpus: null,
    model_ranks: null,
    ppo_profile: null,
    max_samples: 0,
    learning_rate: null,
    checkpoint_every_updates: 50,
    checkpoint_retention_updates: 5,
    round_timeout_seconds: null,
    sampling_start_timeout_seconds: null,
    rollout_sample_timeout_seconds: null,
    sampling_stop_timeout_seconds: null,
    state_sync_timeout_seconds: null,
    update_timeout_seconds: null,
    model_inference_batch_size: null,
    game_envs_per_worker: null,
    samples_per_update: null,
    ppo_clip: null,
    value_clip: null,
    entropy_coef: null,
    value_coef: null,
    max_grad_norm: null,
    ppo_epochs: null,
    minibatch_size: null,
    adam_beta1: null,
    adam_beta2: null,
    weight_decay: null,
  } satisfies ResumeRequest;

  const initCommand = initCommandPreview(init);
  const resumeCommand = resumeCommandPreview(resume);
  if (
    !initCommand.startsWith(
      "python -m server.training_cli --run-dir 'run with spaces' init ",
    )
  ) {
    throw new Error(initCommand);
  }
  if (
    !resumeCommand.startsWith(
      "python -m server.training_cli --run-dir 'run with spaces' resume ",
    )
  ) {
    throw new Error(resumeCommand);
  }
  if (
    !resumeCommand.includes(" resume latest.json ") ||
    resumeCommand.includes(" --checkpoint ")
  ) {
    throw new Error(resumeCommand);
  }
  const replaceCommand = initCommandPreview({
    ...init,
    replace_existing: "yes",
  });
  if (!replaceCommand.includes("--replace-existing yes")) {
    throw new Error(replaceCommand);
  }
});

Deno.test("modal close controls are explicit non-submit buttons", async () => {
  const html = await Deno.readTextFile(
    new URL("../index.html", import.meta.url),
  );
  const closeControls = [
    ...html.matchAll(/<button(?=[^>]*data-close-dialog)[^>]*>/g),
  ];
  if (closeControls.length !== 6) {
    throw new Error(
      `Expected 4 close controls, got ${closeControls.length}`,
    );
  }
  if (
    closeControls.some(([button]) => !button.includes('type="button"'))
  ) {
    throw new Error("Every close control must be type=button");
  }
  if (
    !html.includes('name="replace_existing"') ||
    !html.includes('pattern="yes"') ||
    !html.includes('id="replace-dialog"') ||
    !html.includes('class="modal-shell replace-shell"')
  ) {
    throw new Error(
      "Replacement must use the compact confirmation dialog",
    );
  }
  const initEnd = html.indexOf(
    "</dialog>",
    html.indexOf('id="init-dialog"'),
  );
  const replaceStart = html.indexOf('id="replace-dialog"');
  if (replaceStart < initEnd) {
    throw new Error("Replacement confirmation must not be inside init");
  }
  if (
    !html.includes('id="use-run-directory"') || !html.includes("Change")
  ) {
    throw new Error(
      "Run directory action must use a stable Change label",
    );
  }
});

Deno.test("logs use cursor paging instead of a client window", async () => {
  const html = await Deno.readTextFile(
    new URL("../index.html", import.meta.url),
  );
  if (html.includes('id="log-window"')) {
    throw new Error("Log window is a forbidden legacy stream option");
  }
  if (!html.includes('id="load-older"')) {
    throw new Error("Logs must expose explicit cursor pagination");
  }
});
