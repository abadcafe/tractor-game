import { recordValue } from "../browser/json.ts";

export type FormGroup =
  | "Checkpoint"
  | "Model"
  | "Optimization"
  | "Runtime"
  | "Timeouts";

export interface InitRequest {
  run_dir: string;
  replace_existing: "yes" | null;
  d_model: number;
  layers: number;
  heads: number;
  max_tokens: number;
  seed: number;
  learning_rate: number;
  ppo_clip: number;
  value_clip: number;
  entropy_coef: number;
  value_coef: number;
  max_grad_norm: number;
  ppo_epochs: number;
  minibatch_size: number;
  adam_beta1: number;
  adam_beta2: number;
  weight_decay: number;
}

export interface ResumeRequest {
  run_dir: string;
  checkpoint: string;
  worker_cpus: string | null;
  model_ranks: string | null;
  ppo_profile: "off" | "basic" | "detailed" | null;
  max_samples: number;
  learning_rate: number | null;
  checkpoint_every_updates: number;
  checkpoint_retention_updates: number;
  round_timeout_seconds: number | null;
  sampling_start_timeout_seconds: number | null;
  rollout_sample_timeout_seconds: number | null;
  sampling_stop_timeout_seconds: number | null;
  state_sync_timeout_seconds: number | null;
  update_timeout_seconds: number | null;
  telemetry_interval_seconds: number | null;
  model_inference_batch_size: number | null;
  game_envs_per_worker: number | null;
  samples_per_update: number | null;
  ppo_clip: number | null;
  value_clip: number | null;
  entropy_coef: number | null;
  value_coef: number | null;
  max_grad_norm: number | null;
  ppo_epochs: number | null;
  minibatch_size: number | null;
  adam_beta1: number | null;
  adam_beta2: number | null;
  weight_decay: number | null;
}

type FieldKind = "text" | "number" | "checkbox" | "profile";
type RequestKey = Exclude<
  keyof InitRequest | keyof ResumeRequest,
  "run_dir"
>;

export interface TrainingField {
  readonly key: RequestKey;
  readonly flag: `--${string}`;
  readonly label: string;
  readonly group: FormGroup;
  readonly kind: FieldKind;
  readonly defaultValue: string | boolean;
  readonly min?: string;
  readonly max?: string;
  readonly step?: string;
  readonly optional?: boolean;
}

const text = (
  key: RequestKey,
  label: string,
  group: FormGroup,
  defaultValue: string,
  optional = true,
): TrainingField => ({
  key,
  flag: `--${key.replaceAll("_", "-")}`,
  label,
  group,
  kind: "text",
  defaultValue,
  optional,
});

const number = (
  key: RequestKey,
  label: string,
  group: FormGroup,
  defaultValue: string,
  min: string,
  step: string,
  optional = true,
  max?: string,
): TrainingField => ({
  key,
  flag: `--${key.replaceAll("_", "-")}`,
  label,
  group,
  kind: "number",
  defaultValue,
  min,
  max,
  step,
  optional,
});

const optimizationFields = (
  optional: boolean,
): readonly TrainingField[] => [
  number(
    "learning_rate",
    "Learning rate",
    "Optimization",
    optional ? "" : "0.0003",
    "0.000000001",
    "any",
    optional,
  ),
  number(
    "ppo_clip",
    "PPO clip",
    "Optimization",
    optional ? "" : "0.2",
    "0.000001",
    "any",
    optional,
    "1",
  ),
  number(
    "value_clip",
    "Value clip",
    "Optimization",
    optional ? "" : "0.2",
    "0.000001",
    "any",
    optional,
  ),
  number(
    "entropy_coef",
    "Entropy coefficient",
    "Optimization",
    optional ? "" : "0.01",
    "0",
    "any",
    optional,
  ),
  number(
    "value_coef",
    "Value coefficient",
    "Optimization",
    optional ? "" : "0.5",
    "0",
    "any",
    optional,
  ),
  number(
    "max_grad_norm",
    "Maximum gradient norm",
    "Optimization",
    optional ? "" : "0.5",
    "0",
    "any",
    optional,
  ),
  number(
    "ppo_epochs",
    "PPO epochs",
    "Optimization",
    optional ? "" : "4",
    "1",
    "1",
    optional,
  ),
  number(
    "minibatch_size",
    "Minibatch size",
    "Optimization",
    optional ? "" : "64",
    "1",
    "1",
    optional,
  ),
  number(
    "adam_beta1",
    "Adam beta 1",
    "Optimization",
    optional ? "" : "0.9",
    "0",
    "any",
    optional,
    "0.999999",
  ),
  number(
    "adam_beta2",
    "Adam beta 2",
    "Optimization",
    optional ? "" : "0.999",
    "0",
    "any",
    optional,
    "0.999999",
  ),
  number(
    "weight_decay",
    "Weight decay",
    "Optimization",
    optional ? "" : "0",
    "0",
    "any",
    optional,
  ),
];

export const INIT_GROUPS: readonly FormGroup[] = [
  "Model",
  "Optimization",
];

export const INIT_FIELDS: readonly TrainingField[] = [
  number("d_model", "Model width", "Model", "128", "1", "1", false),
  number("layers", "Transformer layers", "Model", "3", "1", "1", false),
  number("heads", "Attention heads", "Model", "4", "1", "1", false),
  number(
    "max_tokens",
    "Maximum tokens",
    "Model",
    "768",
    "512",
    "1",
    false,
  ),
  number("seed", "Seed", "Model", "0", "0", "1", false),
  ...optimizationFields(false),
];

export const RESUME_GROUPS: readonly FormGroup[] = [
  "Checkpoint",
  "Optimization",
  "Runtime",
  "Timeouts",
];

export const RESUME_FIELDS: readonly TrainingField[] = [
  text(
    "checkpoint",
    "Checkpoint manifest",
    "Checkpoint",
    "latest.json",
    false,
  ),
  number(
    "max_samples",
    "Maximum samples",
    "Checkpoint",
    "0",
    "0",
    "1",
    false,
  ),
  number(
    "checkpoint_every_updates",
    "Checkpoint interval",
    "Checkpoint",
    "50",
    "1",
    "1",
    false,
  ),
  number(
    "checkpoint_retention_updates",
    "Checkpoint retention",
    "Checkpoint",
    "5",
    "0",
    "1",
    false,
  ),
  ...optimizationFields(true),
  text("worker_cpus", "Worker CPUs", "Runtime", ""),
  text("model_ranks", "Model ranks", "Runtime", ""),
  {
    key: "ppo_profile",
    flag: "--ppo-profile",
    label: "PPO profiling",
    group: "Runtime",
    kind: "profile",
    defaultValue: "",
    optional: true,
  },
  number(
    "telemetry_interval_seconds",
    "Telemetry interval",
    "Runtime",
    "",
    "0.001",
    "any",
  ),
  number(
    "model_inference_batch_size",
    "Inference batch size",
    "Runtime",
    "",
    "1",
    "1",
  ),
  number(
    "game_envs_per_worker",
    "Game environments per worker",
    "Runtime",
    "",
    "1",
    "1",
  ),
  number(
    "samples_per_update",
    "Samples per update",
    "Runtime",
    "",
    "1",
    "1",
  ),
  number(
    "round_timeout_seconds",
    "Round timeout",
    "Timeouts",
    "",
    "0.001",
    "any",
  ),
  number(
    "sampling_start_timeout_seconds",
    "Sampling start timeout",
    "Timeouts",
    "",
    "0.001",
    "any",
  ),
  number(
    "rollout_sample_timeout_seconds",
    "Rollout sample timeout",
    "Timeouts",
    "",
    "0.001",
    "any",
  ),
  number(
    "sampling_stop_timeout_seconds",
    "Sampling stop timeout",
    "Timeouts",
    "",
    "0.001",
    "any",
  ),
  number(
    "state_sync_timeout_seconds",
    "State sync timeout",
    "Timeouts",
    "",
    "0.001",
    "any",
  ),
  number(
    "update_timeout_seconds",
    "Update timeout",
    "Timeouts",
    "",
    "0.001",
    "any",
  ),
];

export function initRequestFromForm(
  form: HTMLFormElement,
  runDir: string,
  replaceExisting: "yes" | null,
): InitRequest {
  return {
    run_dir: runDir,
    replace_existing: replaceExisting,
    d_model: requiredNumber(form, "d_model"),
    layers: requiredNumber(form, "layers"),
    heads: requiredNumber(form, "heads"),
    max_tokens: requiredNumber(form, "max_tokens"),
    seed: requiredNumber(form, "seed"),
    learning_rate: requiredNumber(form, "learning_rate"),
    ppo_clip: requiredNumber(form, "ppo_clip"),
    value_clip: requiredNumber(form, "value_clip"),
    entropy_coef: requiredNumber(form, "entropy_coef"),
    value_coef: requiredNumber(form, "value_coef"),
    max_grad_norm: requiredNumber(form, "max_grad_norm"),
    ppo_epochs: requiredNumber(form, "ppo_epochs"),
    minibatch_size: requiredNumber(form, "minibatch_size"),
    adam_beta1: requiredNumber(form, "adam_beta1"),
    adam_beta2: requiredNumber(form, "adam_beta2"),
    weight_decay: requiredNumber(form, "weight_decay"),
  };
}

export function resumeRequestFromForm(
  form: HTMLFormElement,
  runDir: string,
): ResumeRequest {
  const profile = textValue(form, "ppo_profile");
  if (
    profile !== null && profile !== "off" && profile !== "basic" &&
    profile !== "detailed"
  ) {
    throw new Error("Invalid PPO profile");
  }
  return {
    run_dir: runDir,
    checkpoint: requiredText(form, "checkpoint"),
    worker_cpus: textValue(form, "worker_cpus"),
    model_ranks: textValue(form, "model_ranks"),
    ppo_profile: profile,
    max_samples: requiredNumber(form, "max_samples"),
    learning_rate: optionalNumber(form, "learning_rate"),
    checkpoint_every_updates: requiredNumber(
      form,
      "checkpoint_every_updates",
    ),
    checkpoint_retention_updates: requiredNumber(
      form,
      "checkpoint_retention_updates",
    ),
    round_timeout_seconds: optionalNumber(
      form,
      "round_timeout_seconds",
    ),
    sampling_start_timeout_seconds: optionalNumber(
      form,
      "sampling_start_timeout_seconds",
    ),
    rollout_sample_timeout_seconds: optionalNumber(
      form,
      "rollout_sample_timeout_seconds",
    ),
    sampling_stop_timeout_seconds: optionalNumber(
      form,
      "sampling_stop_timeout_seconds",
    ),
    state_sync_timeout_seconds: optionalNumber(
      form,
      "state_sync_timeout_seconds",
    ),
    update_timeout_seconds: optionalNumber(
      form,
      "update_timeout_seconds",
    ),
    telemetry_interval_seconds: optionalNumber(
      form,
      "telemetry_interval_seconds",
    ),
    model_inference_batch_size: optionalNumber(
      form,
      "model_inference_batch_size",
    ),
    game_envs_per_worker: optionalNumber(form, "game_envs_per_worker"),
    samples_per_update: optionalNumber(form, "samples_per_update"),
    ppo_clip: optionalNumber(form, "ppo_clip"),
    value_clip: optionalNumber(form, "value_clip"),
    entropy_coef: optionalNumber(form, "entropy_coef"),
    value_coef: optionalNumber(form, "value_coef"),
    max_grad_norm: optionalNumber(form, "max_grad_norm"),
    ppo_epochs: optionalNumber(form, "ppo_epochs"),
    minibatch_size: optionalNumber(form, "minibatch_size"),
    adam_beta1: optionalNumber(form, "adam_beta1"),
    adam_beta2: optionalNumber(form, "adam_beta2"),
    weight_decay: optionalNumber(form, "weight_decay"),
  };
}

export function initCommandPreview(request: InitRequest): string {
  const command = commandPreview("init", request, INIT_FIELDS);
  return request.replace_existing === null
    ? command
    : `${command} --replace-existing yes`;
}

export function resumeCommandPreview(request: ResumeRequest): string {
  return commandPreview("resume", request, RESUME_FIELDS);
}

function commandPreview(
  command: "init" | "resume",
  request: InitRequest | ResumeRequest,
  fields: readonly TrainingField[],
): string {
  const values = recordValue(request);
  if (values === null) throw new Error("Invalid training request");
  const parts = [
    "python",
    "-m",
    "server.training_cli",
    "--run-dir",
    shellQuote(request.run_dir),
    command,
  ];
  if (command === "resume") {
    const checkpoint = values.checkpoint;
    if (typeof checkpoint !== "string" || checkpoint === "") {
      throw new Error("Checkpoint is required");
    }
    parts.push(shellQuote(checkpoint));
  }
  for (const field of fields) {
    if (field.key === "checkpoint") continue;
    const value = values[field.key];
    if (typeof value === "boolean") {
      if (value) parts.push(field.flag);
    } else if (value !== null) {
      parts.push(field.flag, shellQuote(String(value)));
    }
  }
  return parts.join(" ");
}

function control(
  form: HTMLFormElement,
  key: string,
): HTMLInputElement | HTMLSelectElement {
  const value = form.elements.namedItem(key);
  if (
    !(value instanceof HTMLInputElement ||
      value instanceof HTMLSelectElement)
  ) {
    throw new Error(`Missing training control: ${key}`);
  }
  return value;
}

function textValue(form: HTMLFormElement, key: string): string | null {
  const value = control(form, key).value.trim();
  return value === "" ? null : value;
}

function requiredText(form: HTMLFormElement, key: string): string {
  const value = textValue(form, key);
  if (value === null) throw new Error(`${key} is required`);
  return value;
}

function optionalNumber(
  form: HTMLFormElement,
  key: string,
): number | null {
  const value = textValue(form, key);
  if (value === null) return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${key} must be numeric`);
  }
  return parsed;
}

function requiredNumber(form: HTMLFormElement, key: string): number {
  const value = optionalNumber(form, key);
  if (value === null) throw new Error(`${key} is required`);
  return value;
}

function shellQuote(value: string): string {
  if (/^[A-Za-z0-9_./,:+-]+$/.test(value)) return value;
  return `'${value.replaceAll("'", `'"'"'`)}'`;
}
