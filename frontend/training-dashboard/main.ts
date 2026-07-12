import {
  fetchConfig,
  initializeTraining,
  resumeTraining,
  stopTraining,
} from "./api.ts";
import {
  INIT_FIELDS,
  INIT_GROUPS,
  initCommandPreview,
  type InitRequest,
  initRequestFromForm,
  RESUME_FIELDS,
  RESUME_GROUPS,
  resumeCommandPreview,
  resumeRequestFromForm,
  type TrainingField,
} from "./fields.ts";
import {
  TrainingStreamClient,
  type TrainingStreamTarget,
} from "./stream.ts";
import type {
  CheckpointCatalog,
  CheckpointManifest,
  TelemetryEvent,
  TrainingMetric,
  TrainingProcess,
  TrainingRunStatus,
  TrainingStreamSnapshot,
} from "./types.ts";

type ViewName = "overview" | "metrics" | "logs" | "checkpoints";
type LogStream = "stdout" | "stderr";
type ConnectionState = "online" | "offline" | "pending";

let runDir = "";
let process: TrainingProcess | null = null;
let runStatus: TrainingRunStatus | null = null;
let metrics: readonly TrainingMetric[] = [];
let telemetry: readonly TelemetryEvent[] = [];
let checkpoints: CheckpointCatalog | null = null;
let logStream: LogStream = "stdout";
let logContent = "";
let streamLogSubscription: LogStream | null = null;
let stopping = false;
let initializing = false;
let resuming = false;
let pendingInitRequest: InitRequest | null = null;

const directoryForm = element("directory-form", HTMLFormElement);
const directoryInput = element("run-directory", HTMLInputElement);
const useDirectoryButton = element(
  "use-run-directory",
  HTMLButtonElement,
);
const initDialog = element("init-dialog", HTMLDialogElement);
const initForm = element("init-form", HTMLFormElement);
const replaceDialog = element("replace-dialog", HTMLDialogElement);
const replaceForm = element("replace-form", HTMLFormElement);
const resumeDialog = element("resume-dialog", HTMLDialogElement);
const resumeForm = element("resume-form", HTMLFormElement);
const checkpointDialog = element(
  "checkpoint-dialog",
  HTMLDialogElement,
);
const trainingStream = new TrainingStreamClient(streamTarget, {
  onSnapshot: receiveSnapshot,
  onConnectionChange: (connected) =>
    setConnection(
      connected ? "ONLINE" : "RECONNECTING",
      connected ? "online" : "pending",
    ),
  onError: showStreamError,
});

initialize();

function initialize(): void {
  renderFormFields("init-fields", INIT_GROUPS, INIT_FIELDS);
  renderFormFields("resume-fields", RESUME_GROUPS, RESUME_FIELDS);
  bindEvents();
  renderRoute();
  void loadServerConfig();
}

function bindEvents(): void {
  globalThis.addEventListener("hashchange", () => renderRoute());
  directoryInput.addEventListener("input", renderDirectoryAction);
  directoryForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const next = directoryInput.value.trim();
    if (next === "") {
      element("directory-error", HTMLElement).textContent =
        "Run directory is required";
      return;
    }
    runDir = next;
    renderDirectoryAction();
    resetRunData();
    setConnection("CONNECTING", "pending");
    trainingStream.connect();
  });
  element("open-init", HTMLButtonElement).addEventListener(
    "click",
    () => {
      element("init-status", HTMLElement).textContent = "";
      renderInitCommand();
      initDialog.showModal();
    },
  );
  element("open-resume", HTMLButtonElement).addEventListener(
    "click",
    () => {
      renderResumeCommand();
      resumeDialog.showModal();
    },
  );
  initForm.addEventListener(
    "submit",
    prepareInitialization,
  );
  replaceForm.addEventListener("submit", confirmReplacement);
  element("replace-existing", HTMLInputElement).addEventListener(
    "input",
    renderReplacementAction,
  );
  replaceDialog.addEventListener("close", resetReplacementConfirmation);
  resumeForm.addEventListener("submit", (event) => void resume(event));
  for (
    const button of document.querySelectorAll<HTMLButtonElement>(
      "[data-close-dialog]",
    )
  ) {
    button.addEventListener("click", () => {
      const dialog = button.closest("dialog");
      if (!(dialog instanceof HTMLDialogElement)) {
        throw new Error("Dialog close control is outside a dialog");
      }
      dialog.close();
    });
  }
  element("stop-training", HTMLButtonElement).addEventListener(
    "click",
    () => void stop(),
  );
  element("close-checkpoint", HTMLButtonElement).addEventListener(
    "click",
    () => checkpointDialog.close(),
  );
  initForm.addEventListener("input", renderInitCommand);
  resumeForm.addEventListener("input", renderResumeCommand);
  for (
    const button of document.querySelectorAll<HTMLButtonElement>(
      "[data-refresh]",
    )
  ) {
    button.addEventListener("click", () => trainingStream.connect());
  }
  for (
    const button of document.querySelectorAll<HTMLButtonElement>(
      "[data-stream]",
    )
  ) {
    button.addEventListener("click", () => {
      logStream = button.dataset.stream === "stderr"
        ? "stderr"
        : "stdout";
      renderLogSegments();
      if (currentRoute() === "logs") {
        streamLogSubscription = logStream;
        trainingStream.connect();
      }
    });
  }
}

async function loadServerConfig(): Promise<void> {
  try {
    const config = await fetchConfig();
    runDir = config.default_run_dir;
    directoryInput.value = runDir;
    renderDirectoryAction();
    element("directory-error", HTMLElement).textContent = "";
    setConnection("CONNECTING", "pending");
    trainingStream.connect();
  } catch (error: unknown) {
    setConnection(errorText(error), "offline");
  }
}

function streamTarget(): TrainingStreamTarget | null {
  if (runDir === "") return null;
  return {
    runDir,
    metricSequence: metrics.at(-1)?.sequence ?? null,
    telemetrySequence: telemetry.at(-1)?.sequence ?? null,
    logStream: streamLogSubscription,
  };
}

function receiveSnapshot(snapshot: TrainingStreamSnapshot): void {
  runStatus = snapshot.summary;
  process = snapshot.summary.process;
  metrics = appendBounded(metrics, snapshot.summary.metrics, 5000);
  telemetry = appendBounded(
    telemetry,
    snapshot.summary.telemetry,
    5000,
  );
  checkpoints = snapshot.summary.checkpoints;
  if (
    snapshot.log_stream === logStream &&
    snapshot.log_content !== null
  ) {
    logContent = snapshot.log_content;
  }
  renderAll();
  setConnection("ONLINE", "online");
  element("directory-error", HTMLElement).textContent = "";
}

function showStreamError(message: string): void {
  element("directory-error", HTMLElement).textContent = message;
  setConnection(message, "offline");
}

function renderDirectoryAction(): void {
  const isActive = directoryInput.value.trim() === runDir;
  useDirectoryButton.disabled = isActive;
}

function appendBounded<T>(
  current: readonly T[],
  added: readonly T[],
  limit: number,
): readonly T[] {
  return [...current, ...added].slice(-limit);
}

function resetRunData(): void {
  process = null;
  runStatus = null;
  metrics = [];
  telemetry = [];
  checkpoints = null;
  logContent = "";
  renderAll();
}

function renderAll(): void {
  renderRun();
  renderMetrics();
  renderTelemetry();
  renderCheckpoints();
  renderLogContent();
}

function renderRun(): void {
  element("run-caption", HTMLElement).textContent = runDir;
  const presence = element("process-presence", HTMLElement);
  const state = runStatus?.state ?? "-";
  const latestManifest = checkpoints?.manifests.find(
    (manifest) => manifest.name === "latest.json" && manifest.valid,
  );
  presence.textContent = state;
  presence.className = statusBadgeClass(state);
  replaceWithRows(element("process-details", HTMLElement), [
    ["PID", process === null ? "-" : String(process.pid)],
    ["Kernel state", process?.kernel_state ?? "-"],
    ["Executable", process?.executable ?? "-"],
    ["Working directory", process?.working_directory ?? "-"],
    [
      "Process group",
      process === null ? "-" : String(process.process_group_id),
    ],
    ["Session", process === null ? "-" : String(process.session_id)],
  ]);
  const latestCoordinator = [...telemetry]
    .reverse()
    .find((event) => event.process_label === "coordinator");
  replaceWithRows(element("runtime-details", HTMLElement), [
    ["Run status", state],
    ["Status reason", runStatus?.reason ?? "-"],
    ["Coordinator stage", latestCoordinator?.stage ?? "-"],
    [
      "Last observed",
      formatTime(latestCoordinator?.recorded_at_ms ?? null),
    ],
    ["Database updates", formatValue(metrics.at(-1)?.total_updates)],
    [
      "Checkpoint manifests",
      String(checkpoints?.manifests.length ?? 0),
    ],
    [
      "Unique checkpoint storage",
      formatBytes(checkpoints?.total_unique_state_bytes ?? 0),
    ],
    [
      "Latest checkpoint",
      runStatus?.details?.checkpoint_path ??
        (latestManifest === undefined
          ? "-"
          : `${runDir}/checkpoints/latest.json`),
    ],
    [
      "Model configuration",
      formatConfiguration(
        runStatus?.details?.model_config_values ??
          latestManifest?.model_config_values,
      ),
    ],
    [
      "Training configuration",
      formatConfiguration(
        runStatus?.details?.train_config_values ??
          latestManifest?.train_config_values,
      ),
    ],
  ]);
  element("process-command", HTMLElement).textContent =
    process?.argv.map(shellQuote).join(" ") ?? "No managed process";
  const latest = metrics.at(-1);
  const overview = element("overview", HTMLElement);
  overview.replaceChildren(
    overviewCell(
      "Process",
      runStatus?.state === "RUNNING" && process !== null
        ? `PID ${process.pid}`
        : state,
    ),
    overviewCell("Games", formatValue(latest?.total_games)),
    overviewCell("Samples", formatValue(latest?.total_samples)),
    overviewCell("Updates", formatValue(latest?.total_updates)),
    overviewCell(
      "Samples / second",
      formatValue(latest?.process_samples_per_second),
    ),
  );
  element("open-init", HTMLButtonElement).disabled =
    runStatus === null ||
    runStatus.state === "RUNNING" || stopping || initializing ||
    resuming;
  element("open-resume", HTMLButtonElement).disabled =
    runStatus?.state !== "READY" || stopping || initializing ||
    resuming;
  element("stop-training", HTMLButtonElement).disabled =
    runStatus?.state !== "RUNNING" || stopping;
  element("stop-training", HTMLButtonElement).textContent = stopping
    ? "Stopping..."
    : "Stop";
}

function renderMetrics(): void {
  const latest = metrics.at(-1);
  element("metric-strip", HTMLElement).replaceChildren(
    metricCell("Total games", formatValue(latest?.total_games)),
    metricCell("Total samples", formatValue(latest?.total_samples)),
    metricCell("Total updates", formatValue(latest?.total_updates)),
    metricCell(
      "Samples / second",
      formatValue(latest?.process_samples_per_second),
    ),
    metricCell(
      "Games / second",
      formatValue(latest?.process_games_per_second),
    ),
  );
  const body = element("metric-rows", HTMLTableSectionElement);
  body.replaceChildren(
    ...metrics.slice(-200).reverse().map((metric) =>
      row([
        formatValue(metric.total_updates),
        formatValue(metric.total_samples),
        formatValue(metric.policy_loss),
        formatValue(metric.value_loss),
        formatValue(metric.entropy),
        formatValue(metric.approx_kl),
        formatSeconds(metric.ppo_update_seconds),
      ])
    ),
  );
  drawMetricChart();
}

function renderTelemetry(): void {
  const latest = new Map<string, TelemetryEvent>();
  for (const event of telemetry) latest.set(event.process_label, event);
  element("telemetry-rows", HTMLTableSectionElement).replaceChildren(
    ...[...latest.values()].sort((left, right) =>
      left.process_label.localeCompare(right.process_label)
    ).map((event) =>
      row([
        event.process_label,
        event.stage,
        String(event.total_updates),
        `${event.progress_numerator} / ${event.progress_denominator}`,
        event.measurements.map((item) =>
          `${item.key}=${formatValue(item.value)}`
        ).join(" · ") || "-",
        formatTime(event.recorded_at_ms),
      ])
    ),
  );
}

function renderCheckpoints(): void {
  element("checkpoint-directory", HTMLElement).textContent =
    checkpoints?.checkpoint_directory ?? `${runDir}/checkpoints`;
  const validCount =
    checkpoints?.manifests.filter((item) => item.valid).length ?? 0;
  const orphanCount =
    checkpoints?.objects.filter((item) => item.orphan).length ?? 0;
  element("checkpoint-summary", HTMLElement).replaceChildren(
    metricCell("Valid manifests", String(validCount)),
    metricCell("Objects", String(checkpoints?.objects.length ?? 0)),
    metricCell("Orphan objects", String(orphanCount)),
    metricCell(
      "Unique storage",
      formatBytes(checkpoints?.total_unique_state_bytes ?? 0),
    ),
  );
  const manifestRows = element(
    "manifest-rows",
    HTMLTableSectionElement,
  );
  manifestRows.replaceChildren(
    ...(checkpoints?.manifests ?? []).map((manifest) => {
      const actions = document.createElement("div");
      actions.className = "table-actions";
      actions.append(
        actionButton("Inspect", () => showCheckpoint(manifest)),
        actionButton(
          "Resume",
          () => resumeFrom(manifest),
          manifest.valid && runStatus?.state === "READY",
        ),
      );
      const condition = manifest.valid
        ? "Validated by summary"
        : manifest.error ?? "Invalid";
      return rowWithNode(
        [
          manifest.name,
          formatValue(manifest.total_updates),
          formatValue(manifest.total_samples),
          manifest.state_exists ? "Available" : "Missing",
          formatBytes(manifest.state_size_bytes),
          condition,
        ],
        actions,
        manifest.valid ? "" : "invalid-row",
      );
    }),
  );
  element("object-rows", HTMLTableSectionElement).replaceChildren(
    ...(checkpoints?.objects ?? []).map((item) =>
      row([
        item.checkpoint_id,
        item.state_path,
        formatBytes(item.state_size_bytes),
        item.referenced_by.join(", ") || "None",
        item.error ?? (item.orphan ? "Orphan" : "Referenced"),
      ], item.valid ? "" : "invalid-row")
    ),
  );
}

function prepareInitialization(event: Event): void {
  event.preventDefault();
  if (initializing) return;
  if (!initForm.reportValidity()) return;
  const request = initRequestFromForm(initForm, runDir, null);
  if (runStatus?.state === "READY" || runStatus?.state === "BROKEN") {
    pendingInitRequest = request;
    element("replace-status", HTMLElement).textContent =
      `Current state: ${runStatus.state}`;
    element("replace-run-directory", HTMLElement).textContent = runDir;
    element("replace-result", HTMLElement).textContent = "";
    initDialog.close();
    replaceDialog.showModal();
    element("replace-existing", HTMLInputElement).focus();
    return;
  }
  void runInitialization(
    request,
    initDialog,
    element("init-status", HTMLElement),
    element("confirm-init", HTMLButtonElement),
    "Initializing...",
    "Initialize",
  );
}

function confirmReplacement(event: Event): void {
  event.preventDefault();
  if (initializing || !replaceForm.reportValidity()) return;
  const request = pendingInitRequest;
  if (request === null) {
    throw new Error("Missing pending initialization request");
  }
  void runInitialization(
    { ...request, replace_existing: "yes" },
    replaceDialog,
    element("replace-result", HTMLElement),
    element("confirm-replace", HTMLButtonElement),
    "Replacing...",
    "Replace and initialize",
  );
}

async function runInitialization(
  request: InitRequest,
  dialog: HTMLDialogElement,
  status: HTMLElement,
  button: HTMLButtonElement,
  pendingLabel: string,
  idleLabel: string,
): Promise<void> {
  initializing = true;
  button.disabled = true;
  button.textContent = pendingLabel;
  status.className = "status-value";
  status.textContent = "Creating initial checkpoint...";
  renderRun();
  try {
    await initializeTraining(request);
    status.textContent = "";
    dialog.close();
    trainingStream.connect();
  } catch (reason: unknown) {
    status.className = "error-value";
    status.textContent = errorText(reason);
  } finally {
    initializing = false;
    button.disabled = false;
    button.textContent = idleLabel;
    renderReplacementAction();
    renderRun();
  }
}

function renderReplacementAction(): void {
  const input = element("replace-existing", HTMLInputElement);
  element("confirm-replace", HTMLButtonElement).disabled =
    initializing || input.value !== "yes";
}

function resetReplacementConfirmation(): void {
  pendingInitRequest = null;
  const input = element("replace-existing", HTMLInputElement);
  input.value = "";
  renderReplacementAction();
}

async function resume(event: Event): Promise<void> {
  event.preventDefault();
  if (resuming) return;
  if (!resumeForm.reportValidity()) return;
  const status = element("resume-status", HTMLElement);
  const button = element("confirm-resume", HTMLButtonElement);
  resuming = true;
  button.disabled = true;
  button.textContent = "Starting...";
  status.className = "status-value";
  status.textContent = "Starting training process...";
  renderRun();
  try {
    process = await resumeTraining(
      resumeRequestFromForm(resumeForm, runDir),
    );
    status.textContent = "";
    resumeDialog.close();
    renderRun();
    trainingStream.connect();
  } catch (reason: unknown) {
    status.className = "error-value";
    status.textContent = errorText(reason);
  } finally {
    resuming = false;
    button.disabled = false;
    button.textContent = "Resume";
    renderRun();
  }
}

async function stop(): Promise<void> {
  stopping = true;
  renderRun();
  try {
    const forced = await stopTraining(runDir);
    process = null;
    if (forced) {
      element("directory-error", HTMLElement).textContent =
        "Stop timeout exceeded; process group was killed";
    }
    trainingStream.connect();
  } catch (error: unknown) {
    element("directory-error", HTMLElement).textContent = errorText(
      error,
    );
  } finally {
    stopping = false;
    renderRun();
  }
}

function renderLogContent(): void {
  const target = element("log-content", HTMLElement);
  const atEnd = target.scrollTop + target.clientHeight >=
    target.scrollHeight - 24;
  target.textContent = logContent;
  if (atEnd) target.scrollTop = target.scrollHeight;
}

function resumeFrom(manifest: CheckpointManifest): void {
  if (!manifest.valid) return;
  const input = resumeForm.elements.namedItem("checkpoint");
  if (!(input instanceof HTMLInputElement)) {
    throw new Error("Missing checkpoint field");
  }
  input.value = manifest.name;
  renderResumeCommand();
  resumeDialog.showModal();
}

function showCheckpoint(manifest: CheckpointManifest): void {
  element("checkpoint-dialog-title", HTMLElement).textContent =
    manifest.name;
  element("checkpoint-dialog-content", HTMLElement).textContent = JSON
    .stringify(manifest, null, 2);
  checkpointDialog.showModal();
}

function renderFormFields(
  targetId: string,
  groups: readonly string[],
  fields: readonly TrainingField[],
): void {
  element(targetId, HTMLElement).replaceChildren(
    ...groups.map((group) => {
      const section = document.createElement("section");
      const heading = document.createElement("h3");
      heading.textContent = group;
      const grid = document.createElement("div");
      grid.className = "launch-field-grid";
      grid.append(
        ...fields.filter((field) => field.group === group).map(
          renderTrainingField,
        ),
      );
      section.append(heading, grid);
      return section;
    }),
  );
}

function renderTrainingField(field: TrainingField): HTMLElement {
  const label = document.createElement("label");
  label.className = field.kind === "checkbox"
    ? "field checkbox-field"
    : "field";
  const caption = document.createElement("span");
  caption.textContent = field.label;
  const flag = document.createElement("code");
  flag.textContent = field.flag;
  caption.append(flag);
  label.append(
    caption,
    field.kind === "profile"
      ? profileControl(field)
      : inputControl(field),
  );
  return label;
}

function inputControl(field: TrainingField): HTMLInputElement {
  const input = document.createElement("input");
  input.name = field.key;
  input.className = field.kind === "checkbox"
    ? "toggle-input"
    : "input";
  input.type = field.kind;
  if (field.kind === "checkbox") {
    input.checked = field.defaultValue === true;
  } else {
    input.value = String(field.defaultValue);
    if (field.min !== undefined) input.min = field.min;
    if (field.max !== undefined) input.max = field.max;
    if (field.step !== undefined) input.step = field.step;
    input.required = field.optional !== true;
  }
  return input;
}

function profileControl(field: TrainingField): HTMLSelectElement {
  const select = document.createElement("select");
  select.name = field.key;
  select.className = "input";
  for (const value of ["", "off", "basic", "detailed"]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value === "" ? "Inherit default" : value;
    option.selected = value === field.defaultValue;
    select.append(option);
  }
  return select;
}

function renderInitCommand(): void {
  try {
    element("init-command-preview", HTMLElement).textContent =
      initCommandPreview(initRequestFromForm(initForm, runDir, null));
  } catch (error: unknown) {
    element("init-command-preview", HTMLElement).textContent =
      errorText(
        error,
      );
  }
}

function renderResumeCommand(): void {
  try {
    element("resume-command-preview", HTMLElement).textContent =
      resumeCommandPreview(resumeRequestFromForm(resumeForm, runDir));
  } catch (error: unknown) {
    element("resume-command-preview", HTMLElement).textContent =
      errorText(
        error,
      );
  }
}

function renderRoute(): void {
  const route = currentRoute();
  for (
    const view of document.querySelectorAll<HTMLElement>("[data-view]")
  ) {
    view.hidden = view.dataset.view !== route;
  }
  for (
    const link of document.querySelectorAll<HTMLElement>("[data-route]")
  ) {
    link.classList.toggle("active", link.dataset.route === route);
  }
  const nextLogSubscription = route === "logs" ? logStream : null;
  if (nextLogSubscription !== streamLogSubscription) {
    streamLogSubscription = nextLogSubscription;
    if (runDir !== "") trainingStream.connect();
  }
}

function currentRoute(): ViewName {
  const route = globalThis.location.hash.slice(1);
  if (
    route === "metrics" || route === "logs" || route === "checkpoints"
  ) return route;
  return "overview";
}

function renderLogSegments(): void {
  for (
    const button of document.querySelectorAll<HTMLButtonElement>(
      "[data-stream]",
    )
  ) {
    button.classList.toggle(
      "active",
      button.dataset.stream === logStream,
    );
  }
}

function drawMetricChart(): void {
  const canvas = element("metric-chart", HTMLCanvasElement);
  const context = canvas.getContext("2d");
  if (context === null) return;
  const values = metrics.slice(-500).map((metric) =>
    metric.process_samples_per_second
  );
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = "#dce1e6";
  context.lineWidth = 1;
  for (let index = 1; index < 5; index += 1) {
    const y = index * canvas.height / 5;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(canvas.width, y);
    context.stroke();
  }
  if (values.length < 2) return;
  const maximum = Math.max(...values, 1);
  context.strokeStyle = "#1769aa";
  context.lineWidth = 3;
  context.beginPath();
  values.forEach((value, index) => {
    const x = index * canvas.width / (values.length - 1);
    const y = canvas.height - value / maximum * (canvas.height - 20) -
      10;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
}

function replaceWithRows(
  parent: HTMLElement,
  rows: readonly (readonly [string, string])[],
): void {
  parent.replaceChildren(...rows.map(([label, value]) => {
    const item = document.createElement("div");
    item.className = "detail-row";
    const key = document.createElement("span");
    key.textContent = label;
    const output = document.createElement("strong");
    output.textContent = value;
    item.append(key, output);
    return item;
  }));
}

function overviewCell(label: string, value: string): HTMLElement {
  const cell = document.createElement("div");
  cell.className = "overview-cell";
  const caption = document.createElement("span");
  caption.textContent = label;
  const output = document.createElement("strong");
  output.textContent = value;
  cell.append(caption, output);
  return cell;
}

function metricCell(label: string, value: string): HTMLElement {
  const cell = document.createElement("div");
  cell.className = "metric-cell";
  const caption = document.createElement("span");
  caption.textContent = label;
  const output = document.createElement("strong");
  output.textContent = value;
  cell.append(caption, output);
  return cell;
}

function row(
  values: readonly string[],
  className = "",
): HTMLTableRowElement {
  const item = document.createElement("tr");
  item.className = className;
  for (const value of values) {
    const cell = document.createElement("td");
    cell.textContent = value;
    item.append(cell);
  }
  return item;
}

function rowWithNode(
  values: readonly string[],
  node: Node,
  className: string,
): HTMLTableRowElement {
  const item = row(values, className);
  const cell = document.createElement("td");
  cell.append(node);
  item.append(cell);
  return item;
}

function actionButton(
  label: string,
  action: () => void,
  enabled = true,
): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "table-action";
  button.textContent = label;
  button.disabled = !enabled;
  button.addEventListener("click", action);
  return button;
}

function statusBadgeClass(state: string): string {
  if (state === "READY") return "badge success";
  if (state === "RUNNING") return "badge running";
  if (state === "BROKEN") return "badge danger";
  return "badge neutral";
}

function formatConfiguration(
  value:
    | Readonly<Record<string, string | number | boolean | null>>
    | null
    | undefined,
): string {
  if (value === null || value === undefined) return "-";
  return Object.entries(value)
    .map(([key, item]) => `${key}=${String(item)}`)
    .join(", ");
}

function setConnection(
  label: string,
  state: ConnectionState,
): void {
  const target = element("connection-state", HTMLElement);
  target.textContent = label;
  target.className = state === "pending"
    ? "connection-label"
    : `connection-label ${state}`;
}

function formatValue(
  value: number | string | null | undefined,
): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "-";
    if (Number.isInteger(value)) return value.toLocaleString("en-US");
    return value.toLocaleString("en-US", { maximumFractionDigits: 5 });
  }
  return value;
}

function formatBytes(value: number | null): string {
  if (value === null) return "-";
  if (value < 1024) return `${value} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let output = value / 1024;
  let unit = units[0];
  for (
    let index = 1;
    index < units.length && output >= 1024;
    index += 1
  ) {
    output /= 1024;
    unit = units[index];
  }
  return `${output.toFixed(output >= 10 ? 1 : 2)} ${unit}`;
}

function formatSeconds(value: number | null): string {
  return value === null ? "-" : `${formatValue(value)} s`;
}

function formatTime(value: number | null): string {
  return value === null ? "-" : new Date(value).toLocaleString("en-GB");
}

function shellQuote(value: string): string {
  return /^[A-Za-z0-9_./,:+=-]+$/.test(value)
    ? value
    : `'${value.replaceAll("'", `'"'"'`)}'`;
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function element<T extends HTMLElement>(
  id: string,
  constructor: { new (): T },
): T {
  const value = document.getElementById(id);
  if (!(value instanceof constructor)) {
    throw new Error(`Missing element: ${id}`);
  }
  return value;
}
