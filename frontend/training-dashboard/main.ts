import {
  fetchConfig,
  fetchMetrics,
  fetchSummary,
  initializeTraining,
  resumeTraining,
  stopTraining,
} from "./api.ts";
import { DashboardCharts, type MetricAxis } from "./charts.ts";
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
import { DashboardSelection } from "./selection.ts";
import { DashboardRefreshController } from "./refresh.ts";
import {
  type DashboardErrorSource,
  DashboardStatus,
} from "./status.ts";
import type {
  CheckpointManifest,
  JsonPrimitive,
  TrainingEvent,
  TrainingLogMessage,
  TrainingMetrics,
  TrainingProcess,
  TrainingSummary,
} from "./types.ts";

type ViewName = "overview" | "metrics" | "logs" | "checkpoints";
type LogEntry = {
  readonly sequence: number;
  readonly event: TrainingEvent;
};

const SUMMARY_REFRESH_INTERVAL_MS = 5_000;
const selection = new DashboardSelection();
const dashboardStatus = new DashboardStatus();
let summary: TrainingSummary | null = null;
let metrics: TrainingMetrics | null = null;
let process: TrainingProcess | null = null;
let logEvents: readonly LogEntry[] = [];
let pendingLogEvents: LogEntry[] = [];
let logRenderFrame: number | null = null;
let logWindow = 5000;
let followLogs = true;
let metricAxis: MetricAxis = "update";
let stopping = false;
let initializing = false;
let resuming = false;
let pendingInitRequest: InitRequest | null = null;
let metricRefreshTimer: ReturnType<typeof setTimeout> | null = null;
let summaryRefreshTimer: ReturnType<typeof setTimeout> | null = null;

const charts = new DashboardCharts();
const refreshController = new DashboardRefreshController(
  selection,
  { fetchSummary, fetchMetrics },
  {
    currentSummary: () => summary,
    applySummary: (nextSummary) => {
      summary = nextSummary;
      process = nextSummary.process;
      renderRun();
      renderCheckpoints();
    },
    applyMetrics: (nextMetrics) => {
      metrics = nextMetrics;
      renderMetrics();
    },
    setRefreshPending: () => {
      dashboardStatus.setRefreshPending();
      renderDashboardStatus();
    },
    setRefreshIdle: () => {
      dashboardStatus.setRefreshIdle();
      renderDashboardStatus();
    },
    reportError: reportDashboardError,
    clearError: clearDashboardError,
  },
);
const directoryInput = element("run-directory", HTMLInputElement);
const directoryForm = element("directory-form", HTMLFormElement);
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
const stream = new TrainingStreamClient(streamTarget, {
  onMessage: receiveLogMessage,
  onConnectionChange: (connected) => {
    dashboardStatus.setStreamConnection(
      connected ? "online" : "reconnecting",
    );
    if (connected) dashboardStatus.clearError("stream");
    renderDashboardStatus();
  },
  onError: (message) => reportDashboardError("stream", message),
});

initialize();

function initialize(): void {
  renderFormFields("init-fields", INIT_GROUPS, INIT_FIELDS);
  renderFormFields("resume-fields", RESUME_GROUPS, RESUME_FIELDS);
  bindEvents();
  renderRoute();
  scheduleSummaryRefresh();
  void loadServerConfig();
}

function bindEvents(): void {
  globalThis.addEventListener("hashchange", renderRoute);
  directoryInput.addEventListener("input", renderDirectoryAction);
  directoryForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const next = directoryInput.value.trim();
    if (next === "") {
      reportDashboardError("directory", "Run directory is required");
      return;
    }
    selection.setRunDirectory(next);
    resetRunData();
    renderDirectoryAction();
    void refreshAll();
    connectStream();
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
  initForm.addEventListener("submit", prepareInitialization);
  replaceForm.addEventListener("submit", confirmReplacement);
  resumeForm.addEventListener("submit", (event) => void resume(event));
  initForm.addEventListener("input", renderInitCommand);
  resumeForm.addEventListener("input", renderResumeCommand);
  element("replace-existing", HTMLInputElement).addEventListener(
    "input",
    renderReplacementAction,
  );
  replaceDialog.addEventListener("close", resetReplacementConfirmation);
  element("stop-training", HTMLButtonElement).addEventListener(
    "click",
    () => void stop(),
  );
  element("close-checkpoint", HTMLButtonElement).addEventListener(
    "click",
    () => checkpointDialog.close(),
  );
  for (
    const button of document.querySelectorAll<HTMLButtonElement>(
      "[data-close-dialog]",
    )
  ) {
    button.addEventListener(
      "click",
      () => button.closest("dialog")?.close(),
    );
  }
  for (
    const button of document.querySelectorAll<HTMLButtonElement>(
      "[data-refresh]",
    )
  ) {
    button.addEventListener("click", () => void refreshAll());
  }
  for (
    const button of document.querySelectorAll<HTMLButtonElement>(
      "[data-axis]",
    )
  ) {
    button.addEventListener("click", () => {
      metricAxis = button.dataset.axis === "elapsed"
        ? "elapsed"
        : "update";
      renderAxisButtons();
      renderMetrics();
    });
  }
  element("toggle-follow", HTMLButtonElement).addEventListener(
    "click",
    () => {
      followLogs = !followLogs;
      element("toggle-follow", HTMLButtonElement).textContent =
        followLogs ? "Pause" : "Follow";
      if (followLogs) scrollLogsToEnd();
    },
  );
  element("log-window", HTMLInputElement).addEventListener(
    "change",
    updateLogWindow,
  );
  element("metrics-range", HTMLSelectElement).addEventListener(
    "change",
    () => void refreshMetrics(),
  );
  element("metrics-resolution", HTMLSelectElement).addEventListener(
    "change",
    () => void refreshMetrics(),
  );
  element("metrics-session-select", HTMLSelectElement).addEventListener(
    "change",
    () => {
      selection.setMetricSession(
        element(
          "metrics-session-select",
          HTMLSelectElement,
        ).value || null,
      );
      void refreshMetrics();
    },
  );
}

async function loadServerConfig(): Promise<void> {
  const origin = selection.captureRun();
  try {
    const config = await fetchConfig();
    if (!selection.ownsRun(origin)) return;
    selection.setRunDirectory(config.default_run_dir);
    directoryInput.value = selection.runDir;
    renderDirectoryAction();
    await refreshAll();
    connectStream();
  } catch (error: unknown) {
    if (selection.ownsRun(origin)) {
      reportDashboardError("config", errorText(error));
    }
  }
}

async function refreshAll(): Promise<void> {
  await refreshController.refreshAll(metricRefreshOptions());
}

async function refreshMetrics(): Promise<void> {
  await refreshController.refreshMetrics(metricRefreshOptions());
}

function streamTarget(): TrainingStreamTarget | null {
  if (selection.runDir === "") return null;
  return {
    runDir: selection.runDir,
    window: logWindow,
    eventTypes: [],
    sessionId: null,
  };
}

function receiveLogMessage(message: TrainingLogMessage): void {
  if (message.type === "error") {
    reportDashboardError("stream", message.message);
    return;
  }
  if (message.type === "reset") {
    if (logRenderFrame !== null) cancelAnimationFrame(logRenderFrame);
    logRenderFrame = null;
    pendingLogEvents = [];
    logEvents = [];
    logWindow = message.window;
    element("log-window", HTMLInputElement).value = String(logWindow);
    renderRun();
    renderLogs();
    return;
  }
  pendingLogEvents.push({
    sequence: message.sequence,
    event: message.event,
  });
  scheduleLogRender();
  if (isMetricBoundary(message.event.event)) scheduleMetricRefresh();
  if (message.event.event.startsWith("session.")) {
    void refreshSummaryOnly();
  }
}

async function refreshSummaryOnly(): Promise<void> {
  await refreshController.refreshSummary();
}

function scheduleSummaryRefresh(): void {
  if (summaryRefreshTimer !== null) return;
  summaryRefreshTimer = setTimeout(() => {
    summaryRefreshTimer = null;
    const refresh = process !== null
      ? refreshSummaryOnly()
      : Promise.resolve();
    void refresh.finally(scheduleSummaryRefresh);
  }, SUMMARY_REFRESH_INTERVAL_MS);
}

function scheduleMetricRefresh(): void {
  if (currentRoute() !== "metrics") return;
  if (metricRefreshTimer !== null) clearTimeout(metricRefreshTimer);
  metricRefreshTimer = setTimeout(() => {
    metricRefreshTimer = null;
    void refreshMetrics();
  }, 500);
}

function isMetricBoundary(eventType: string): boolean {
  return eventType === "update.completed" ||
    eventType === "rollout.completed" ||
    eventType.startsWith("session.");
}

function resetRunData(): void {
  if (logRenderFrame !== null) cancelAnimationFrame(logRenderFrame);
  logRenderFrame = null;
  pendingLogEvents = [];
  if (metricRefreshTimer !== null) clearTimeout(metricRefreshTimer);
  metricRefreshTimer = null;
  summary = null;
  process = null;
  metrics = null;
  logEvents = [];
  selection.setMetricSession(null);
  dashboardStatus.reset();
  renderAll();
  renderDashboardStatus();
}

function renderAll(): void {
  renderRun();
  renderMetrics();
  renderCheckpoints();
  renderLogs();
}

function renderRun(): void {
  element("run-caption", HTMLElement).textContent = selection.runDir;
  const state = summary?.state ?? "-";
  const presence = element("process-presence", HTMLElement);
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
  ]);
  const latestEvent = logEvents.at(-1)?.event;
  const sessionBadge = element("session-presence", HTMLElement);
  sessionBadge.textContent = latestEvent?.event ?? "-";
  sessionBadge.className = sessionBadgeClass(latestEvent?.event ?? "");
  replaceWithRows(element("runtime-details", HTMLElement), [
    ["Run status", state],
    ["Status reason", summary?.reason ?? "-"],
    ["Latest event", latestEvent?.event ?? "-"],
    ["Last observed", formatTime(latestEvent?.recorded_at_ms ?? null)],
    ["Rounds", formatValue(summary?.details?.total_rounds)],
    ["Samples", formatValue(summary?.details?.total_samples)],
    ["Updates", formatValue(summary?.details?.total_updates)],
    ["Checkpoint", summary?.details?.checkpoint_path ?? "-"],
  ]);
  const launchArgv = latestLaunchArgv();
  element("process-command", HTMLElement).textContent =
    (process?.argv ?? launchArgv)?.map(shellQuote).join(" ") ??
      "No training session";
  element("overview", HTMLElement).replaceChildren(
    overviewCell("State", state),
    overviewCell("Rounds", formatValue(summary?.details?.total_rounds)),
    overviewCell(
      "Samples",
      formatValue(summary?.details?.total_samples),
    ),
    overviewCell(
      "Updates",
      formatValue(summary?.details?.total_updates),
    ),
    overviewCell("Latest event", latestEvent?.event ?? "-"),
  );
  element("open-init", HTMLButtonElement).disabled = summary === null ||
    process !== null || stopping || initializing || resuming;
  element("open-resume", HTMLButtonElement).disabled =
    state !== "READY" ||
    stopping || initializing || resuming;
  const stopButton = element("stop-training", HTMLButtonElement);
  stopButton.disabled = process === null || stopping;
  stopButton.textContent = stopping ? "Stopping..." : "Stop";
}

function renderMetrics(): void {
  const totals = metrics?.totals ?? {};
  element("metrics-session", HTMLElement).textContent =
    metrics?.session_id ?? "No training session";
  renderMetricSessions();
  element("metric-strip", HTMLElement).replaceChildren(
    metricCell("Rounds", formatValue(totals.total_rounds)),
    metricCell("Samples", formatValue(totals.total_samples)),
    metricCell("Updates", formatValue(totals.total_updates)),
    metricCell("Samples/s", formatValue(totals.samples_per_second)),
    metricCell("Update time", formatSeconds(totals.update_seconds)),
    metricCell(
      "Log integrity",
      metrics?.complete === false ? "INCOMPLETE" : "COMPLETE",
    ),
  );
  if (metrics !== null) charts.setData(metrics, metricAxis);
  else charts.clear();
}

function latestLaunchArgv(): readonly string[] | null {
  for (let index = logEvents.length - 1; index >= 0; index -= 1) {
    const event = logEvents[index]?.event;
    if (event?.event !== "session.started") continue;
    const argv = event.fields.argv;
    if (
      Array.isArray(argv) &&
      argv.every((item) => typeof item === "string")
    ) {
      return argv;
    }
  }
  return null;
}

function renderMetricSessions(): void {
  const select = element("metrics-session-select", HTMLSelectElement);
  const options: HTMLOptionElement[] = [];
  const latest = document.createElement("option");
  latest.value = "";
  latest.textContent = "Latest session";
  options.push(latest);
  for (const session of metrics?.sessions ?? []) {
    const option = document.createElement("option");
    option.value = session.session_id;
    option.textContent = new Date(session.started_at_ms).toLocaleString(
      "en-GB",
    );
    options.push(option);
  }
  select.replaceChildren(...options);
  select.value = selection.metricSession ?? "";
}

function renderLogs(): void {
  const target = element("log-content", HTMLElement);
  target.replaceChildren(
    ...logEvents.map(({ sequence, event }) => logRow(sequence, event)),
  );
  element("log-count", HTMLElement).textContent = `${
    logEvents.length.toLocaleString("en-US")
  } events`;
  if (followLogs) scrollLogsToEnd();
}

function scheduleLogRender(): void {
  if (logRenderFrame !== null) return;
  logRenderFrame = requestAnimationFrame(() => {
    logRenderFrame = null;
    appendPendingLogs();
  });
}

function appendPendingLogs(): void {
  if (pendingLogEvents.length === 0) return;
  const additions = pendingLogEvents;
  pendingLogEvents = [];
  logEvents = [...logEvents, ...additions].slice(-logWindow);
  renderRun();
  const target = element("log-content", HTMLElement);
  const fragment = document.createDocumentFragment();
  for (const { sequence, event } of additions) {
    fragment.append(logRow(sequence, event));
  }
  target.append(fragment);
  while (target.childElementCount > logWindow) {
    target.firstElementChild?.remove();
  }
  element("log-count", HTMLElement).textContent = `${
    logEvents.length.toLocaleString("en-US")
  } events`;
  if (followLogs) scrollLogsToEnd();
}

function logRow(sequence: number, event: TrainingEvent): HTMLElement {
  const details = document.createElement("details");
  details.className = `log-row level-${event.level.toLowerCase()}`;
  const summaryElement = document.createElement("summary");
  const time = document.createElement("time");
  time.textContent = new Date(event.recorded_at_ms).toLocaleTimeString(
    "en-GB",
  );
  const level = document.createElement("span");
  level.className = "log-level";
  level.textContent = event.level;
  const name = document.createElement("strong");
  name.textContent = event.event;
  const cursor = document.createElement("code");
  cursor.textContent = `#${sequence}`;
  summaryElement.append(time, level, name, cursor);
  const body = document.createElement("pre");
  body.textContent = JSON.stringify(event, null, 2);
  details.append(summaryElement, body);
  return details;
}

function scrollLogsToEnd(): void {
  const target = element("log-content", HTMLElement);
  target.scrollTop = target.scrollHeight;
}

function renderCheckpoints(): void {
  const checkpoints = summary?.checkpoints;
  element("checkpoint-directory", HTMLElement).textContent =
    checkpoints?.checkpoint_directory ??
      `${selection.runDir}/checkpoints`;
  element("checkpoint-summary", HTMLElement).replaceChildren(
    metricCell(
      "Valid manifests",
      String(
        checkpoints?.manifests.filter((item) => item.valid).length ?? 0,
      ),
    ),
    metricCell("Objects", String(checkpoints?.objects.length ?? 0)),
    metricCell(
      "Orphans",
      String(
        checkpoints?.objects.filter((item) => item.orphan).length ?? 0,
      ),
    ),
    metricCell(
      "Unique storage",
      formatBytes(checkpoints?.total_unique_state_bytes ?? 0),
    ),
  );
  element("manifest-rows", HTMLTableSectionElement).replaceChildren(
    ...(checkpoints?.manifests ?? []).map((manifest) => {
      const actions = document.createElement("div");
      actions.className = "table-actions";
      actions.append(
        actionButton("Inspect", () => showCheckpoint(manifest)),
        actionButton(
          "Resume",
          () => resumeFrom(manifest),
          manifest.valid && summary?.state === "READY",
        ),
      );
      return rowWithNode(
        [
          manifest.name,
          formatValue(manifest.total_updates),
          formatValue(manifest.total_samples),
          manifest.state_exists ? "Available" : "Missing",
          formatBytes(manifest.state_size_bytes),
          manifest.error ?? "Validated",
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
  if (initializing || !initForm.reportValidity()) return;
  const request = initRequestFromForm(initForm, selection.runDir, null);
  if (summary?.state === "READY" || summary?.state === "BROKEN") {
    pendingInitRequest = request;
    element("replace-status", HTMLElement).textContent =
      `Current state: ${summary.state}`;
    element("replace-run-directory", HTMLElement).textContent =
      selection.runDir;
    element("replace-result", HTMLElement).textContent = "";
    initDialog.close();
    replaceDialog.showModal();
    element("replace-existing", HTMLInputElement).focus();
    return;
  }
  void runInitialization(
    request,
    initDialog,
    "init-status",
    "confirm-init",
    "Initialize",
  );
}

function confirmReplacement(event: Event): void {
  event.preventDefault();
  if (initializing || !replaceForm.reportValidity()) return;
  if (pendingInitRequest === null) {
    throw new Error("Missing initialization request");
  }
  void runInitialization(
    { ...pendingInitRequest, replace_existing: "yes" },
    replaceDialog,
    "replace-result",
    "confirm-replace",
    "Replace and initialize",
  );
}

async function runInitialization(
  request: InitRequest,
  dialog: HTMLDialogElement,
  statusId: string,
  buttonId: string,
  idleLabel: string,
): Promise<void> {
  const origin = selection.captureRun();
  if (request.run_dir !== origin.runDir) return;
  initializing = true;
  const status = element(statusId, HTMLElement);
  const button = element(buttonId, HTMLButtonElement);
  button.disabled = true;
  status.textContent = "Creating initial checkpoint...";
  renderRun();
  try {
    await initializeTraining(request);
    dialog.close();
    if (!selection.ownsRun(origin)) return;
    selection.markRunReplaced();
    metrics = null;
    connectStream();
    await refreshAll();
  } catch (error: unknown) {
    if (selection.ownsRun(origin)) {
      status.className = "error-value";
      status.textContent = errorText(error);
    }
  } finally {
    initializing = false;
    button.disabled = false;
    button.textContent = idleLabel;
    renderRun();
  }
}

async function resume(event: Event): Promise<void> {
  event.preventDefault();
  if (resuming || !resumeForm.reportValidity()) return;
  resuming = true;
  const status = element("resume-status", HTMLElement);
  const origin = selection.captureRun();
  const request = resumeRequestFromForm(resumeForm, origin.runDir);
  try {
    const nextProcess = await resumeTraining(request);
    if (!selection.ownsRun(origin)) return;
    process = nextProcess;
    status.textContent = "";
    resumeDialog.close();
    connectStream();
    await refreshSummaryOnly();
  } catch (error: unknown) {
    if (selection.ownsRun(origin)) {
      status.className = "error-value";
      status.textContent = errorText(error);
    }
  } finally {
    resuming = false;
    renderRun();
  }
}

async function stop(): Promise<void> {
  const origin = selection.captureRun();
  stopping = true;
  renderRun();
  try {
    const forced = await stopTraining(origin.runDir);
    if (!selection.ownsRun(origin)) return;
    clearDashboardError("control");
    await refreshAll();
    if (!selection.ownsRun(origin)) return;
    if (forced) {
      showWarning(
        "Stop timeout exceeded; process group was killed",
      );
    }
  } catch (error: unknown) {
    if (selection.ownsRun(origin)) {
      reportDashboardError("control", errorText(error));
    }
  } finally {
    stopping = false;
    renderRun();
  }
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
      initCommandPreview(
        initRequestFromForm(initForm, selection.runDir, null),
      );
  } catch (error: unknown) {
    element("init-command-preview", HTMLElement).textContent =
      errorText(error);
  }
}

function renderResumeCommand(): void {
  try {
    element("resume-command-preview", HTMLElement).textContent =
      resumeCommandPreview(
        resumeRequestFromForm(resumeForm, selection.runDir),
      );
  } catch (error: unknown) {
    element("resume-command-preview", HTMLElement).textContent =
      errorText(error);
  }
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

function renderRoute(): void {
  const route = currentRoute();
  for (
    const view of document.querySelectorAll<HTMLElement>("[data-view]")
  ) view.hidden = view.dataset.view !== route;
  for (
    const link of document.querySelectorAll<HTMLElement>("[data-route]")
  ) link.classList.toggle("active", link.dataset.route === route);
  if (route === "metrics") {
    charts.resize();
    void refreshMetrics();
  }
}

function currentRoute(): ViewName {
  const route = globalThis.location.hash.slice(1);
  if (
    route === "metrics" || route === "logs" || route === "checkpoints"
  ) return route;
  return "overview";
}

function renderDirectoryAction(): void {
  useDirectoryButton.disabled =
    directoryInput.value.trim() === selection.runDir;
}

function renderReplacementAction(): void {
  element("confirm-replace", HTMLButtonElement).disabled =
    initializing ||
    element("replace-existing", HTMLInputElement).value !== "yes";
}

function resetReplacementConfirmation(): void {
  pendingInitRequest = null;
  element("replace-existing", HTMLInputElement).value = "";
  renderReplacementAction();
}

function renderAxisButtons(): void {
  for (
    const button of document.querySelectorAll<HTMLButtonElement>(
      "[data-axis]",
    )
  ) {
    button.classList.toggle(
      "active",
      button.dataset.axis === metricAxis,
    );
  }
}

function updateLogWindow(): void {
  const input = element("log-window", HTMLInputElement);
  const value = input.valueAsNumber;
  if (
    !Number.isFinite(value) || !Number.isInteger(value) || value <= 0
  ) {
    input.setCustomValidity("Window must be a positive integer");
    input.reportValidity();
    return;
  }
  input.setCustomValidity("");
  logWindow = value;
  connectStream();
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

function sessionBadgeClass(eventType: string): string {
  if (eventType.endsWith("failed")) return "badge danger";
  if (
    eventType === "session.completed" || eventType === "session.stopped"
  ) return "badge success";
  return eventType === "" ? "badge neutral" : "badge running";
}

function setConnection(
  label: string,
  state: "online" | "offline" | "pending",
): void {
  const target = element("connection-state", HTMLElement);
  target.textContent = label;
  target.className = state === "pending"
    ? "connection-label"
    : `connection-label ${state}`;
}

function connectStream(): void {
  dashboardStatus.setStreamConnection("connecting");
  renderDashboardStatus();
  stream.connect();
}

function reportDashboardError(
  source: DashboardErrorSource,
  message: string,
): void {
  dashboardStatus.reportError(source, message);
  renderDashboardStatus();
}

function clearDashboardError(source: DashboardErrorSource): void {
  dashboardStatus.clearError(source);
  renderDashboardStatus();
}

function renderDashboardStatus(): void {
  const status = dashboardStatus.snapshot();
  element("directory-error", HTMLElement).textContent = status.message;
  setConnection(status.label, status.tone);
}

function showWarning(message: string): void {
  dashboardStatus.reportWarning(message);
  renderDashboardStatus();
}

function formatValue(value: JsonPrimitive | undefined): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "-";
    return value.toLocaleString("en-US", { maximumFractionDigits: 5 });
  }
  return String(value);
}

function formatBytes(value: number | null): string {
  if (value === null) return "-";
  if (value < 1024) return `${value} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let output = value / 1024;
  let unit = units[0] ?? "KiB";
  for (
    let index = 1;
    index < units.length && output >= 1024;
    index += 1
  ) {
    output /= 1024;
    unit = units[index] ?? unit;
  }
  return `${output.toFixed(output >= 10 ? 1 : 2)} ${unit}`;
}

function formatSeconds(value: JsonPrimitive | undefined): string {
  return typeof value === "number" ? `${formatValue(value)} s` : "-";
}

function formatTime(value: number | null): string {
  return value === null ? "-" : new Date(value).toLocaleString("en-GB");
}

function shellQuote(value: string): string {
  return /^[A-Za-z0-9_./,:+=-]+$/.test(value)
    ? value
    : `'${value.replaceAll("'", `'"'"'`)}'`;
}

function selectedNumber(id: string): number {
  const value = Number(element(id, HTMLSelectElement).value);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`Invalid selection: ${id}`);
  }
  return value;
}

function metricRefreshOptions(): {
  readonly updateLimit: number;
  readonly seriesPoints: number;
} {
  return {
    updateLimit: selectedNumber("metrics-range"),
    seriesPoints: selectedNumber("metrics-resolution"),
  };
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
