import {
  fetchConfig,
  initializeTraining,
  resumeTraining,
  stopTraining,
} from "./api.ts";
import { CheckpointsDomain } from "./checkpoints-domain.ts";
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
import { LogsDomain } from "./logs-domain.ts";
import { MetricsDomain } from "./metrics-domain.ts";
import { ProcessDomain } from "./process-domain.ts";
import { DashboardSelection } from "./selection.ts";
import {
  type DashboardErrorSource,
  DashboardStatus,
} from "./status.ts";
import type { CheckpointManifest } from "./types.ts";

type ViewName = "process" | "metrics" | "logs" | "checkpoints";

const selection = new DashboardSelection();
const dashboardStatus = new DashboardStatus();
let initializing = false;
let resuming = false;
let stopping = false;
let pendingInitRequest: InitRequest | null = null;

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

const processDomain = new ProcessDomain(
  () => selection.runDir,
  {
    reportError: (message) => reportDashboardError("process", message),
    clearError: () => clearDashboardError("process"),
    connectionChanged: (connected) => {
      dashboardStatus.setStreamConnection(
        connected ? "online" : "reconnecting",
      );
      renderDashboardStatus();
    },
  },
);

const metricsDomain = new MetricsDomain(
  () => selection.runDir,
  () => currentRoute() === "metrics",
  {
    reportError: (message) => reportDashboardError("metrics", message),
    clearError: () => clearDashboardError("metrics"),
    setPending: (pending) => {
      if (pending) dashboardStatus.setRefreshPending();
      else dashboardStatus.setRefreshIdle();
      renderDashboardStatus();
    },
  },
);

const logsDomain = new LogsDomain(
  () => selection.runDir,
  () => currentRoute() === "logs",
  {
    reportError: (message) => reportDashboardError("logs", message),
    clearError: () => clearDashboardError("logs"),
  },
);

const checkpointsDomain = new CheckpointsDomain(
  () => selection.runDir,
  () => currentRoute() === "checkpoints",
  {
    reportError: (message) =>
      reportDashboardError("checkpoints", message),
    clearError: () => clearDashboardError("checkpoints"),
    canResume: () =>
      processDomain.process === null && !initializing && !resuming &&
      !stopping,
    resumeFrom,
    inspect: showCheckpoint,
  },
);

initialize();

function initialize(): void {
  renderFormFields("init-fields", INIT_GROUPS, INIT_FIELDS);
  renderFormFields("resume-fields", RESUME_GROUPS, RESUME_FIELDS);
  bindEvents();
  renderRoute();
  processDomain.render();
  metricsDomain.render();
  logsDomain.render();
  checkpointsDomain.render();
  void loadServerConfig();
}

function bindEvents(): void {
  globalThis.addEventListener("hashchange", renderRoute);
  directoryInput.addEventListener("input", renderDirectoryAction);
  directoryForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const runDir = directoryInput.value.trim();
    if (runDir === "") {
      reportDashboardError("directory", "Run directory is required");
      return;
    }
    clearDashboardError("directory");
    selection.setRunDirectory(runDir);
    resetDomains();
    connectStreams();
    renderDirectoryAction();
    void refreshCurrentDomain();
  });
  element("open-init", HTMLButtonElement).addEventListener(
    "click",
    () => {
      resetLaunchStatus("init-status");
      renderInitCommand();
      initDialog.showModal();
    },
  );
  element("open-replace", HTMLButtonElement).addEventListener(
    "click",
    prepareReplacement,
  );
  element("open-resume", HTMLButtonElement).addEventListener(
    "click",
    () => {
      resetLaunchStatus("resume-status");
      renderResumeCommand();
      resumeDialog.showModal();
    },
  );
  initForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!initForm.reportValidity()) return;
    void runInitialization(
      initRequestFromForm(initForm, selection.runDir, null),
      initDialog,
      "init-status",
      "confirm-init",
      "Initialize",
    );
  });
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
      "[data-refresh-domain]",
    )
  ) {
    button.addEventListener("click", () => {
      if (button.dataset.refreshDomain === currentRoute()) {
        void refreshCurrentDomain();
      }
    });
  }
}

async function loadServerConfig(): Promise<void> {
  const origin = selection.captureRun();
  try {
    const config = await fetchConfig();
    if (!selection.ownsRun(origin)) return;
    selection.setRunDirectory(config.default_run_dir);
    directoryInput.value = selection.runDir;
    renderDirectoryAction();
    resetDomains();
    connectStreams();
    await refreshCurrentDomain();
  } catch (error: unknown) {
    if (selection.ownsRun(origin)) {
      reportDashboardError("config", errorText(error));
    }
  }
}

function resetDomains(): void {
  processDomain.reset();
  metricsDomain.reset();
  logsDomain.reset();
  checkpointsDomain.reset();
  dashboardStatus.reset();
  renderDashboardStatus();
}

function resetArtifactDomains(): void {
  metricsDomain.reset();
  logsDomain.reset();
  checkpointsDomain.reset();
}

function connectStreams(): void {
  dashboardStatus.setStreamConnection("connecting");
  renderDashboardStatus();
  processDomain.connect();
  metricsDomain.connect();
  checkpointsDomain.connect();
  if (currentRoute() === "logs") logsDomain.activate();
}

async function refreshCurrentDomain(): Promise<void> {
  switch (currentRoute()) {
    case "process":
      await processDomain.refresh();
      return;
    case "metrics":
      await metricsDomain.refresh();
      return;
    case "logs":
      await logsDomain.refresh();
      return;
    case "checkpoints":
      await checkpointsDomain.refresh();
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
  ) link.classList.toggle("active", link.dataset.route === route);
  if (route === "logs") logsDomain.activate();
  else logsDomain.deactivate();
  if (route === "metrics") metricsDomain.activate();
  if (route === "checkpoints" && !checkpointsDomain.loaded) {
    void checkpointsDomain.refresh();
  }
  if (route === "process" && selection.runDir !== "") {
    void processDomain.refresh();
  }
}

function currentRoute(): ViewName {
  const route = globalThis.location.hash.slice(1);
  if (
    route === "metrics" || route === "logs" || route === "checkpoints"
  ) return route;
  return "process";
}

function syncProcessOperations(): void {
  processDomain.setOperations({ initializing, resuming, stopping });
}

function prepareReplacement(): void {
  if (!initForm.reportValidity()) return;
  pendingInitRequest = initRequestFromForm(
    initForm,
    selection.runDir,
    null,
  );
  element("replace-run-directory", HTMLElement).textContent =
    selection.runDir;
  element("replace-result", HTMLElement).textContent = "";
  initDialog.close();
  replaceDialog.showModal();
  element("replace-existing", HTMLInputElement).focus();
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
  if (initializing) return;
  const origin = selection.captureRun();
  initializing = true;
  syncProcessOperations();
  const status = element(statusId, HTMLElement);
  const button = element(buttonId, HTMLButtonElement);
  button.disabled = true;
  status.className = "status-value";
  status.textContent = "Initializing…";
  try {
    await initializeTraining(request);
    if (!selection.ownsRun(origin)) return;
    dialog.close();
    selection.markRunReplaced();
    resetArtifactDomains();
    await processDomain.refresh();
  } catch (error: unknown) {
    if (selection.ownsRun(origin)) {
      status.className = "error-value";
      status.textContent = errorText(error);
    }
  } finally {
    initializing = false;
    button.disabled = false;
    button.textContent = idleLabel;
    syncProcessOperations();
  }
}

async function resume(event: Event): Promise<void> {
  event.preventDefault();
  if (resuming || !resumeForm.reportValidity()) return;
  const origin = selection.captureRun();
  const status = element("resume-status", HTMLElement);
  const button = element("confirm-resume", HTMLButtonElement);
  resuming = true;
  syncProcessOperations();
  button.disabled = true;
  button.textContent = "Starting…";
  status.className = "status-value";
  status.textContent = "Waiting for CLI readiness…";
  try {
    const value = await resumeTraining(
      resumeRequestFromForm(resumeForm, origin.runDir),
    );
    if (!selection.ownsRun(origin)) return;
    processDomain.apply(value);
    resumeDialog.close();
  } catch (error: unknown) {
    if (selection.ownsRun(origin)) {
      status.className = "error-value";
      status.textContent = errorText(error);
    }
  } finally {
    resuming = false;
    button.disabled = false;
    button.textContent = "Resume";
    syncProcessOperations();
  }
}

async function stop(): Promise<void> {
  const origin = selection.captureRun();
  stopping = true;
  syncProcessOperations();
  try {
    const value = await stopTraining(origin.runDir);
    if (!selection.ownsRun(origin)) return;
    processDomain.apply(value);
    clearDashboardError("control");
    if (value.forced) {
      dashboardStatus.reportWarning(
        "Stop timeout exceeded; the process group was killed",
      );
    }
  } catch (error: unknown) {
    if (selection.ownsRun(origin)) {
      reportDashboardError("control", errorText(error));
    }
  } finally {
    stopping = false;
    syncProcessOperations();
    renderDashboardStatus();
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
  caption.className = "field-caption";
  const fieldLabel = document.createElement("span");
  fieldLabel.className = "field-label";
  fieldLabel.textContent = field.label;
  const flag = document.createElement("code");
  flag.textContent = field.flag;
  caption.append(fieldLabel, flag);
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
  renderCommandPreview("init-command-preview", () =>
    initCommandPreview(
      initRequestFromForm(initForm, selection.runDir, null),
    ));
}

function renderResumeCommand(): void {
  renderCommandPreview(
    "resume-command-preview",
    () =>
      resumeCommandPreview(
        resumeRequestFromForm(resumeForm, selection.runDir),
      ),
  );
}

function renderCommandPreview(id: string, build: () => string): void {
  try {
    element(id, HTMLElement).textContent = build();
  } catch (error: unknown) {
    element(id, HTMLElement).textContent = errorText(error);
  }
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

function resetLaunchStatus(statusId: string): void {
  const status = element(statusId, HTMLElement);
  status.className = "status-value";
  status.textContent = "";
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
  const connection = element("connection-state", HTMLElement);
  connection.textContent = status.label;
  connection.className = status.tone === "pending"
    ? "connection-label"
    : `connection-label ${status.tone}`;
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
