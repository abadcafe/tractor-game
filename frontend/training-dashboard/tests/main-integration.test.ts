import { assert, assertEquals } from "./stubs/assert/mod.ts";

interface FakeResponse {
  readonly status: number;
  readonly ok: boolean;
  json(): Promise<unknown>;
}

interface FetchCall {
  readonly method: string;
  readonly path: string;
}

interface DashboardHarness {
  readonly openInitButton: HTMLButtonElement;
  readonly openResumeButton: HTMLButtonElement;
  readonly initForm: HTMLFormElement;
  readonly resumeForm: HTMLFormElement;
  readonly confirmInitButton: HTMLButtonElement;
  readonly confirmResumeButton: HTMLButtonElement;
  readonly fetchCalls: FetchCall[];
  readonly cleanup: () => Promise<void>;
}

type Listener = (event: Event) => void;

class FakeDataset {
  readonly #values = new Map<string, string>();

  set(name: string, value: string): void {
    this.#values.set(name, value);
    Object.defineProperty(this, name, {
      configurable: true,
      enumerable: true,
      writable: true,
      value,
    });
  }

  has(name: string): boolean {
    return this.#values.has(name);
  }

  get(name: string): string | undefined {
    return this.#values.get(name);
  }
}

class FakeClassList {
  #tokens = new Set<string>();
  constructor(private readonly host: FakeElement) {}
  add(...tokens: readonly string[]): void {
    for (const token of tokens) {
      if (token.trim() !== "") this.#tokens.add(token);
    }
    this.host._syncClassName(this.#tokens);
  }
  remove(...tokens: readonly string[]): void {
    for (const token of tokens) this.#tokens.delete(token);
    this.host._syncClassName(this.#tokens);
  }
  toggle(token: string, force?: boolean): boolean {
    if (force === false) {
      this.remove(token);
      return false;
    }
    if (force === true || !this.#tokens.has(token)) {
      this.add(token);
      return true;
    }
    this.remove(token);
    return false;
  }
  contains(token: string): boolean {
    return this.#tokens.has(token);
  }
}

class FakeElement extends EventTarget {
  readonly tagName: string;
  #id = "";
  #parent: FakeElement | null = null;
  #children: FakeElement[] = [];
  #listeners = new Map<string, Set<Listener>>();
  #document: FakeDocument | null = null;
  #scrollHeight = 0;

  textContent = "";
  className = "";
  hidden = false;
  scrollTop = 0;
  value = "";
  required = false;
  checked = false;
  disabled = false;
  name = "";
  type = "";
  min = "";
  max = "";
  step = "";
  open = false;

  readonly dataset = new FakeDataset();
  readonly attributes = new Map<string, string>();
  readonly style = new Map<string, string>();
  readonly classList: FakeClassList;

  constructor(tag: string, owner?: FakeDocument) {
    super();
    this.tagName = tag.toUpperCase();
    this.classList = new FakeClassList(this);
    if (owner !== undefined) this.#document = owner;
  }

  override addEventListener(type: string, listener: Listener): void {
    const bucket = this.#listeners.get(type);
    if (bucket === undefined) {
      this.#listeners.set(type, new Set([listener]));
    } else bucket.add(listener);
  }

  override dispatchEvent(event: Event): boolean {
    const bucket = this.#listeners.get(event.type);
    if (bucket !== undefined) {
      for (const listener of bucket) listener(event);
    }
    return true;
  }

  _syncClassName(tokens: Set<string>): void {
    this.className = [...tokens].join(" ");
  }

  get id(): string {
    return this.#id;
  }
  set id(next: string) {
    const old = this.#id;
    this.#id = next;
    if (old === next) return;
    this.#document?._registerElement(this);
  }

  get parentElement(): FakeElement | null {
    return this.#parent;
  }

  get children(): readonly FakeElement[] {
    return this.#children;
  }

  get childElementCount(): number {
    return this.#children.length;
  }

  get firstElementChild(): FakeElement | null {
    return this.#children.at(0) ?? null;
  }

  get lastElementChild(): FakeElement | null {
    return this.#children.at(-1) ?? null;
  }

  get ownerDocument(): FakeDocument | null {
    if (this.#parent === null && this === this.#document?.body) {
      return this.#document;
    }
    return this.#parent?.ownerDocument ?? this.#document;
  }

  get valueAsNumber(): number {
    const parsed = Number(this.value);
    return Number.isFinite(parsed) ? parsed : NaN;
  }
  set valueAsNumber(next: number) {
    this.value = String(next);
  }

  setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
    if (name.startsWith("data-")) {
      this.dataset.set(
        dataPropertyName(name.slice("data-".length)),
        value,
      );
    }
    if (name === "id") this.id = value;
    if (name === "class") this.className = value;
    if (name === "name") this.name = value;
    if (name === "type") this.type = value;
  }

  getAttribute(name: string): string | null {
    return this.attributes.get(name) ?? null;
  }

  append(...nodes: readonly FakeElement[]): void {
    for (const node of nodes) {
      node.#parent = this;
      node._setDocument(this.ownerDocument);
      this.#children.push(node);
      this.ownerDocument?._registerSubtree(node);
    }
  }

  appendChild(node: FakeElement): void {
    this.append(node);
  }

  removeChild(node: FakeElement): void {
    this.#children = this.#children.filter((candidate) =>
      candidate !== node
    );
    node.#parent = null;
  }

  replaceChildren(...nodes: readonly FakeElement[]): void {
    for (const child of this.#children) child.#parent = null;
    this.#children = [];
    this.append(...nodes);
  }

  remove(): void {
    this.#parent?.removeChild(this);
    this.#parent = null;
  }

  closest(selector: string): FakeElement | null {
    if (selector !== "dialog") return null;
    if (this.tagName === "DIALOG") return this;
    return this.#parent?.closest(selector) ?? null;
  }

  reportValidity(): boolean {
    return true;
  }

  setCustomValidity(_message: string): void {}

  querySelectorAll(selector: string): FakeElement[] {
    if (selector === "*") return this._descendants();
    if (selector.startsWith("#")) {
      const target = this.ownerDocument?.getElementById(
        selector.slice(1),
      );
      return target === null || target === undefined ? [] : [target];
    }
    return (this.ownerDocument ?? this).querySelectorAll(
      selector,
      this,
    );
  }

  _descendants(): FakeElement[] {
    const output: FakeElement[] = [];
    const stack = [...this.#children];
    while (stack.length > 0) {
      const next = stack.shift();
      if (next === undefined) break;
      output.push(next);
      stack.push(...next.children);
    }
    return output;
  }

  showModal(): void {
    this.open = true;
  }

  close(): void {
    this.open = false;
    this.dispatchEvent(new Event("close"));
  }

  focus(): void {
    // no-op in fake DOM
  }

  get scrollHeight(): number {
    return Math.max(
      0,
      this._descendants().length * 16,
      this.#scrollHeight,
    );
  }

  get clientHeight(): number {
    return 0;
  }

  _setDocument(document: FakeDocument | null): void {
    this.#document = document;
    for (const child of this.#children) child._setDocument(document);
  }
}

class FakeFormControls {
  constructor(private readonly form: FakeFormElement) {}
  namedItem(key: string): FakeInputElement | FakeSelectElement | null {
    for (const node of this.form._descendants()) {
      if (
        node instanceof FakeInputElement ||
        node instanceof FakeSelectElement
      ) {
        if (node.name === key) return node;
      }
    }
    return null;
  }
}

class FakeInputElement extends FakeElement {
  constructor(document?: FakeDocument) {
    super("input", document);
    this.type = "text";
  }
}

class FakeSelectElement extends FakeElement {
  constructor(document?: FakeDocument) {
    super("select", document);
  }
}

class FakeButtonElement extends FakeElement {
  constructor(document?: FakeDocument) {
    super("button", document);
    this.type = "button";
  }
}

class FakeFormElement extends FakeElement {
  readonly elements = new FakeFormControls(this);
  constructor(document?: FakeDocument) {
    super("form", document);
  }
}

class FakeDialogElement extends FakeElement {
  constructor(document?: FakeDocument) {
    super("dialog", document);
  }
}

class FakeTableSectionElement extends FakeElement {
  constructor(document?: FakeDocument) {
    super("tbody", document);
  }
}

class FakeDocument extends EventTarget {
  readonly body: FakeElement;
  readonly hidden = false;
  #byId = new Map<string, FakeElement>();

  constructor() {
    super();
    this.body = new FakeElement("body", this);
    this._registerSubtree(this.body);
  }

  createElement(tag: string): FakeElement {
    if (tag === "input") return new FakeInputElement(this);
    if (tag === "select") return new FakeSelectElement(this);
    if (tag === "button") return new FakeButtonElement(this);
    if (tag === "form") return new FakeFormElement(this);
    if (tag === "dialog") return new FakeDialogElement(this);
    if (tag === "tbody") return new FakeTableSectionElement(this);
    return new FakeElement(tag, this);
  }

  createDocumentFragment(): FakeElement {
    return new FakeElement("document-fragment", this);
  }

  getElementById(id: string): FakeElement | null {
    return this.#byId.get(id) ?? null;
  }

  querySelectorAll(
    selector: string,
    root: FakeElement | null = null,
  ): FakeElement[] {
    const roots = root === null ? [this.body] : [root];
    const all = this._collect(roots);
    if (selector.startsWith("#")) {
      const found = this.getElementById(selector.slice(1));
      return found === null ? [] : [found];
    }
    if (selector === "[data-route]") {
      return all.filter((node) => node.dataset.has("route"));
    }
    if (selector === "[data-view]") {
      return all.filter((node) => node.dataset.has("view"));
    }
    if (selector === "[data-refresh-domain]") {
      return all.filter((node) => node.dataset.has("refreshDomain"));
    }
    if (selector === "[data-axis]") {
      return all.filter((node) => node.dataset.has("axis"));
    }
    if (selector === "[data-close-dialog]") {
      return all.filter((node) => node.dataset.has("closeDialog"));
    }
    return [];
  }

  _registerElement(element: FakeElement): void {
    const id = element.id;
    if (id !== "") this.#byId.set(id, element);
  }

  _registerSubtree(element: FakeElement): void {
    for (const node of this._collect([element])) {
      this._registerElement(node);
    }
  }

  _collect(roots: readonly FakeElement[]): FakeElement[] {
    const output: FakeElement[] = [];
    const stack = [...roots];
    while (stack.length > 0) {
      const next = stack.shift();
      if (next === undefined) continue;
      output.push(next);
      stack.push(...next.children);
    }
    return output;
  }
}

function defineGlobal<T>(name: string, value: T): void {
  Object.defineProperty(globalThis, name, {
    configurable: true,
    writable: true,
    value,
  });
}

function jsonResponse(payload: unknown): FakeResponse {
  return {
    status: 200,
    ok: true,
    json(): Promise<unknown> {
      return Promise.resolve(payload);
    },
  };
}

function noContentResponse(): FakeResponse {
  return {
    status: 204,
    ok: true,
    json(): Promise<unknown> {
      return Promise.resolve(null);
    },
  };
}

function waitFrame(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(() => resolve(), 0);
  });
}

async function waitUntil(
  condition: () => boolean,
  message: string,
): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (condition()) return;
    await waitFrame();
  }
  assert(condition(), message);
}

function withFakeDashboardDom(): DashboardHarness {
  const document = new FakeDocument();
  const fetchCalls: FetchCall[] = [];
  const openEventSources: Array<{ close: () => void }> = [];

  defineGlobal("Event", Event);
  defineGlobal("document", document as unknown as Document);
  defineGlobal(
    "HTMLInputElement",
    FakeInputElement as unknown as typeof HTMLInputElement,
  );
  defineGlobal(
    "HTMLSelectElement",
    FakeSelectElement as unknown as typeof HTMLSelectElement,
  );
  defineGlobal(
    "HTMLButtonElement",
    FakeButtonElement as unknown as typeof HTMLButtonElement,
  );
  defineGlobal(
    "HTMLFormElement",
    FakeFormElement as unknown as typeof HTMLFormElement,
  );
  defineGlobal(
    "HTMLDialogElement",
    FakeDialogElement as unknown as typeof HTMLDialogElement,
  );
  defineGlobal(
    "HTMLTableSectionElement",
    FakeTableSectionElement as unknown as typeof HTMLTableSectionElement,
  );
  defineGlobal(
    "HTMLDivElement",
    FakeElement as unknown as typeof HTMLDivElement,
  );
  defineGlobal(
    "HTMLElement",
    FakeElement as unknown as typeof HTMLElement,
  );
  defineGlobal(
    "HTMLSpanElement",
    FakeElement as unknown as typeof HTMLSpanElement,
  );
  defineGlobal(
    "HTMLTimeElement",
    FakeElement as unknown as typeof HTMLTimeElement,
  );
  defineGlobal(
    "HTMLPreElement",
    FakeElement as unknown as typeof HTMLPreElement,
  );

  defineGlobal(
    "location",
    {
      protocol: "https:",
      host: "localhost",
      hash: "",
      addEventListener() {},
      removeEventListener() {},
    } as unknown as Location,
  );

  defineGlobal(
    "requestAnimationFrame",
    ((callback: FrameRequestCallback): number => {
      return setTimeout(() => callback(0), 0) as unknown as number;
    }) as typeof requestAnimationFrame,
  );
  defineGlobal(
    "cancelAnimationFrame",
    ((id: number): void => {
      clearTimeout(id);
    }) as typeof cancelAnimationFrame,
  );

  defineGlobal(
    "ResizeObserver",
    class {
      constructor(_callback: ResizeObserverCallback) {}
      observe(): void {}
      unobserve(): void {}
      disconnect(): void {}
    } as unknown as typeof ResizeObserver,
  );

  class FakeEventSource {
    #listeners = new Map<string, Set<(event: Event) => void>>();
    #closed = false;
    readonly CONNECTING = 0;
    readonly OPEN = 1;
    readonly CLOSED = 2;
    readyState = this.CONNECTING;
    withCredentials = false;
    onopen: ((this: EventSource, event: Event) => void) | null = null;
    onmessage:
      | ((this: EventSource, event: MessageEvent) => void)
      | null = null;
    onerror: ((this: EventSource, event: Event) => void) | null = null;
    constructor(readonly url: string) {
      openEventSources.push(this);
      setTimeout(() => this.#dispatch("open", new Event("open")), 0);
    }
    addEventListener(
      event: string,
      listener: (event: Event) => void,
    ): void {
      const next = this.#listeners.get(event);
      if (next === undefined) {
        this.#listeners.set(event, new Set([listener]));
      } else next.add(listener);
    }
    close(): void {
      if (this.#closed) return;
      this.#closed = true;
      this.readyState = this.CLOSED;
    }
    #dispatch(type: string, event: Event): void {
      if (type === "open") this.readyState = this.OPEN;
      const handlers = this.#listeners.get(type);
      if (handlers === undefined) return;
      for (const handler of handlers) handler(event);
    }
  }

  defineGlobal(
    "EventSource",
    FakeEventSource as unknown as typeof EventSource,
  );

  const create = (id: string, tag: string = "div"): FakeElement => {
    const element = document.createElement(tag);
    element.id = id;
    return element;
  };

  const createOption = (): FakeSelectElement =>
    document.createElement("option") as FakeSelectElement;

  const directoryForm = create(
    "directory-form",
    "form",
  ) as FakeFormElement;
  const runDirectory = create(
    "run-directory",
    "input",
  ) as FakeInputElement;
  const useRunDirectory = create(
    "use-run-directory",
    "button",
  ) as FakeButtonElement;

  const openInit = create("open-init", "button") as FakeButtonElement;
  const openResume = create(
    "open-resume",
    "button",
  ) as FakeButtonElement;
  const stopTraining = create(
    "stop-training",
    "button",
  ) as FakeButtonElement;

  const initDialog = create("init-dialog", "dialog");
  const initForm = create("init-form", "form") as FakeFormElement;
  const initFields = create("init-fields");
  const initStatus = create("init-status");
  const initCommandPreview = create("init-command-preview");
  const confirmInit = create(
    "confirm-init",
    "button",
  ) as FakeButtonElement;
  const closeInit = create("close-init", "button");
  closeInit.setAttribute("data-close-dialog", "");
  const replaceDialog = create("replace-dialog", "dialog");
  const replaceForm = create("replace-form", "form") as FakeFormElement;
  const replaceStatus = create("replace-status");
  const replaceRunDirectory = create("replace-run-directory");
  const replaceExisting = create(
    "replace-existing",
    "input",
  ) as FakeInputElement;
  const replaceResult = create("replace-result");
  const confirmReplace = create(
    "confirm-replace",
    "button",
  ) as FakeButtonElement;
  const closeReplace = create("close-replace", "button");
  closeReplace.setAttribute("data-close-dialog", "");

  const resumeDialog = create("resume-dialog", "dialog");
  const resumeForm = create("resume-form", "form") as FakeFormElement;
  const resumeFields = create("resume-fields");
  const resumeCommandPreview = create("resume-command-preview");
  const confirmResume = create(
    "confirm-resume",
    "button",
  ) as FakeButtonElement;
  const resumeStatus = create("resume-status");
  const closeResume = create("close-resume", "button");
  closeResume.setAttribute("data-close-dialog", "");

  const checkpointDialog = create("checkpoint-dialog", "dialog");
  const closeCheckpoint = create("close-checkpoint", "button");
  const checkpointDialogTitle = create("checkpoint-dialog-title");
  const checkpointDialogContent = create("checkpoint-dialog-content");

  const runCaption = create("run-caption");
  const processPresence = create("process-presence");
  const processDetails = create("process-details");
  const processCommand = create("process-command");
  const processConnectionState = create("process-connection-state");
  const processError = create("process-error");
  const metricsRange = create(
    "metrics-range",
    "select",
  ) as FakeSelectElement;
  const metricsResolution = create(
    "metrics-resolution",
    "select",
  ) as FakeSelectElement;
  const metricStrip = create("metric-strip");
  const logCount = create("log-count");
  const logContent = create("log-content");
  const loadOlder = create(
    "load-older",
    "button",
  ) as FakeButtonElement;
  const toggleFollow = create(
    "toggle-follow",
    "button",
  ) as FakeButtonElement;
  const checkpointDirectory = create("checkpoint-directory");
  const checkpointSummary = create("checkpoint-summary");
  const manifestRows = create(
    "manifest-rows",
    "tbody",
  ) as FakeTableSectionElement;
  const objectRows = create(
    "object-rows",
    "tbody",
  ) as FakeTableSectionElement;
  const directoryError = create("directory-error");
  const metricsError = create("metrics-error");
  const logsError = create("logs-error");
  const checkpointsError = create("checkpoints-error");

  for (
    const chartId of [
      "chart-throughput",
      "chart-loss",
      "chart-policy",
      "chart-ppo-timing",
      "chart-rollout",
      "chart-rewards",
      "chart-inference",
      "chart-processes",
    ]
  ) {
    create(chartId);
  }

  const axisUpdate = document.createElement(
    "button",
  ) as FakeButtonElement;
  axisUpdate.setAttribute("data-axis", "update");
  const axisElapsed = document.createElement(
    "button",
  ) as FakeButtonElement;
  axisElapsed.setAttribute("data-axis", "elapsed");

  const refreshButtons: FakeButtonElement[] = [];
  for (const domain of ["metrics", "logs", "checkpoints"]) {
    const refresh = document.createElement(
      "button",
    ) as FakeButtonElement;
    refresh.setAttribute("data-refresh-domain", domain);
    refreshButtons.push(refresh);
  }

  for (const route of ["process", "metrics", "logs", "checkpoints"]) {
    const link = document.createElement("a");
    link.setAttribute("data-route", route);
    link.textContent = route;
    document.body.append(link);
  }
  for (const view of ["process", "metrics", "logs", "checkpoints"]) {
    const section = document.createElement("section");
    section.setAttribute("data-view", view);
    document.body.append(section);
  }

  initForm.append(
    initFields,
    initCommandPreview,
    confirmInit,
    initStatus,
    closeInit,
  );
  resumeForm.append(
    resumeFields,
    resumeCommandPreview,
    confirmResume,
    resumeStatus,
    closeResume,
  );
  replaceForm.append(
    replaceStatus,
    replaceRunDirectory,
    replaceExisting,
    replaceResult,
    closeReplace,
    confirmReplace,
  );
  checkpointDialog.append(
    checkpointDialogTitle,
    checkpointDialogContent,
    closeCheckpoint,
  );
  initDialog.append(initForm);
  resumeDialog.append(resumeForm);
  replaceDialog.append(replaceForm);

  directoryForm.append(runDirectory, useRunDirectory);

  metricsRange.value = "200";
  metricsResolution.value = "500";

  metricsRange.append(createOption(), createOption(), createOption());
  metricsResolution.append(
    createOption(),
    createOption(),
    createOption(),
  );

  for (
    const routeNode of [
      openInit,
      openResume,
      stopTraining,
      initForm,
      resumeForm,
      replaceForm,
      replaceDialog,
      initDialog,
      resumeDialog,
      checkpointDialog,
      closeInit,
      closeResume,
      closeReplace,
      closeCheckpoint,
      confirmInit,
      confirmResume,
      confirmReplace,
      openResume,
      openInit,
      runCaption,
      runDirectory,
      useRunDirectory,
      directoryForm,
      directoryError,
      metricsError,
      logsError,
      checkpointsError,
      initFields,
      initStatus,
      initCommandPreview,
      resumeStatus,
      resumeCommandPreview,
      resumeFields,
      processPresence,
      processDetails,
      processCommand,
      processConnectionState,
      processError,
      metricsRange,
      metricsResolution,
      metricStrip,
      logCount,
      logContent,
      loadOlder,
      toggleFollow,
      checkpointDirectory,
      checkpointSummary,
      manifestRows,
      objectRows,
      replaceStatus,
      replaceRunDirectory,
      replaceExisting,
      replaceResult,
      axisUpdate,
      axisElapsed,
      ...refreshButtons,
      ...[
        ...Array.from(document.createDocumentFragment().children),
      ],
    ]
  ) {
    document.body.append(routeNode);
  }
  for (
    const chartId of [
      "chart-throughput",
      "chart-loss",
      "chart-policy",
      "chart-ppo-timing",
      "chart-rollout",
      "chart-rewards",
      "chart-inference",
      "chart-processes",
    ]
  ) {
    const chart = document.getElementById(chartId);
    if (chart !== null) document.body.append(chart);
  }

  defineGlobal(
    "fetch",
    ((
      input: string | URL,
      init?: RequestInit,
    ): Promise<FakeResponse> => {
      const target = input.toString();
      const method = init?.method ?? "GET";
      fetchCalls.push({ method, path: target });
      if (target.includes("/api/training/config")) {
        return Promise.resolve(
          jsonResponse({
            default_run_dir: "/tmp/run",
            stop_timeout_seconds: 1.0,
          }),
        );
      }
      if (target.includes("/api/training/init") && method === "POST") {
        return Promise.resolve(noContentResponse());
      }
      if (
        target.includes("/api/training/resume") && method === "POST"
      ) {
        return Promise.resolve(noContentResponse());
      }
      if (target.includes("/api/training/stop")) {
        return Promise.resolve(jsonResponse({ forced: false }));
      }
      return Promise.resolve(jsonResponse({}));
    }) as typeof fetch,
  );

  return {
    openInitButton: openInit as unknown as HTMLButtonElement,
    openResumeButton: openResume as unknown as HTMLButtonElement,
    initForm: initForm as unknown as HTMLFormElement,
    resumeForm: resumeForm as unknown as HTMLFormElement,
    confirmInitButton: confirmInit as unknown as HTMLButtonElement,
    confirmResumeButton: confirmResume as unknown as HTMLButtonElement,
    fetchCalls,
    cleanup: async (): Promise<void> => {
      for (const source of openEventSources) {
        source.close();
      }
      for (const timer of scheduledTimeouts) {
        globalThis.clearTimeout(timer);
      }
      scheduledTimeouts.length = 0;
      await waitFrame();
    },
  };
}

const scheduledTimeouts: number[] = [];

function dataPropertyName(attributeName: string): string {
  return attributeName.replace(
    /-([a-z])/g,
    (_match: string, letter: string) => letter.toUpperCase(),
  );
}

const originalSetTimeout = globalThis.setTimeout;
const originalClearTimeout = globalThis.clearTimeout;
defineGlobal(
  "setTimeout",
  ((callback: TimerHandler, timeout?: number, ...rest: unknown[]) => {
    const id = originalSetTimeout(callback, timeout, ...rest) as number;
    scheduledTimeouts.push(id);
    return id;
  }) as typeof setTimeout,
);
defineGlobal(
  "clearTimeout",
  ((id: number): void => {
    const index = scheduledTimeouts.indexOf(id);
    if (index !== -1) scheduledTimeouts.splice(index, 1);
    originalClearTimeout(id);
  }) as typeof clearTimeout,
);

Deno.test(
  "initialize followed by resume restores action state",
  async () => {
    const harness = withFakeDashboardDom();
    try {
      await import("../main.ts");

      await waitUntil(
        () => !harness.openInitButton.disabled,
        "initialize action did not become available",
      );

      harness.openInitButton.dispatchEvent(
        new Event("click", { bubbles: true }),
      );
      assert(
        harness.initForm.reportValidity(),
        "init form must be valid",
      );
      harness.initForm.dispatchEvent(
        new Event("submit", { bubbles: true, cancelable: true }),
      );
      await waitUntil(
        () => !harness.confirmInitButton.disabled,
        "initialize button remained disabled after the request completed",
      );

      assertEquals(
        harness.fetchCalls.some((item) =>
          item.path.includes("/api/training/init")
        ),
        true,
      );

      harness.openResumeButton.dispatchEvent(
        new Event("click", { bubbles: true }),
      );
      assert(
        !harness.confirmResumeButton.disabled,
        "resume button was disabled before submit",
      );
      assert(
        harness.resumeForm.reportValidity(),
        "resume form must be valid",
      );
      harness.resumeForm.dispatchEvent(
        new Event("submit", { bubbles: true, cancelable: true }),
      );
      await waitUntil(
        () => !harness.confirmResumeButton.disabled,
        "resume button remained disabled after process launch",
      );

      assertEquals(
        harness.fetchCalls.some((item) =>
          item.path.includes("/api/training/resume")
        ),
        true,
      );
    } finally {
      await harness.cleanup();
      defineGlobal("setTimeout", originalSetTimeout);
      defineGlobal("clearTimeout", originalClearTimeout);
    }
  },
);
