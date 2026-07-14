Deno.test("dashboard hard-cuts legacy summary and sessions", async () => {
  const html = await Deno.readTextFile(
    new URL("../index.html", import.meta.url),
  );
  const source = await Deno.readTextFile(
    new URL("../main.ts", import.meta.url),
  );
  for (
    const forbidden of [
      "#overview",
      'data-view="overview"',
      "Latest event",
      "Latest milestone",
      "Current runtime",
      "metrics-session-select",
      "fetchSummary",
      "TrainingSummary",
      "DashboardRefreshController",
      "event-policy",
    ]
  ) {
    if (html.includes(forbidden) || source.includes(forbidden)) {
      throw new Error(
        `Forbidden legacy frontend surface: ${forbidden}`,
      );
    }
  }
  if (
    !html.includes('href="#process"') ||
    !html.includes('data-view="process"')
  ) throw new Error("Process must be the canonical first route");
  if (html.includes('data-refresh-domain="process"')) {
    throw new Error(
      "Process snapshots are pushed and must not expose manual refresh",
    );
  }
});

Deno.test("status is local to each domain", async () => {
  const html = await Deno.readTextFile(
    new URL("../index.html", import.meta.url),
  );
  const source = await Deno.readTextFile(
    new URL("../main.ts", import.meta.url),
  );
  const processStatus = await Deno.readTextFile(
    new URL("../process-status.ts", import.meta.url),
  );

  if (html.includes('id="connection-state"')) {
    throw new Error("Global connection status must not exist");
  }
  const processView = html.slice(
    html.indexOf('data-view="process"'),
    html.indexOf('data-view="metrics"'),
  );
  if (!processView.includes('id="process-connection-state"')) {
    throw new Error(
      "Process connection status is not local to Process",
    );
  }

  for (
    const id of [
      "process-error",
      "metrics-error",
      "logs-error",
      "checkpoints-error",
    ]
  ) {
    if (!html.includes(`id="${id}"`)) {
      throw new Error(`Missing domain error surface: ${id}`);
    }
  }
  for (const domain of ["metrics", "logs", "checkpoints"]) {
    if (processStatus.includes(`"${domain}"`)) {
      throw new Error(`Process status owns ${domain} errors`);
    }
    if (!source.includes(`setDomainError("${domain}-error"`)) {
      throw new Error(`${domain} errors are not rendered locally`);
    }
  }
});

Deno.test("layout gives code and errors distinct wrapping semantics", async () => {
  const css = await Deno.readTextFile(
    new URL("../style.css", import.meta.url),
  );
  if (css.includes("overflow-wrap: anywhere")) {
    throw new Error("Character-level wrapping is forbidden");
  }
  if (
    !css.includes(".code-value") || !css.includes("overflow-x: auto")
  ) {
    throw new Error(
      "Code and paths need their own horizontal scrolling",
    );
  }
  const start = css.indexOf(".error-value");
  const errorRule = css.slice(start, css.indexOf("}", start));
  if (!errorRule.includes("white-space: normal")) {
    throw new Error(
      "Natural-language errors must remain fully visible",
    );
  }
});

Deno.test("replacement actions stay horizontal and legible on hover", async () => {
  const css = await Deno.readTextFile(
    new URL("../style.css", import.meta.url),
  );
  const actions = rule(css, ".replace-shell .modal-actions");
  if (!actions.includes("flex-direction: row")) {
    throw new Error("Replacement actions must stay in one row");
  }
  const buttons = rule(css, ".replace-shell .modal-actions .button");
  if (!buttons.includes("width: auto")) {
    throw new Error("Replacement buttons must keep intrinsic width");
  }
  const hover = rule(css, ".button-danger-primary:hover");
  if (
    !hover.includes("background:") ||
    !hover.includes("color: #fff")
  ) {
    throw new Error(
      "Destructive button hover must preserve contrast",
    );
  }
});

Deno.test("live metrics precede completed-update metrics", async () => {
  const html = await Deno.readTextFile(
    new URL("../index.html", import.meta.url),
  );
  const inference = html.indexOf('id="chart-inference"');
  const processes = html.indexOf('id="chart-processes"');
  const throughput = html.indexOf('id="chart-throughput"');
  if (
    inference < 0 || processes < 0 || throughput < 0 ||
    inference > throughput || processes > throughput
  ) {
    throw new Error(
      "Metrics available before the first update must be visible first",
    );
  }
});

Deno.test("Metrics owns one route-scoped WebSocket transport", async () => {
  const domain = await Deno.readTextFile(
    new URL("../metrics-domain.ts", import.meta.url),
  );
  const main = await Deno.readTextFile(
    new URL("../main.ts", import.meta.url),
  );
  for (
    const forbidden of [
      "fetchMetrics",
      "MetricsInvalidationStream",
      "dirtyThrough",
      "DEBOUNCE_MS",
    ]
  ) {
    if (domain.includes(forbidden)) {
      throw new Error(`Forbidden dual Metrics transport: ${forbidden}`);
    }
  }
  if (
    !domain.includes("MetricsSnapshotStream") ||
    !domain.includes("this.#stream.disconnect()") ||
    !main.includes("metricsDomain.deactivate()")
  ) {
    throw new Error("Metrics WebSocket must follow the active route");
  }
});

function rule(css: string, selector: string): string {
  const start = css.indexOf(`${selector} {`);
  if (start < 0) throw new Error(`Missing CSS rule: ${selector}`);
  return css.slice(start, css.indexOf("}", start));
}
