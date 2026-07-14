Deno.test("dashboard hard-cuts overview summary and sessions", async () => {
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
