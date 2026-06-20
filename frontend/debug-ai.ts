type SlotKind = "api_request" | "api_response" | "api_error" | "tool_result";

interface TranscriptRecord {
  id: number;
  event_id: number;
  created_at: string;
  player_index: number;
  seq: number;
  attempt: number;
  api_request: string | null;
  api_response: string | null;
  api_error: string | null;
  tool_result: string | null;
}

interface ViewState {
  knownCount: number;
  newCount: number;
  stickToBottom: boolean;
}

interface SlotDefinition {
  kind: SlotKind;
  title: string;
}

type JsonParseResult = { ok: true; value: unknown } | { ok: false };
type KvEntry = readonly [label: string, value: unknown];
type Path = readonly string[];

const pane = requiredElement("pane", HTMLDivElement);
const tabs = requiredElement("tabs", HTMLDivElement);
const newPill = requiredElement("new-pill", HTMLButtonElement);
const gameIdElement = requiredElement("game-id", HTMLDivElement);

const expanded = new Map<string, boolean>();
const rawViews = new Map<string, boolean>();
const views = new Map<string, ViewState>();
const slotOrder: readonly SlotDefinition[] = [
  { kind: "api_request", title: "API REQUEST" },
  { kind: "api_response", title: "API RESPONSE" },
  { kind: "api_error", title: "API ERROR" },
  { kind: "tool_result", title: "TOOL RESULT" },
];

let selectedPlayer = parsePlayer(
  new URLSearchParams(window.location.search).get("player"),
);
let latestRecords: TranscriptRecord[] = [];
let socket: WebSocket | null = null;
let streamGeneration = 0;

start();

function start(): void {
  const gameId = gameIdFromPath();
  if (gameId === null) {
    renderError("missing game id");
    return;
  }
  gameIdElement.textContent = gameId;
  renderCurrentTab();
  if (selectedPlayer === null) {
    renderRecords([]);
    return;
  }
  openStream(gameId, selectedPlayer);
}

function parsePlayer(raw: string | null): number | null {
  if (raw === null || raw.trim() === "") return null;
  const value = Number(raw);
  return Number.isInteger(value) ? value : null;
}

function gameIdFromPath(): string | null {
  const parts = window.location.pathname.split("/").filter((part) => part !== "");
  if (parts.length < 3 || parts[0] !== "debug" || parts[1] !== "ai") {
    return null;
  }
  return decodeURIComponent(parts[2]);
}

function viewState(player: number | null): ViewState {
  const key = player === null ? "none" : String(player);
  const existing = views.get(key);
  if (existing !== undefined) return existing;
  const created: ViewState = {
    knownCount: 0,
    newCount: 0,
    stickToBottom: true,
  };
  views.set(key, created);
  return created;
}

function isNearBottom(): boolean {
  return pane.scrollHeight - pane.scrollTop - pane.clientHeight < 40;
}

function renderCurrentTab(): void {
  tabs.innerHTML = "";
  if (selectedPlayer === null) return;
  const tab = document.createElement("span");
  tab.className = "tab active";
  tab.textContent = `Player ${selectedPlayer}`;
  tabs.appendChild(tab);
}

function openStream(gameId: string, player: number): void {
  selectedPlayer = player;
  renderCurrentTab();
  resetTranscript(player);
  const generation = streamGeneration + 1;
  streamGeneration = generation;
  if (socket !== null) {
    socket.close();
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const url =
    `${protocol}//${window.location.host}/ws/debug/ai/${encodeURIComponent(gameId)}?player=${player}`;
  socket = new WebSocket(url);
  socket.addEventListener("message", (event: MessageEvent<string>) => {
    if (generation !== streamGeneration) return;
    const parsed = parseJson(event.data);
    if (!parsed.ok) return;
    const record = transcriptRecord(parsed.value);
    if (record === null) return;
    applyStreamRecord(record);
  });
  socket.addEventListener("close", () => {
    if (generation !== streamGeneration) return;
    socket = null;
  });
}

function resetTranscript(player: number | null): void {
  latestRecords = [];
  const key = player === null ? "none" : String(player);
  views.set(key, { knownCount: 0, newCount: 0, stickToBottom: true });
  renderRecords([]);
  updateNewPill();
}

function applyStreamRecord(record: TranscriptRecord): void {
  const beforeNearBottom = isNearBottom();
  const beforeTop = pane.scrollTop;
  const state = viewState(selectedPlayer);
  const shouldStickToBottom = beforeNearBottom || state.stickToBottom;
  appendRecord(record);
  state.knownCount += 1;
  if (!shouldStickToBottom) {
    state.newCount += 1;
  }
  renderRecords(latestRecords);
  if (shouldStickToBottom) {
    scrollToBottom();
    state.newCount = 0;
  } else {
    pane.scrollTop = beforeTop;
  }
  updateNewPill();
}

function appendRecord(record: TranscriptRecord): void {
  const index = latestRecords.findIndex((candidate) => candidate.id === record.id);
  if (index === -1) {
    latestRecords.push(record);
  } else {
    latestRecords[index] = record;
  }
  latestRecords.sort((left, right) => left.id - right.id);
}

function renderRecords(records: readonly TranscriptRecord[]): void {
  pane.innerHTML = "";
  if (selectedPlayer === null || records.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "";
    pane.appendChild(empty);
    return;
  }
  for (const record of records) {
    pane.appendChild(renderRecord(record));
  }
}

function renderRecord(record: TranscriptRecord): HTMLElement {
  const root = document.createElement("section");
  root.className = "decision";
  const head = document.createElement("div");
  head.className = "decision-head";
  const key = `record:${record.id}`;
  const isOpen = expanded.has(key) ? expanded.get(key) === true : true;
  const toggle = document.createElement("button");
  toggle.className = "item-toggle";
  toggle.textContent = `${isOpen ? "▾" : "▸"} #${record.id}`;
  toggle.addEventListener("click", () => {
    expanded.set(key, !isOpen);
    renderRecords(latestRecords);
  });
  head.append(
    toggle,
    textSpan(`player ${record.player_index}`),
    textSpan(`seq ${record.seq}`),
    textSpan(`attempt ${record.attempt}`),
    textSpan(record.created_at),
  );
  root.appendChild(head);
  if (isOpen) {
    for (const slot of slotOrder) {
      root.appendChild(renderSlot(record, slot));
    }
  }
  return root;
}

function renderSlot(record: TranscriptRecord, slot: SlotDefinition): HTMLElement {
  const root = document.createElement("div");
  root.className = "item";
  const head = document.createElement("div");
  head.className = "item-head";
  const raw = record[slot.kind];
  const rawKey = `raw:${record.id}:${slot.kind}`;
  const slotKey = `slot:${record.id}:${slot.kind}`;
  const isOpen = expanded.has(slotKey) ? expanded.get(slotKey) === true : false;
  const showRaw = rawViews.has(rawKey) ? rawViews.get(rawKey) === true : false;
  const content = raw === null
    ? "<empty>"
    : (showRaw ? raw : formatSlot(slot.kind, raw));
  const toggle = document.createElement("button");
  toggle.className = "item-toggle";
  toggle.textContent = `${isOpen ? "▾" : "▸"} ${slot.title}`;
  toggle.addEventListener("click", () => {
    expanded.set(slotKey, !isOpen);
    renderRecords(latestRecords);
  });
  const kindLabel = document.createElement("span");
  kindLabel.className = "item-kind";
  kindLabel.textContent = raw === null ? "empty" : (showRaw ? "raw" : "parsed");
  const actions = document.createElement("div");
  actions.className = "item-actions";
  const rawButton = actionButton(showRaw ? "解析" : "raw", () => {
    rawViews.set(rawKey, !showRaw);
    renderRecords(latestRecords);
  });
  if (raw === null) rawButton.disabled = true;
  if (isOpen) {
    actions.append(kindLabel, rawButton, copyButton("复制", () => content));
  } else {
    actions.appendChild(slotSummaryNode(slot.kind, raw));
  }
  head.append(toggle, actions);
  root.appendChild(head);
  if (!isOpen) return root;
  if (raw === null || showRaw) {
    root.appendChild(preBlock(content, ""));
  } else {
    const parsed = parseJson(raw);
    root.appendChild(parsed.ok
      ? renderParsedSlot(slot.kind, parsed.value, [`record:${record.id}`, slot.kind])
      : preBlock(raw, ""));
  }
  return root;
}

function formatSlot(kind: SlotKind, raw: string): string {
  const parsed = parseJson(raw);
  if (!parsed.ok) return raw;
  if (kind === "api_request") return formatApiRequest(parsed.value);
  if (kind === "api_response") return formatApiResponse(parsed.value);
  if (kind === "api_error") return formatApiError(parsed.value);
  if (kind === "tool_result") return formatToolResult(parsed.value);
  return stringify(parsed.value);
}

function slotSummaryNode(kind: SlotKind, raw: string | null): HTMLSpanElement {
  const node = document.createElement("span");
  node.className = "slot-summary";
  node.textContent = raw === null ? "empty" : slotSummary(kind, raw);
  return node;
}

function slotSummary(kind: SlotKind, raw: string): string {
  const parsed = parseJson(raw);
  if (!parsed.ok) return "unparseable";
  if (kind === "api_request") return apiRequestSummary(parsed.value);
  if (kind === "api_response") return apiResponseSummary(parsed.value);
  if (kind === "api_error") return apiErrorSummary(parsed.value);
  if (kind === "tool_result") return toolResultSummary(parsed.value);
  return "parsed";
}

function apiRequestSummary(value: unknown): string {
  const payload = recordValue(valueAt(value, "json"));
  if (payload === null) return "invalid request";
  const tools = toolNamesFromSpecs(payload.tools);
  const toolText = tools.length === 0
    ? "no tools"
    : (tools.length === 1 ? tools[0] : `${tools.length} tools`);
  return summaryParts([
    summaryText(payload.model),
    toolText,
    `prompt ${formatCompactNumber(promptCharCount(payload.messages))}`,
    `max ${textValue(payload.max_tokens)}`,
  ]);
}

function apiResponseSummary(value: unknown): string {
  const status = summaryText(valueAt(value, "status_code"));
  const body = recordValue(valueAt(value, "body"));
  const choice = body === null ? null : firstRecord(body.choices);
  const message = choice === null ? null : recordValue(choice.message);
  const tools = message === null ? [] : toolNamesFromCalls(message.tool_calls);
  const toolText = tools.length === 0
    ? "no tool call"
    : (tools.length === 1 ? tools[0] : `${tools.length} tool calls`);
  const usage = body === null ? null : recordValue(body.usage);
  const totalTokens = usage === null ? null : numericValue(usage.total_tokens);
  return summaryParts([
    status,
    summaryText(choice?.finish_reason) === null ? null : `finish=${summaryText(choice?.finish_reason)}`,
    toolText,
    totalTokens === null ? null : `${formatCompactNumber(totalTokens)} tok`,
    durationText(valueAt(value, "duration_ms")),
  ]);
}

function apiErrorSummary(value: unknown): string {
  const record = recordValue(value);
  const errors = record !== null && Array.isArray(record.errors) ? record.errors : [value];
  const last = recordValue(errors[errors.length - 1]) ?? recordValue(value);
  if (last === null) return "invalid error";
  const status = summaryText(last.status_code);
  const title = summaryText(last.title) ?? "error";
  const attempt = attemptText(last);
  const reason = summaryText(last.reason) ?? summaryText(last.error);
  const count = errors.length > 1 ? `${errors.length} errors` : null;
  return summaryParts([
    status === null ? count : `HTTP ${status}`,
    errors.length > 1 ? `last=${title}` : title,
    attempt,
    durationText(last.duration_ms),
    status === null ? truncateText(reason, 56) : null,
  ]);
}

function toolResultSummary(value: unknown): string {
  const record = recordValue(value);
  if (record === null) return "invalid tool result";
  const status = summaryText(record.status) ?? "unknown";
  const errorType = summaryText(record.error_type);
  const toolCall = recordValue(record.tool_call);
  const name = summaryText(toolCall?.name);
  const reason = summaryText(record.reason);
  const cardCount = toolCardCount(toolCall, recordValue(record.message));
  const statusText = status === "rejected" && errorType !== null ? `rejected(${errorType})` : status;
  if (status === "accepted") {
    return summaryParts([
      statusText,
      name,
      cardCount === null ? null : `${cardCount} cards`,
    ]);
  }
  return summaryParts([
    statusText,
    name,
    truncateText(reason, 56),
  ]);
}

function renderParsedSlot(kind: SlotKind, value: unknown, path: Path): HTMLElement {
  if (kind === "api_request") return renderParsedApiRequest(value, path);
  if (kind === "api_response") return renderParsedApiResponse(value, path);
  if (kind === "api_error") return renderParsedApiError(value, path);
  if (kind === "tool_result") return renderParsedToolResult(value, path);
  const root = parsedRoot();
  root.appendChild(preBlock(stringify(value), "parsed-code"));
  return root;
}

function renderParsedApiRequest(value: unknown, path: Path): HTMLElement {
  const root = parsedRoot();
  const json = recordValue(valueAt(value, "json"));
  const summary: KvEntry[] = [
    ["method", valueAt(value, "method")],
    ["endpoint", valueAt(value, "endpoint")],
  ];
  if (json !== null) {
    summary.push(["model", json.model]);
    summary.push(["max_tokens", json.max_tokens]);
    summary.push(["tool_choice", json.tool_choice]);
    summary.push(["thinking", compactJson(json.thinking)]);
  }
  root.appendChild(sectionBlock("REQUEST SUMMARY", kvGrid(summary), [...path, "summary"]));
  if (json !== null) {
    root.appendChild(sectionBlock(
      "SYSTEM PROMPT",
      preBlock(messageContent(json.messages, "system"), "parsed-code"),
      [...path, "system_prompt"],
    ));
    root.appendChild(sectionBlock(
      "USER PROMPT",
      preBlock(messageContent(json.messages, "user"), "parsed-code"),
      [...path, "user_prompt"],
    ));
    root.appendChild(sectionBlock("TOOLS", renderToolList(json.tools, [...path, "tools"]), [...path, "tools"]));
  }
  return root;
}

function renderParsedApiResponse(value: unknown, path: Path): HTMLElement {
  const root = parsedRoot();
  const body = recordValue(valueAt(value, "body"));
  const summary: KvEntry[] = [
    ["title", valueAt(value, "title")],
    ["attempt", `${textValue(valueAt(value, "attempt"))}/${textValue(valueAt(value, "max_attempts"))}`],
    ["duration", durationText(valueAt(value, "duration_ms"))],
    ["status_code", valueAt(value, "status_code")],
  ];
  if (body !== null) {
    summary.push(["model", body.model]);
    summary.push(["usage", compactJson(body.usage)]);
  }
  root.appendChild(sectionBlock("RESPONSE SUMMARY", kvGrid(summary), [...path, "summary"]));
  if (body === null) {
    root.appendChild(sectionBlock("BODY", preBlock(stringify(valueAt(value, "body")), "parsed-code"), [...path, "body"]));
    return root;
  }
  const choice = firstRecord(body.choices);
  if (choice === null) {
    root.appendChild(sectionBlock("BODY", preBlock(stringify(body), "parsed-code"), [...path, "body"]));
    return root;
  }
  root.appendChild(sectionBlock(
    "CHOICE SUMMARY",
    kvGrid([["finish_reason", choice.finish_reason]]),
    [...path, "choice_summary"],
  ));
  const message = recordValue(choice.message);
  if (message === null) {
    root.appendChild(sectionBlock("MESSAGE", preBlock(stringify(choice), "parsed-code"), [...path, "message"]));
    return root;
  }
  root.appendChild(sectionBlock(
    "ASSISTANT RESPONSE",
    preBlock(textValue(message.content), "parsed-code"),
    [...path, "assistant_response"],
  ));
  if (message.reasoning_content !== undefined && message.reasoning_content !== null) {
    root.appendChild(sectionBlock(
      "REASONING CONTENT",
      preBlock(textValue(message.reasoning_content), "parsed-code"),
      [...path, "reasoning_content"],
    ));
  }
  root.appendChild(sectionBlock("TOOL CALLS", renderToolCalls(message.tool_calls, [...path, "tool_calls"]), [
    ...path,
    "tool_calls",
  ]));
  return root;
}

function renderParsedApiError(value: unknown, path: Path): HTMLElement {
  const root = parsedRoot();
  const record = recordValue(value);
  const errors = record !== null && Array.isArray(record.errors) ? record.errors : [value];
  const list = document.createElement("div");
  list.className = "parsed-list";
  errors.forEach((error, index) => {
    const title = errors.length > 1 ? `ERROR ${index + 1}` : "ERROR";
    list.appendChild(sectionBlock(title, renderObjectSummary(error, [...path, `error:${index}`]), [
      ...path,
      `error:${index}`,
    ]));
  });
  root.appendChild(list);
  return root;
}

function renderParsedToolResult(value: unknown, path: Path): HTMLElement {
  const root = parsedRoot();
  root.appendChild(sectionBlock("RESULT SUMMARY", kvGrid([
    ["status", valueAt(value, "status")],
    ["error_type", valueAt(value, "error_type")],
    ["reason", valueAt(value, "reason")],
    ["repair", valueAt(value, "repair")],
  ]), [...path, "summary"]));
  const toolCall = valueAt(value, "tool_call");
  const message = valueAt(value, "message");
  const previous = valueAt(value, "previous_tool_result");
  if (toolCall !== undefined && toolCall !== null) {
    root.appendChild(sectionBlock("TOOL CALL", renderToolCall(toolCall, [...path, "tool_call"]), [...path, "tool_call"]));
  }
  if (message !== undefined && message !== null) {
    root.appendChild(sectionBlock("PLAYER MESSAGE", renderObjectSummary(message, [...path, "message"]), [
      ...path,
      "message",
    ]));
  }
  if (previous !== undefined && previous !== null) {
    root.appendChild(sectionBlock("PREVIOUS TOOL RESULT", renderObjectSummary(previous, [...path, "previous"]), [
      ...path,
      "previous",
    ]));
  }
  return root;
}

function formatApiRequest(value: unknown): string {
  const json = recordValue(valueAt(value, "json"));
  const lines = [
    `method: ${textValue(valueAt(value, "method"))}`,
    `endpoint: ${textValue(valueAt(value, "endpoint"))}`,
  ];
  if (json !== null) {
    lines.push(`model: ${textValue(json.model)}`);
    lines.push(`max_tokens: ${textValue(json.max_tokens)}`);
    lines.push(`tool_choice: ${textValue(json.tool_choice)}`);
    lines.push(`thinking: ${stringify(json.thinking)}`);
    lines.push("");
    lines.push("SYSTEM PROMPT");
    lines.push(messageContent(json.messages, "system"));
    lines.push("");
    lines.push("USER PROMPT");
    lines.push(messageContent(json.messages, "user"));
    lines.push("");
    lines.push("TOOLS");
    lines.push(formatTools(json.tools));
  }
  return lines.join("\n");
}

function formatApiResponse(value: unknown): string {
  const body = recordValue(valueAt(value, "body"));
  const lines = [
    `title: ${textValue(valueAt(value, "title"))}`,
    `attempt: ${textValue(valueAt(value, "attempt"))}/${textValue(valueAt(value, "max_attempts"))}`,
    `duration: ${textValue(durationText(valueAt(value, "duration_ms")))}`,
    `status_code: ${textValue(valueAt(value, "status_code"))}`,
  ];
  if (body !== null) {
    lines.push(`model: ${textValue(body.model)}`);
    lines.push(`usage: ${stringify(body.usage)}`);
    const choice = firstRecord(body.choices);
    if (choice !== null) {
      lines.push(`finish_reason: ${textValue(choice.finish_reason)}`);
      const message = recordValue(choice.message);
      if (message !== null) {
        lines.push("");
        lines.push("ASSISTANT RESPONSE");
        lines.push(textValue(message.content));
        if (message.reasoning_content !== undefined && message.reasoning_content !== null) {
          lines.push("");
          lines.push("REASONING CONTENT");
          lines.push(textValue(message.reasoning_content));
        }
        lines.push("");
        lines.push("TOOL CALLS");
        lines.push(stringify(message.tool_calls));
      }
    }
  } else {
    lines.push("");
    lines.push(stringify(valueAt(value, "body")));
  }
  return lines.join("\n");
}

function formatApiError(value: unknown): string {
  return stringify(value);
}

function formatToolResult(value: unknown): string {
  const lines = [
    `status: ${textValue(valueAt(value, "status"))}`,
  ];
  const errorType = valueAt(value, "error_type");
  if (errorType !== undefined && errorType !== null) {
    lines.push(`error_type: ${textValue(errorType)}`);
  }
  const reason = valueAt(value, "reason");
  if (reason !== undefined && reason !== null) {
    lines.push(`reason: ${textValue(reason)}`);
  }
  const repair = valueAt(value, "repair");
  if (repair !== undefined && repair !== null) {
    lines.push(`repair: ${textValue(repair)}`);
  }
  const toolCall = valueAt(value, "tool_call");
  if (toolCall !== undefined && toolCall !== null) {
    lines.push("");
    lines.push("TOOL CALL");
    lines.push(stringify(toolCall));
  }
  const message = valueAt(value, "message");
  if (message !== undefined && message !== null) {
    lines.push("");
    lines.push("PLAYER MESSAGE");
    lines.push(stringify(message));
  }
  const previous = valueAt(value, "previous_tool_result");
  if (previous !== undefined && previous !== null) {
    lines.push("");
    lines.push("PREVIOUS TOOL RESULT");
    lines.push(stringify(previous));
  }
  return lines.join("\n");
}

function messageContent(messages: unknown, role: string): string {
  if (!Array.isArray(messages)) return "<missing>";
  const message = messages.find((item) => {
    const candidate = recordValue(item);
    return candidate !== null && candidate.role === role;
  });
  const record = recordValue(message);
  if (record === null || typeof record.content !== "string") return "<missing>";
  return record.content;
}

function toolNamesFromSpecs(tools: unknown): string[] {
  if (!Array.isArray(tools)) return [];
  const result: string[] = [];
  for (const tool of tools) {
    const toolRecord = recordValue(tool);
    const fn = toolRecord === null ? null : recordValue(toolRecord.function);
    const name = summaryText(fn?.name);
    if (name !== null) result.push(name);
  }
  return result;
}

function toolNamesFromCalls(toolCalls: unknown): string[] {
  if (!Array.isArray(toolCalls)) return [];
  const result: string[] = [];
  for (const call of toolCalls) {
    const callRecord = recordValue(call);
    const fn = callRecord === null ? null : recordValue(callRecord.function);
    const name = summaryText(fn?.name);
    if (name !== null) result.push(name);
  }
  return result;
}

function promptCharCount(messages: unknown): number {
  if (!Array.isArray(messages)) return 0;
  let total = 0;
  for (const item of messages) {
    const message = recordValue(item);
    if (message !== null && typeof message.content === "string") {
      total += message.content.length;
    }
  }
  return total;
}

function toolCardCount(toolCall: Record<string, unknown> | null, message: Record<string, unknown> | null): number | null {
  const argumentsRecord = toolCall === null ? null : recordValue(toolCall.arguments);
  const argumentCount = stringArrayLength(argumentsRecord?.card_ids);
  if (argumentCount !== null) return argumentCount;
  const raw = message === null ? null : recordValue(message.raw);
  return stringArrayLength(raw?.cards);
}

function stringArrayLength(value: unknown): number | null {
  if (!Array.isArray(value)) return null;
  return value.every((item) => typeof item === "string") ? value.length : null;
}

function attemptText(record: Record<string, unknown>): string | null {
  const attempt = summaryText(record.attempt);
  if (attempt === null) return null;
  const maxAttempts = summaryText(record.max_attempts);
  return maxAttempts === null ? `attempt ${attempt}` : `attempt ${attempt}/${maxAttempts}`;
}

function summaryParts(parts: readonly (string | null)[]): string {
  const present = parts.filter((part): part is string => part !== null && part !== "");
  return present.length === 0 ? "empty" : present.join(" · ");
}

function summaryText(value: unknown): string | null {
  if (value === undefined || value === null) return null;
  if (typeof value === "string") return value === "" ? null : value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return null;
}

function numericValue(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}

function formatCompactNumber(value: number): string {
  if (value < 1000) return String(value);
  const formatted = (value / 1000).toFixed(value < 10000 ? 1 : 0);
  return `${formatted.replace(/\.0$/, "")}k`;
}

function durationText(value: unknown): string | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  if (value < 1000) return `${Math.round(value)}ms`;
  const seconds = value / 1000;
  const decimals = seconds < 10 ? 1 : 0;
  return `${seconds.toFixed(decimals).replace(/\.0$/, "")}s`;
}

function truncateText(value: string | null, maxLength: number): string | null {
  if (value === null || value.length <= maxLength) return value;
  return `${value.slice(0, maxLength - 1)}…`;
}

function formatTools(tools: unknown): string {
  if (!Array.isArray(tools)) return "<missing>";
  return tools.map((tool, index) => {
    const toolRecord = recordValue(tool);
    const fn = toolRecord === null ? null : recordValue(toolRecord.function);
    if (fn === null) return `${index + 1}. ${stringify(tool)}`;
    return `${index + 1}. ${textValue(fn.name)}\n${textValue(fn.description)}\n${stringify(fn.parameters)}`;
  }).join("\n\n");
}

function renderToolList(tools: unknown, path: Path): HTMLElement {
  const list = document.createElement("div");
  list.className = "parsed-list";
  if (!Array.isArray(tools)) {
    list.appendChild(preBlock("<missing>", "parsed-code"));
    return list;
  }
  tools.forEach((tool, index) => {
    const toolRecord = recordValue(tool);
    const fn = toolRecord === null ? null : recordValue(toolRecord.function);
    if (fn === null) {
      list.appendChild(sectionBlock(
        `TOOL ${index + 1}`,
        preBlock(stringify(tool), "parsed-code"),
        [...path, `tool:${index}`],
      ));
      return;
    }
    const body = document.createElement("div");
    body.appendChild(sectionBlock("TOOL SUMMARY", kvGrid([
      ["name", fn.name],
      ["description", fn.description],
      ["strict", fn.strict],
    ]), [...path, `tool:${index}`, "summary"]));
    body.appendChild(sectionBlock(
      "PARAMETERS",
      preBlock(stringify(fn.parameters), "parsed-code"),
      [...path, `tool:${index}`, "parameters"],
    ));
    list.appendChild(sectionBlock(`TOOL ${index + 1}`, body, [...path, `tool:${index}`]));
  });
  return list;
}

function renderToolCalls(toolCalls: unknown, path: Path): HTMLElement {
  const list = document.createElement("div");
  list.className = "parsed-list";
  if (!Array.isArray(toolCalls)) {
    list.appendChild(preBlock(stringify(toolCalls), "parsed-code"));
    return list;
  }
  toolCalls.forEach((toolCall, index) => {
    list.appendChild(sectionBlock(
      `CALL ${index + 1}`,
      renderToolCall(toolCall, [...path, `call:${index}`]),
      [...path, `call:${index}`],
    ));
  });
  return list;
}

function renderToolCall(value: unknown, path: Path): HTMLElement {
  const root = document.createElement("div");
  const record = recordValue(value);
  if (record === null) {
    root.appendChild(preBlock(stringify(value), "parsed-code"));
    return root;
  }
  const fn = recordValue(record.function);
  if (fn === null) {
    root.appendChild(renderObjectSummary(record, [...path, "summary"]));
    return root;
  }
  root.appendChild(sectionBlock("CALL SUMMARY", kvGrid([
    ["type", record.type],
    ["name", fn.name],
  ]), [...path, "summary"]));
  const argumentsRaw = typeof fn.arguments === "string"
    ? fn.arguments
    : stringify(fn.arguments);
  const argumentsParsed = parseJson(argumentsRaw);
  root.appendChild(sectionBlock(
    "ARGUMENTS",
    preBlock(argumentsParsed.ok ? stringify(argumentsParsed.value) : argumentsRaw, "parsed-code"),
    [...path, "arguments"],
  ));
  return root;
}

function renderObjectSummary(value: unknown, path: Path): HTMLElement {
  const root = document.createElement("div");
  const record = recordValue(value);
  if (record === null) {
    root.appendChild(preBlock(stringify(value), "parsed-code"));
    return root;
  }
  const simpleEntries: KvEntry[] = [];
  const complexEntries: KvEntry[] = [];
  for (const [key, item] of Object.entries(record)) {
    if (
      item === null || typeof item === "string" || typeof item === "number" ||
      typeof item === "boolean"
    ) {
      simpleEntries.push([key, item]);
    } else {
      complexEntries.push([key, item]);
    }
  }
  if (simpleEntries.length > 0) {
    root.appendChild(sectionBlock("SUMMARY", kvGrid(simpleEntries), [...path, "summary"]));
  }
  for (const [key, item] of complexEntries) {
    root.appendChild(sectionBlock(
      key.toUpperCase(),
      preBlock(stringify(item), "parsed-code"),
      [...path, key],
    ));
  }
  return root;
}

function parsedRoot(): HTMLDivElement {
  const root = document.createElement("div");
  root.className = "parsed-view";
  return root;
}

function kvGrid(entries: readonly KvEntry[]): HTMLDivElement {
  const grid = document.createElement("div");
  grid.className = "kv-grid";
  for (const [label, value] of entries) {
    if (value === undefined || value === null) continue;
    const labelNode = document.createElement("div");
    labelNode.className = "kv-label";
    labelNode.textContent = label;
    const valueNode = document.createElement("div");
    valueNode.className = "kv-value";
    valueNode.textContent = textValue(value);
    grid.append(labelNode, valueNode);
  }
  return grid;
}

function sectionBlock(title: string, body: Node, path: Path): HTMLDivElement {
  const section = document.createElement("div");
  section.className = "parsed-section";
  const key = `section:${path.join("/")}`;
  const isOpen = expanded.has(key) ? expanded.get(key) === true : true;
  const titleNode = document.createElement("button");
  titleNode.className = "parsed-section-title";
  titleNode.type = "button";
  titleNode.textContent = `${isOpen ? "▾" : "▸"} ${title}`;
  titleNode.addEventListener("click", () => {
    expanded.set(key, !isOpen);
    renderRecords(latestRecords);
  });
  section.appendChild(titleNode);
  if (isOpen) {
    section.appendChild(body);
  }
  return section;
}

function preBlock(text: string, className: string): HTMLPreElement {
  const body = document.createElement("pre");
  body.className = className;
  body.textContent = text;
  return body;
}

function firstRecord(value: unknown): Record<string, unknown> | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  return recordValue(value[0]);
}

function recordValue(value: unknown): Record<string, unknown> | null {
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return null;
}

function valueAt(value: unknown, key: string): unknown {
  const record = recordValue(value);
  return record === null ? undefined : record[key];
}

function textValue(value: unknown): string {
  if (value === undefined || value === null) return "<empty>";
  if (typeof value === "string") return value === "" ? "<empty>" : value;
  return String(value);
}

function stringify(value: unknown): string {
  if (value === undefined) return "<missing>";
  if (typeof value === "string") return value === "" ? "<empty>" : value;
  return JSON.stringify(value, null, 2);
}

function compactJson(value: unknown): string {
  if (value === undefined) return "<missing>";
  if (typeof value === "string") return value === "" ? "<empty>" : value;
  return JSON.stringify(value);
}

function parseJson(raw: string): JsonParseResult {
  try {
    return { ok: true, value: JSON.parse(raw) as unknown };
  } catch (_error) {
    return { ok: false };
  }
}

function transcriptRecord(value: unknown): TranscriptRecord | null {
  const record = recordValue(value);
  if (record === null) return null;
  const id = record.id;
  const eventId = record.event_id;
  const createdAt = record.created_at;
  const playerIndex = record.player_index;
  const seq = record.seq;
  const attempt = record.attempt;
  if (
    typeof id !== "number" || typeof eventId !== "number" ||
    typeof createdAt !== "string" || typeof playerIndex !== "number" ||
    typeof seq !== "number" || typeof attempt !== "number"
  ) {
    return null;
  }
  const apiRequest = nullableString(record.api_request);
  const apiResponse = nullableString(record.api_response);
  const apiError = nullableString(record.api_error);
  const toolResult = nullableString(record.tool_result);
  if (
    apiRequest === undefined || apiResponse === undefined ||
    apiError === undefined || toolResult === undefined
  ) {
    return null;
  }
  return {
    id,
    event_id: eventId,
    created_at: createdAt,
    player_index: playerIndex,
    seq,
    attempt,
    api_request: apiRequest,
    api_response: apiResponse,
    api_error: apiError,
    tool_result: toolResult,
  };
}

function nullableString(value: unknown): string | null | undefined {
  if (value === null) return null;
  if (typeof value === "string") return value;
  return undefined;
}

function actionButton(label: string, onClick: () => void): HTMLButtonElement {
  const button = document.createElement("button");
  button.className = "copy-btn";
  button.textContent = label;
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    onClick();
  });
  return button;
}

function copyButton(label: string, getText: () => string): HTMLButtonElement {
  const button = document.createElement("button");
  button.className = "copy-btn";
  button.textContent = label;
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    void copyText(getText(), button, label);
  });
  return button;
}

async function copyText(
  text: string,
  button: HTMLButtonElement,
  label: string,
): Promise<void> {
  if (text === "") return;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      fallbackCopy(text);
    }
    button.textContent = "已复制";
    setTimeout(() => {
      button.textContent = label;
    }, 900);
  } catch (error: unknown) {
    console.warn("copy failed", error);
    button.textContent = "复制失败";
    setTimeout(() => {
      button.textContent = label;
    }, 1200);
  }
}

function fallbackCopy(text: string): void {
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.left = "-9999px";
  area.style.top = "0";
  document.body.appendChild(area);
  area.focus();
  area.select();
  const ok = document.execCommand("copy");
  area.remove();
  if (!ok) throw new Error("document.execCommand copy failed");
}

function textSpan(value: string): HTMLSpanElement {
  const span = document.createElement("span");
  span.textContent = value;
  return span;
}

function updateNewPill(): void {
  const state = viewState(selectedPlayer);
  if (state.newCount <= 0) {
    newPill.style.display = "none";
    return;
  }
  newPill.textContent = `有 ${state.newCount} 条新消息`;
  newPill.style.display = "block";
}

function scrollToBottom(): void {
  pane.style.scrollBehavior = "auto";
  pane.scrollTop = pane.scrollHeight;
  requestAnimationFrame(() => {
    pane.scrollTop = pane.scrollHeight;
    pane.style.scrollBehavior = "";
  });
}

function renderError(message: string): void {
  pane.innerHTML = "";
  const empty = document.createElement("div");
  empty.className = "empty";
  empty.textContent = message;
  pane.appendChild(empty);
}

function requiredElement<T extends HTMLElement>(
  id: string,
  ctor: { new (): T },
): T {
  const element = document.getElementById(id);
  if (!(element instanceof ctor)) {
    throw new Error(`missing element #${id}`);
  }
  return element;
}

newPill.addEventListener("click", () => {
  const state = viewState(selectedPlayer);
  state.stickToBottom = true;
  state.newCount = 0;
  scrollToBottom();
  updateNewPill();
});

pane.addEventListener("scroll", () => {
  const state = viewState(selectedPlayer);
  if (isNearBottom()) {
    state.stickToBottom = true;
    state.newCount = 0;
    updateNewPill();
  } else {
    state.stickToBottom = false;
  }
});
