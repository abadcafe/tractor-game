import { requiredElement } from "./dom.ts";
import {
  renderTranscriptRecord,
  setTranscriptRecordRerender,
} from "./record-renderer.ts";
import { LLMTranscriptStream } from "./stream.ts";
import type { SeatId, TranscriptRecord, ViewState } from "./types.ts";

const pane = requiredElement("pane", HTMLDivElement);
const tabs = requiredElement("tabs", HTMLDivElement);
const newPill = requiredElement("new-pill", HTMLButtonElement);
const gameIdElement = requiredElement("game-id", HTMLDivElement);

const views = new Map<string, ViewState>();

let selectedSeat = parseSeat(
  new URLSearchParams(globalThis.location.search).get("seat"),
);
let latestRecords: TranscriptRecord[] = [];
const stream = new LLMTranscriptStream();

setTranscriptRecordRerender(() => renderRecords(latestRecords));
start();

function start(): void {
  const gameId = gameIdFromPath();
  if (gameId === null) {
    renderError("missing game id");
    return;
  }
  gameIdElement.textContent = gameId;
  renderCurrentTab();
  if (selectedSeat === null) {
    renderRecords([]);
    return;
  }
  openStream(gameId, selectedSeat);
}

function parseSeat(raw: string | null): SeatId | null {
  if (raw === "a" || raw === "b" || raw === "c" || raw === "d") {
    return raw;
  }
  return null;
}

function gameIdFromPath(): string | null {
  const parts = globalThis.location.pathname.split("/").filter((part) =>
    part !== ""
  );
  if (parts.length < 3 || parts[0] !== "debug" || parts[1] !== "llm") {
    return null;
  }
  return decodeURIComponent(parts[2]);
}

function viewState(seat: SeatId | null): ViewState {
  const key = seat === null ? "none" : String(seat);
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
  if (selectedSeat === null) return;
  const tab = document.createElement("span");
  tab.className = "tab active";
  tab.textContent = `Seat ${selectedSeat}`;
  tabs.appendChild(tab);
}

function openStream(gameId: string, seat: SeatId): void {
  selectedSeat = seat;
  renderCurrentTab();
  resetTranscript(seat);
  stream.open(gameId, seat, applyStreamRecord);
}

function resetTranscript(seat: SeatId | null): void {
  latestRecords = [];
  const key = seat === null ? "none" : String(seat);
  views.set(key, { knownCount: 0, newCount: 0, stickToBottom: true });
  renderRecords([]);
  updateNewPill();
}

function applyStreamRecord(record: TranscriptRecord): void {
  const beforeNearBottom = isNearBottom();
  const beforeTop = pane.scrollTop;
  const state = viewState(selectedSeat);
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
  const index = latestRecords.findIndex((candidate) =>
    candidate.id === record.id
  );
  if (index === -1) {
    latestRecords.push(record);
  } else {
    latestRecords[index] = record;
  }
  latestRecords.sort((left, right) => left.id - right.id);
}

function renderRecords(records: readonly TranscriptRecord[]): void {
  pane.innerHTML = "";
  if (selectedSeat === null || records.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "";
    pane.appendChild(empty);
    return;
  }
  for (const record of records) {
    pane.appendChild(renderTranscriptRecord(record));
  }
}

function updateNewPill(): void {
  const state = viewState(selectedSeat);
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

newPill.addEventListener("click", () => {
  const state = viewState(selectedSeat);
  state.stickToBottom = true;
  state.newCount = 0;
  scrollToBottom();
  updateNewPill();
});

pane.addEventListener("scroll", () => {
  const state = viewState(selectedSeat);
  if (isNearBottom()) {
    state.stickToBottom = true;
    state.newCount = 0;
    updateNewPill();
  } else {
    state.stickToBottom = false;
  }
});
