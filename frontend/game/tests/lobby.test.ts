import {
  assert,
  assertEquals,
  assertNotEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import {
  type LobbyCallbacks,
  type LobbyState,
  renderLobby,
} from "../ui/lobby.ts";
import type { SeatId } from "../config.ts";
import type { BotPolicyName, ListedSeat } from "../net/rest-client.ts";

function makeState(overrides: Partial<LobbyState> = {}): LobbyState {
  return {
    games: [],
    loading: false,
    creating: false,
    pendingSeatGameId: null,
    pendingSeatId: null,
    deletingGameId: null,
    selectedGameId: null,
    botFillMode: "none",
    errorMessage: null,
    statusMessage: null,
    ...overrides,
  };
}

function callbacksStub(
  overrides: Partial<LobbyCallbacks> = {},
): LobbyCallbacks {
  return {
    onCreateGame: () => {},
    onSelectGame: () => {},
    onDeleteGame: () => {},
    onToggleSeat: () => {},
    onEnterSeat: () => {},
    enterSeatHref: (gameId, seatId) =>
      `/game/${gameId}/seat/${seatId}?user_id=test`,
    onChangeBotFillMode: () => {},
    onRefreshGames: () => {},
    ...overrides,
  };
}

function emptySeat(seat: SeatId): ListedSeat {
  return { seat, player: null, ready: false };
}

function botSeat(
  seat: SeatId,
  policy: BotPolicyName = "auto",
  ready: boolean = false,
): ListedSeat {
  return {
    seat,
    player: { kind: "bot", policy },
    ready,
  };
}

function humanSeat(
  seat: SeatId,
  options: {
    mine: boolean;
    connected: boolean;
  },
): ListedSeat {
  return {
    seat,
    player: {
      kind: "human",
      mine: options.mine,
      connected: options.connected,
    },
    ready: false,
  };
}

function freshRoot(): Element {
  const doc = new DOMParser().parseFromString(
    `<html><body><div id="root"></div></body></html>`,
    "text/html",
  );
  assert(doc !== null);
  Object.defineProperty(globalThis, "document", {
    value: doc as unknown as Document,
    configurable: true,
  });
  const root = doc.querySelector("#root");
  assert(root !== null);
  return root as unknown as Element;
}

Deno.test("test_renderLobby_empty_games", () => {
  const root = freshRoot();
  root.appendChild(renderLobby(makeState(), callbacksStub()));

  assertNotEquals(root.querySelector(".lobby"), null);
  assertEquals(
    root.querySelector(".lobby-empty")?.textContent?.includes(
      "没有可加入的牌局",
    ),
    true,
  );
});

Deno.test("test_renderLobby_shows_game_counts", () => {
  const root = freshRoot();
  root.appendChild(
    renderLobby(
      makeState({
        games: [{
          gameId: "abcdef123456",
          userCount: 2,
          capacity: 4,
          userSeats: ["b", "d"],
          seats: [
            botSeat("a"),
            humanSeat("b", { mine: false, connected: true }),
            emptySeat("c"),
            humanSeat("d", { mine: false, connected: false }),
          ],
        }],
        selectedGameId: "abcdef123456",
      }),
      callbacksStub(),
    ),
  );

  assertEquals(root.querySelector(".lobby-game-row__count"), null);
  assertEquals(
    root.querySelectorAll(".lobby-player-dot--filled").length,
    3,
  );
  assertEquals(
    root.querySelector(".lobby-preview__summary")?.textContent,
    "3/4 人",
  );
});

Deno.test("test_renderLobby_callbacks", () => {
  const root = freshRoot();
  let created = false;
  let refreshed = false;
  let selectedGameId: string | null = null;
  let toggledGameId: string | null = null;
  let toggledSeatId: SeatId | null = null;
  root.appendChild(
    renderLobby(
      makeState({
        games: [{
          gameId: "game-to-join",
          userCount: 1,
          capacity: 4,
          userSeats: ["c"],
          seats: [
            botSeat("a"),
            emptySeat("b"),
            humanSeat("c", { mine: false, connected: true }),
            emptySeat("d"),
          ],
        }],
        selectedGameId: "game-to-join",
      }),
      callbacksStub({
        onCreateGame: () => {
          created = true;
        },
        onRefreshGames: () => {
          refreshed = true;
        },
        onSelectGame: (gameId: string) => {
          selectedGameId = gameId;
        },
        onToggleSeat: (gameId, seatId) => {
          toggledGameId = gameId;
          toggledSeatId = seatId;
        },
      }),
    ),
  );

  const buttons = Array.from(root.querySelectorAll("button"));
  const createButton = buttons.find((button) =>
    button.textContent === "创建牌局"
  );
  const refreshButton = buttons.find((button) =>
    button.textContent === "刷新"
  );
  const gameButton = root.querySelector(".lobby-game-row");
  const seatDButton = Array.from(
    root.querySelectorAll(".lobby-preview-player"),
  ).find((button) => button.getAttribute("data-seat") === "d");
  assert(createButton !== undefined);
  assert(refreshButton !== undefined);
  assert(gameButton !== null);
  assert(seatDButton !== undefined);
  assertEquals(
    buttons.some((button) => button.textContent === "加入"),
    false,
  );

  createButton.dispatchEvent(new Event("click", { bubbles: true }));
  refreshButton.dispatchEvent(new Event("click", { bubbles: true }));
  gameButton.dispatchEvent(new Event("click", { bubbles: true }));
  seatDButton.dispatchEvent(
    new Event("click", { bubbles: true }),
  );

  assertEquals(created, true);
  assertEquals(refreshed, true);
  assertEquals(selectedGameId, "game-to-join");
  assertEquals(toggledGameId, "game-to-join");
  assertEquals(toggledSeatId, "d");
});

Deno.test("test_renderLobby_delete_button_deletes_without_selecting", () => {
  const root = freshRoot();
  let deletedGameId: string | null = null;
  let selectedGameId: string | null = null;
  root.appendChild(
    renderLobby(
      makeState({
        games: [{
          gameId: "delete-game",
          userCount: 0,
          capacity: 4,
          userSeats: [],
          seats: [
            botSeat("a"),
            emptySeat("b"),
            botSeat("c"),
            botSeat("d"),
          ],
        }],
      }),
      callbacksStub({
        onDeleteGame: (gameId) => {
          deletedGameId = gameId;
        },
        onSelectGame: (gameId) => {
          selectedGameId = gameId;
        },
      }),
    ),
  );

  const deleteButton = root.querySelector(".lobby-game-row__delete");
  assert(deleteButton !== null);
  assertEquals(deleteButton.textContent, "删除");

  deleteButton.dispatchEvent(new Event("click", { bubbles: true }));

  assertEquals(deletedGameId, "delete-game");
  assertEquals(selectedGameId, null);
});

Deno.test("test_renderLobby_my_player_toggles_and_center_enters", () => {
  const root = freshRoot();
  let toggledSeatId: SeatId | null = null;
  let enteredGameId: string | null = null;
  let enteredSeatId: SeatId | null = null;
  root.appendChild(
    renderLobby(
      makeState({
        games: [{
          gameId: "mine-game",
          userCount: 1,
          capacity: 4,
          userSeats: ["b"],
          seats: [
            botSeat("a"),
            humanSeat("b", { mine: true, connected: true }),
            botSeat("c"),
            botSeat("d"),
          ],
        }],
        selectedGameId: "mine-game",
      }),
      callbacksStub({
        onToggleSeat: (_gameId, seatId) => {
          toggledSeatId = seatId;
        },
        onEnterSeat: (gameId, seatId) => {
          enteredGameId = gameId;
          enteredSeatId = seatId;
        },
      }),
    ),
  );

  const mySeatButton = Array.from(
    root.querySelectorAll(".lobby-preview-player"),
  ).find((button) => button.getAttribute("data-seat") === "b");
  const enterButton = root.querySelector("[data-enter-table='true']");
  assert(mySeatButton !== undefined);
  assert(enterButton !== null);
  assertEquals(mySeatButton.hasAttribute("disabled"), false);
  assertEquals(enterButton.hasAttribute("disabled"), false);
  assertEquals(enterButton.textContent, "进入牌桌");
  assertEquals(enterButton.getAttribute("target"), "_blank");
  assertEquals(
    enterButton.getAttribute("href"),
    "/game/mine-game/seat/b?user_id=test",
  );
  assertEquals(
    mySeatButton.getAttribute("class")?.includes(
      "lobby-preview-player--mine",
    ),
    true,
  );
  assertEquals(
    root.querySelector(".lobby-preview__summary")?.textContent,
    "4/4 人 · 玩家 b",
  );

  mySeatButton.dispatchEvent(new Event("click", { bubbles: true }));
  enterButton.dispatchEvent(new Event("click", { bubbles: true }));

  assertEquals(toggledSeatId, "b");
  assertEquals(enteredGameId, "mine-game");
  assertEquals(enteredSeatId, "b");
});

Deno.test("test_renderLobby_center_disabled_before_controlling_player", () => {
  const root = freshRoot();
  let entered = false;
  root.appendChild(
    renderLobby(
      makeState({
        games: [{
          gameId: "empty-game",
          userCount: 0,
          capacity: 4,
          userSeats: [],
          seats: [
            emptySeat("a"),
            emptySeat("b"),
            emptySeat("c"),
            emptySeat("d"),
          ],
        }],
        selectedGameId: "empty-game",
      }),
      callbacksStub({
        onEnterSeat: () => {
          entered = true;
        },
      }),
    ),
  );

  const enterButton = root.querySelector("[data-enter-table='true']");
  const botModeButtons = Array.from(
    root.querySelectorAll(".lobby-bot-mode button"),
  );
  assert(enterButton !== null);
  assertEquals(enterButton.tagName.toLowerCase(), "button");
  assertEquals(enterButton.hasAttribute("disabled"), true);
  assertEquals(enterButton.textContent, "进入牌桌");
  assertEquals(
    botModeButtons.every((button) => button.hasAttribute("disabled")),
    true,
  );

  enterButton.dispatchEvent(new Event("click", { bubbles: true }));

  assertEquals(entered, false);
});

Deno.test("test_renderLobby_bot_fill_control_in_player_header", () => {
  const root = freshRoot();
  let selectedMode: string | null = null;
  root.appendChild(
    renderLobby(
      makeState({
        botFillMode: "auto",
        games: [{
          gameId: "bot-game",
          userCount: 1,
          capacity: 4,
          userSeats: ["c"],
          seats: [
            emptySeat("a"),
            emptySeat("b"),
            humanSeat("c", { mine: true, connected: true }),
            emptySeat("d"),
          ],
        }],
        selectedGameId: "bot-game",
      }),
      callbacksStub({
        onChangeBotFillMode: (mode) => {
          selectedMode = mode;
        },
      }),
    ),
  );

  const header = root.querySelector(".lobby-preview__head");
  assert(header !== null);
  assertEquals(
    header.querySelector(".lobby-section-title")?.textContent,
    "玩家",
  );
  const buttons = Array.from(
    header.querySelectorAll(".lobby-bot-mode button"),
  );
  assertEquals(
    buttons.map((button) => button.getAttribute("data-bot-fill-mode")),
    ["none", "llm", "auto"],
  );
  assertEquals(
    buttons.map((button) => button.textContent),
    ["不填充", "LLM", "AUTO"],
  );
  const autoButton = buttons.find((button) =>
    button.getAttribute("data-bot-fill-mode") === "auto"
  );
  const llmButton = buttons.find((button) =>
    button.getAttribute("data-bot-fill-mode") === "llm"
  );
  assert(autoButton !== undefined);
  assert(llmButton !== undefined);
  assertEquals(autoButton.getAttribute("aria-pressed"), "true");

  llmButton.dispatchEvent(new Event("click", { bubbles: true }));

  assertEquals(selectedMode, "llm");
});

Deno.test("test_renderLobby_bot_filled_players_show_kind_labels", () => {
  const root = freshRoot();
  root.appendChild(
    renderLobby(
      makeState({
        games: [{
          gameId: "bot-filled-game",
          userCount: 1,
          capacity: 4,
          userSeats: ["c"],
          seats: [
            botSeat("a", "llm", true),
            botSeat("b", "auto", true),
            humanSeat("c", { mine: true, connected: true }),
            emptySeat("d"),
          ],
        }],
        selectedGameId: "bot-filled-game",
      }),
      callbacksStub(),
    ),
  );

  const seatA = root.querySelector("[data-seat='a']");
  const seatB = root.querySelector("[data-seat='b']");
  assert(seatA !== null);
  assert(seatB !== null);
  assertEquals(seatA.textContent, "ALLM");
  assertEquals(seatB.textContent, "BAUTO");
  assertEquals(
    root.querySelectorAll(".lobby-player-dot--filled").length,
    3,
  );
  assertEquals(
    seatA.getAttribute("class")?.includes(
      "lobby-preview-player--bot",
    ),
    true,
  );
});

Deno.test("test_renderLobby_disables_bot_fill_when_no_empty_players", () => {
  const root = freshRoot();
  let selectedMode: string | null = null;
  root.appendChild(
    renderLobby(
      makeState({
        botFillMode: "auto",
        games: [{
          gameId: "filled-game",
          userCount: 1,
          capacity: 4,
          userSeats: ["b"],
          seats: [
            botSeat("a", "auto", true),
            humanSeat("b", { mine: true, connected: false }),
            botSeat("c", "auto", true),
            botSeat("d", "auto", true),
          ],
        }],
        selectedGameId: "filled-game",
      }),
      callbacksStub({
        onChangeBotFillMode: (mode) => {
          selectedMode = mode;
        },
      }),
    ),
  );

  const botModeButtons = Array.from(
    root.querySelectorAll(".lobby-bot-mode button"),
  );
  const enterButton = root.querySelector("[data-enter-table='true']");
  assert(enterButton !== null);
  assertEquals(enterButton.tagName.toLowerCase(), "a");
  assertEquals(enterButton.hasAttribute("disabled"), false);
  assertEquals(
    botModeButtons.every((button) => button.hasAttribute("disabled")),
    true,
  );

  const llmButton = botModeButtons.find((button) =>
    button.getAttribute("data-bot-fill-mode") === "llm"
  );
  assert(llmButton !== undefined);
  llmButton.dispatchEvent(new Event("click", { bubbles: true }));

  assertEquals(selectedMode, null);
});
