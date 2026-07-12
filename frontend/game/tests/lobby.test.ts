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

function makeState(overrides: Partial<LobbyState> = {}): LobbyState {
  return {
    games: [],
    loading: false,
    creating: false,
    pendingPlayerGameId: null,
    pendingPlayerIndex: null,
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
    onTogglePlayer: () => {},
    onEnterPlayer: () => {},
    enterPlayerHref: (gameId, playerIndex) =>
      `/game/${gameId}/player/${playerIndex}?user_id=test`,
    onChangeBotFillMode: () => {},
    onRefreshGames: () => {},
    ...overrides,
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
          userPlayers: [1, 3],
          players: [
            {
              index: 0,
              occupied: true,
              connected: false,
              kind: "auto",
              mine: false,
              ready: false,
            },
            {
              index: 1,
              occupied: true,
              connected: true,
              mine: false,
              ready: false,
            },
            {
              index: 2,
              occupied: false,
              connected: false,
              mine: false,
              ready: false,
            },
            {
              index: 3,
              occupied: true,
              connected: false,
              mine: false,
              ready: false,
            },
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
  let toggledPlayerIndex: number | null = null;
  root.appendChild(
    renderLobby(
      makeState({
        games: [{
          gameId: "game-to-join",
          userCount: 1,
          capacity: 4,
          userPlayers: [2],
          players: [
            {
              index: 0,
              occupied: true,
              connected: false,
              kind: "auto",
              mine: false,
              ready: false,
            },
            {
              index: 1,
              occupied: false,
              connected: false,
              mine: false,
              ready: false,
            },
            {
              index: 2,
              occupied: true,
              connected: true,
              mine: false,
              ready: false,
            },
            {
              index: 3,
              occupied: false,
              connected: false,
              mine: false,
              ready: false,
            },
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
        onTogglePlayer: (gameId, playerIndex) => {
          toggledGameId = gameId;
          toggledPlayerIndex = playerIndex;
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
  const playerThreeButton = Array.from(
    root.querySelectorAll(".lobby-preview-player"),
  ).find((button) => button.getAttribute("data-player-index") === "3");
  assert(createButton !== undefined);
  assert(refreshButton !== undefined);
  assert(gameButton !== null);
  assert(playerThreeButton !== undefined);
  assertEquals(
    buttons.some((button) => button.textContent === "加入"),
    false,
  );

  createButton.dispatchEvent(new Event("click", { bubbles: true }));
  refreshButton.dispatchEvent(new Event("click", { bubbles: true }));
  gameButton.dispatchEvent(new Event("click", { bubbles: true }));
  playerThreeButton.dispatchEvent(
    new Event("click", { bubbles: true }),
  );

  assertEquals(created, true);
  assertEquals(refreshed, true);
  assertEquals(selectedGameId, "game-to-join");
  assertEquals(toggledGameId, "game-to-join");
  assertEquals(toggledPlayerIndex, 3);
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
          userPlayers: [],
          players: [
            {
              index: 0,
              occupied: true,
              connected: false,
              kind: "auto",
              mine: false,
              ready: false,
            },
            {
              index: 1,
              occupied: false,
              connected: false,
              mine: false,
              ready: false,
            },
            {
              index: 2,
              occupied: true,
              connected: false,
              kind: "auto",
              mine: false,
              ready: false,
            },
            {
              index: 3,
              occupied: true,
              connected: false,
              kind: "auto",
              mine: false,
              ready: false,
            },
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
  let toggledPlayerIndex: number | null = null;
  let enteredGameId: string | null = null;
  let enteredPlayerIndex: number | null = null;
  root.appendChild(
    renderLobby(
      makeState({
        games: [{
          gameId: "mine-game",
          userCount: 1,
          capacity: 4,
          userPlayers: [1],
          players: [
            {
              index: 0,
              occupied: true,
              connected: false,
              kind: "auto",
              mine: false,
              ready: false,
            },
            {
              index: 1,
              occupied: true,
              connected: true,
              mine: true,
              ready: false,
            },
            {
              index: 2,
              occupied: true,
              connected: false,
              kind: "auto",
              mine: false,
              ready: false,
            },
            {
              index: 3,
              occupied: true,
              connected: false,
              kind: "auto",
              mine: false,
              ready: false,
            },
          ],
        }],
        selectedGameId: "mine-game",
      }),
      callbacksStub({
        onTogglePlayer: (_gameId, playerIndex) => {
          toggledPlayerIndex = playerIndex;
        },
        onEnterPlayer: (gameId, playerIndex) => {
          enteredGameId = gameId;
          enteredPlayerIndex = playerIndex;
        },
      }),
    ),
  );

  const myPlayerButton = Array.from(
    root.querySelectorAll(".lobby-preview-player"),
  ).find((button) => button.getAttribute("data-player-index") === "1");
  const enterButton = root.querySelector("[data-enter-table='true']");
  assert(myPlayerButton !== undefined);
  assert(enterButton !== null);
  assertEquals(myPlayerButton.hasAttribute("disabled"), false);
  assertEquals(enterButton.hasAttribute("disabled"), false);
  assertEquals(enterButton.textContent, "进入牌桌");
  assertEquals(enterButton.getAttribute("target"), "_blank");
  assertEquals(
    enterButton.getAttribute("href"),
    "/game/mine-game/player/1?user_id=test",
  );
  assertEquals(
    myPlayerButton.getAttribute("class")?.includes(
      "lobby-preview-player--mine",
    ),
    true,
  );
  assertEquals(
    root.querySelector(".lobby-preview__summary")?.textContent,
    "4/4 人 · 玩家 1",
  );

  myPlayerButton.dispatchEvent(new Event("click", { bubbles: true }));
  enterButton.dispatchEvent(new Event("click", { bubbles: true }));

  assertEquals(toggledPlayerIndex, 1);
  assertEquals(enteredGameId, "mine-game");
  assertEquals(enteredPlayerIndex, 1);
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
          userPlayers: [],
          players: [
            {
              index: 0,
              occupied: false,
              connected: false,
              mine: false,
              ready: false,
            },
            {
              index: 1,
              occupied: false,
              connected: false,
              mine: false,
              ready: false,
            },
            {
              index: 2,
              occupied: false,
              connected: false,
              mine: false,
              ready: false,
            },
            {
              index: 3,
              occupied: false,
              connected: false,
              mine: false,
              ready: false,
            },
          ],
        }],
        selectedGameId: "empty-game",
      }),
      callbacksStub({
        onEnterPlayer: () => {
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
          userPlayers: [2],
          players: [
            {
              index: 0,
              occupied: false,
              connected: false,
              mine: false,
              ready: false,
            },
            {
              index: 1,
              occupied: false,
              connected: false,
              mine: false,
              ready: false,
            },
            {
              index: 2,
              occupied: true,
              connected: true,
              mine: true,
              ready: false,
            },
            {
              index: 3,
              occupied: false,
              connected: false,
              mine: false,
              ready: false,
            },
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
    ["none", "ai", "auto"],
  );
  assertEquals(
    buttons.map((button) => button.textContent),
    ["不填充", "AI", "AUTO"],
  );
  const autoButton = buttons.find((button) =>
    button.getAttribute("data-bot-fill-mode") === "auto"
  );
  const aiButton = buttons.find((button) =>
    button.getAttribute("data-bot-fill-mode") === "ai"
  );
  assert(autoButton !== undefined);
  assert(aiButton !== undefined);
  assertEquals(autoButton.getAttribute("aria-pressed"), "true");

  aiButton.dispatchEvent(new Event("click", { bubbles: true }));

  assertEquals(selectedMode, "ai");
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
          userPlayers: [2],
          players: [
            {
              index: 0,
              occupied: true,
              connected: false,
              kind: "ai",
              mine: false,
              ready: true,
            },
            {
              index: 1,
              occupied: true,
              connected: false,
              kind: "auto",
              mine: false,
              ready: true,
            },
            {
              index: 2,
              occupied: true,
              connected: true,
              kind: "user",
              mine: true,
              ready: false,
            },
            {
              index: 3,
              occupied: false,
              connected: false,
              kind: "empty",
              mine: false,
              ready: false,
            },
          ],
        }],
        selectedGameId: "bot-filled-game",
      }),
      callbacksStub(),
    ),
  );

  const playerZero = root.querySelector("[data-player-index='0']");
  const playerOne = root.querySelector("[data-player-index='1']");
  assert(playerZero !== null);
  assert(playerOne !== null);
  assertEquals(playerZero.textContent, "0AI");
  assertEquals(playerOne.textContent, "1AUTO");
  assertEquals(
    root.querySelectorAll(".lobby-player-dot--filled").length,
    3,
  );
  assertEquals(
    playerZero.getAttribute("class")?.includes(
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
          userPlayers: [1],
          players: [
            {
              index: 0,
              occupied: true,
              connected: false,
              kind: "auto",
              mine: false,
              ready: true,
            },
            {
              index: 1,
              occupied: true,
              connected: false,
              kind: "user",
              mine: true,
              ready: false,
            },
            {
              index: 2,
              occupied: true,
              connected: false,
              kind: "auto",
              mine: false,
              ready: true,
            },
            {
              index: 3,
              occupied: true,
              connected: false,
              kind: "auto",
              mine: false,
              ready: true,
            },
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

  const aiButton = botModeButtons.find((button) =>
    button.getAttribute("data-bot-fill-mode") === "ai"
  );
  assert(aiButton !== undefined);
  aiButton.dispatchEvent(new Event("click", { bubbles: true }));

  assertEquals(selectedMode, null);
});
