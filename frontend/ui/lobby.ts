import { PLAYER_INDEXES, type PlayerIndex } from "../config.ts";
import type {
  BotFillMode,
  ListedGame,
  ListedPlayer,
  PlayerKind,
} from "../net/rest-client.ts";
import { el } from "./dom.ts";

export interface LobbyState {
  games: readonly ListedGame[];
  loading: boolean;
  creating: boolean;
  pendingPlayerGameId: string | null;
  pendingPlayerIndex: PlayerIndex | null;
  deletingGameId: string | null;
  selectedGameId: string | null;
  botFillMode: BotFillMode;
  errorMessage: string | null;
  statusMessage: string | null;
}

export interface LobbyCallbacks {
  onCreateGame: () => void;
  onSelectGame: (gameId: string) => void;
  onDeleteGame: (gameId: string) => void;
  onTogglePlayer: (gameId: string, playerIndex: PlayerIndex) => void;
  onEnterPlayer: (gameId: string, playerIndex: PlayerIndex) => void;
  enterPlayerHref: (gameId: string, playerIndex: PlayerIndex) => string;
  onChangeBotFillMode: (mode: BotFillMode) => void;
  onRefreshGames: () => void;
}

const TABLE_CAPACITY = 4;
const BOT_FILL_MODES: readonly {
  mode: BotFillMode;
  label: string;
}[] = [
  { mode: "none", label: "不填充" },
  { mode: "ai", label: "AI" },
  { mode: "auto", label: "AUTO" },
];
const LOBBY_PLAYERS: readonly {
  index: PlayerIndex;
  label: string;
  area: string;
}[] = [
  { index: 0, label: "0", area: "north" },
  { index: 1, label: "1", area: "west" },
  { index: 2, label: "2", area: "south" },
  { index: 3, label: "3", area: "east" },
];

export function renderLobby(
  state: LobbyState,
  callbacks: LobbyCallbacks,
): HTMLElement {
  const shell = el("div", { class: "lobby" });
  shell.appendChild(renderLobbyHeader(state, callbacks));
  shell.appendChild(renderLobbyBody(state, callbacks));
  return shell;
}

function renderLobbyHeader(
  state: LobbyState,
  callbacks: LobbyCallbacks,
): HTMLElement {
  const createButton = el("button", {
    class: "lobby-button lobby-button--primary",
    type: "button",
  }, state.creating ? "创建中" : "创建牌局");
  createButton.disabled = state.loading || state.creating ||
    state.pendingPlayerGameId !== null || state.deletingGameId !== null;
  createButton.addEventListener("click", callbacks.onCreateGame);

  const refreshButton = el("button", {
    class: "lobby-button lobby-button--secondary",
    type: "button",
  }, state.loading ? "刷新中" : "刷新");
  refreshButton.disabled = state.loading ||
    state.pendingPlayerGameId !== null || state.deletingGameId !== null;
  refreshButton.addEventListener("click", callbacks.onRefreshGames);

  return el(
    "header",
    { class: "lobby-header" },
    el(
      "div",
      { class: "lobby-title-block" },
      el("div", { class: "lobby-kicker" }, "TRACTOR"),
      el("h1", { class: "lobby-title" }, "游戏大厅"),
    ),
    renderLobbyMetrics(state.games),
    el(
      "div",
      { class: "lobby-actions" },
      refreshButton,
      createButton,
    ),
  );
}

function renderLobbyMetrics(games: readonly ListedGame[]): HTMLElement {
  const activeUsers = games.reduce(
    (sum, game) => sum + boundedUserCount(game),
    0,
  );
  const totalPlayers = games.reduce(
    (sum, game) => sum + game.capacity,
    0,
  );
  return el(
    "div",
    { class: "lobby-metrics" },
    renderMetric("牌局", String(games.length)),
    renderMetric("玩家", `${activeUsers}/${totalPlayers}`),
  );
}

function renderMetric(label: string, value: string): HTMLElement {
  return el(
    "div",
    { class: "lobby-metric" },
    el("span", { class: "lobby-metric__label" }, label),
    el("strong", { class: "lobby-metric__value" }, value),
  );
}

function renderLobbyBody(
  state: LobbyState,
  callbacks: LobbyCallbacks,
): HTMLElement {
  return el(
    "main",
    { class: "lobby-body" },
    el(
      "section",
      { class: "lobby-panel lobby-panel--games" },
      renderGameListHeader(state),
      renderGameList(state, callbacks),
    ),
    el(
      "aside",
      { class: "lobby-panel lobby-panel--preview" },
      renderTablePreview(state, callbacks),
    ),
  );
}

function renderGameListHeader(state: LobbyState): HTMLElement {
  const statusText = state.errorMessage ?? state.statusMessage ?? "";
  return el(
    "div",
    { class: "lobby-section-head" },
    el(
      "div",
      {},
      el("h2", { class: "lobby-section-title" }, "当前牌局"),
      el("p", { class: "lobby-section-status" }, statusText),
    ),
  );
}

function renderGameList(
  state: LobbyState,
  callbacks: LobbyCallbacks,
): HTMLElement {
  if (state.loading && state.games.length === 0) {
    return el(
      "div",
      { class: "lobby-empty" },
      el("div", { class: "lobby-empty__title" }, "正在加载牌局"),
    );
  }
  if (state.games.length === 0) {
    return el(
      "div",
      { class: "lobby-empty" },
      el("div", { class: "lobby-empty__title" }, "没有可加入的牌局"),
    );
  }
  const list = el("div", { class: "lobby-game-list" });
  for (const game of state.games) {
    list.appendChild(renderGameRow(game, state, callbacks));
  }
  return list;
}

function renderGameRow(
  game: ListedGame,
  state: LobbyState,
  callbacks: LobbyCallbacks,
): HTMLElement {
  const isSelected = state.selectedGameId === game.gameId;
  const busy = state.pendingPlayerGameId !== null ||
    state.deletingGameId !== null;
  const deleting = state.deletingGameId === game.gameId;
  const rowClass = isSelected
    ? "lobby-game-row lobby-game-row--selected"
    : "lobby-game-row";
  const row = el(
    "button",
    {
      class: rowClass,
      type: "button",
      "aria-pressed": isSelected ? "true" : "false",
    },
    el(
      "div",
      { class: "lobby-game-row__main" },
      el(
        "div",
        { class: "lobby-game-row__name" },
        `牌局 ${shortGameId(game.gameId)}`,
      ),
      el(
        "div",
        { class: "lobby-game-row__id" },
        game.gameId,
      ),
    ),
    el(
      "div",
      { class: "lobby-game-row__players" },
      renderPlayerDots(game),
    ),
  );
  row.disabled = busy;
  row.addEventListener(
    "click",
    () => callbacks.onSelectGame(game.gameId),
  );

  const deleteButton = el("button", {
    class: "lobby-game-row__delete",
    type: "button",
    "aria-label": `删除牌局 ${shortGameId(game.gameId)}`,
  }, deleting ? "删除中" : "删除");
  deleteButton.disabled = busy || state.loading || state.creating;
  deleteButton.addEventListener(
    "click",
    (event) => {
      event.stopPropagation();
      callbacks.onDeleteGame(game.gameId);
    },
  );

  return el(
    "div",
    { class: "lobby-game-row-shell" },
    row,
    deleteButton,
  );
}

function renderPlayerDots(game: ListedGame): HTMLElement {
  const players = el("div", { class: "lobby-player-dots" });
  for (const playerIndex of PLAYER_INDEXES) {
    const status = playerStatus(game, playerIndex);
    const className = status?.occupied === true
      ? "lobby-player-dot lobby-player-dot--filled"
      : "lobby-player-dot";
    players.appendChild(el("span", { class: className }));
  }
  return players;
}

function renderTablePreview(
  state: LobbyState,
  callbacks: LobbyCallbacks,
): HTMLElement {
  const highlightedGame =
    state.games.find((game) => game.gameId === state.selectedGameId) ??
      null;
  const occupiedCount = highlightedGame === null
    ? 0
    : occupiedPlayerCount(highlightedGame);
  const capacity = highlightedGame?.capacity ?? TABLE_CAPACITY;
  const myPlayer = highlightedGame === null ? null : currentMinePlayer(
    highlightedGame,
  );
  const allPlayersOccupied =
    highlightedGame?.players.every((player) => player.occupied) ??
      false;
  const hasEmptyPlayer =
    highlightedGame?.players.some((player) => !player.occupied) ??
      false;
  const pendingSelectedGame = highlightedGame !== null &&
    state.pendingPlayerGameId === highlightedGame.gameId;
  const busy = state.pendingPlayerGameId !== null ||
    state.deletingGameId !== null;
  const onPlayerClick = highlightedGame === null ||
      busy
    ? null
    : (playerIndex: PlayerIndex) =>
      callbacks.onTogglePlayer(highlightedGame.gameId, playerIndex);
  const onEnterClick = highlightedGame === null || myPlayer === null ||
      !allPlayersOccupied || busy
    ? null
    : () =>
      callbacks.onEnterPlayer(highlightedGame.gameId, myPlayer.index);
  const enterHref = highlightedGame === null || myPlayer === null ||
      !allPlayersOccupied || busy
    ? null
    : callbacks.enterPlayerHref(highlightedGame.gameId, myPlayer.index);
  return el(
    "div",
    { class: "lobby-preview" },
    el(
      "div",
      { class: "lobby-preview__head" },
      el("h2", { class: "lobby-section-title" }, "玩家"),
      renderBotFillControl(
        state.botFillMode,
        highlightedGame === null || myPlayer === null ||
          busy || !hasEmptyPlayer,
        callbacks,
      ),
    ),
    el(
      "div",
      { class: "lobby-table-preview" },
      ...LOBBY_PLAYERS.map((player) =>
        renderPreviewPlayer(
          player,
          highlightedGame,
          pendingSelectedGame ? state.pendingPlayerIndex : null,
          onPlayerClick,
        )
      ),
      renderEnterTableButton(onEnterClick, enterHref),
    ),
    el(
      "div",
      { class: "lobby-preview__summary" },
      highlightedGame === null
        ? "未选择牌局"
        : myPlayer === null
        ? `${occupiedCount}/${capacity} 人`
        : `${occupiedCount}/${capacity} 人 · 玩家 ${myPlayer.index}`,
    ),
  );
}

function renderEnterTableButton(
  onEnterClick: (() => void) | null,
  href: string | null,
): HTMLElement {
  if (onEnterClick !== null && href !== null) {
    const link = el(
      "a",
      {
        class: "lobby-table-preview__felt",
        href,
        target: "_blank",
        rel: "noopener noreferrer",
        "data-enter-table": "true",
      },
      "进入牌桌",
    );
    link.addEventListener("click", onEnterClick);
    return link;
  }

  const button = el(
    "button",
    {
      class: "lobby-table-preview__felt",
      type: "button",
      "data-enter-table": "true",
    },
    "进入牌桌",
  );
  setButtonDisabled(button, true);
  return button;
}

function renderPreviewPlayer(
  player: { index: PlayerIndex; label: string; area: string },
  game: ListedGame | null,
  joiningPlayerIndex: PlayerIndex | null,
  onPlayerClick: ((playerIndex: PlayerIndex) => void) | null,
): HTMLButtonElement {
  const status = game === null
    ? null
    : playerStatus(game, player.index);
  const occupied = status?.occupied === true;
  const mine = status?.mine === true;
  const kind = playerKind(status);
  const pending = joiningPlayerIndex === player.index;
  const selected = pending || occupied;
  const className = previewPlayerClassName(
    selected,
    mine,
    pending,
    kind,
  );
  const playerButton = el(
    "button",
    {
      class: className,
      "data-player-index": String(player.index),
      "data-player-area": player.area,
      type: "button",
    },
    el(
      "span",
      { class: "lobby-preview-player__label" },
      player.label,
    ),
  );
  const statusText = previewPlayerStatusText(kind, mine, pending);
  if (statusText !== null) {
    playerButton.appendChild(
      el("span", { class: "lobby-preview-player__status" }, statusText),
    );
  }
  setButtonDisabled(
    playerButton,
    game === null || (occupied && !mine) || onPlayerClick === null,
  );
  if (onPlayerClick !== null) {
    playerButton.addEventListener(
      "click",
      () => onPlayerClick(player.index),
    );
  }
  return playerButton;
}

function previewPlayerClassName(
  selected: boolean,
  mine: boolean,
  pending: boolean,
  kind: PlayerKind,
): string {
  const classes = ["lobby-preview-player"];
  if (selected) {
    classes.push("lobby-preview-player--filled");
  }
  if (kind === "ai" || kind === "auto") {
    classes.push("lobby-preview-player--bot");
  }
  if (mine) {
    classes.push("lobby-preview-player--mine");
  }
  if (pending) {
    classes.push("lobby-preview-player--pending");
  }
  return classes.join(" ");
}

function renderBotFillControl(
  selectedMode: BotFillMode,
  disabled: boolean,
  callbacks: LobbyCallbacks,
): HTMLElement {
  const group = el("div", {
    class: "lobby-bot-mode",
    role: "group",
    "aria-label": "bot填充",
  });
  for (const option of BOT_FILL_MODES) {
    const selected = selectedMode === option.mode;
    const button = el(
      "button",
      {
        class: selected
          ? "lobby-bot-mode__button lobby-bot-mode__button--selected"
          : "lobby-bot-mode__button",
        type: "button",
        "aria-pressed": selected ? "true" : "false",
        "data-bot-fill-mode": option.mode,
      },
      option.label,
    );
    setButtonDisabled(button, disabled);
    if (!disabled) {
      button.addEventListener(
        "click",
        () => callbacks.onChangeBotFillMode(option.mode),
      );
    }
    group.appendChild(button);
  }
  return group;
}

function playerKind(status: ListedPlayer | null): PlayerKind {
  if (status === null) {
    return "empty";
  }
  return status.kind ?? (status.occupied ? "user" : "empty");
}

function previewPlayerStatusText(
  kind: PlayerKind,
  mine: boolean,
  pending: boolean,
): string | null {
  if (mine || pending) {
    return "你";
  }
  if (kind === "ai") {
    return "AI";
  }
  if (kind === "auto") {
    return "AUTO";
  }
  return null;
}

function setButtonDisabled(
  button: HTMLButtonElement,
  disabled: boolean,
): void {
  button.disabled = disabled;
  if (disabled) {
    button.setAttribute("disabled", "");
  } else {
    button.removeAttribute("disabled");
  }
}

function boundedUserCount(game: ListedGame): number {
  return Math.min(game.userCount, game.capacity);
}

function occupiedPlayerCount(game: ListedGame): number {
  return game.players.filter((player) => player.occupied).length;
}

function playerStatus(
  game: ListedGame,
  playerIndex: PlayerIndex,
): ListedPlayer | null {
  return game.players.find((player) => player.index === playerIndex) ??
    null;
}

function currentMinePlayer(game: ListedGame): ListedPlayer | null {
  return game.players.find((player) => player.mine) ?? null;
}

function shortGameId(gameId: string): string {
  return gameId.slice(0, 8);
}
