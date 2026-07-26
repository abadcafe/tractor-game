import { SEAT_IDS, type SeatId } from "../config.ts";
import type {
  BotFillMode,
  BotPolicyName,
  ListedGame,
  ListedSeat,
} from "../net/rest-client.ts";
import { el } from "./dom.ts";

export interface LobbyState {
  games: readonly ListedGame[];
  loading: boolean;
  creating: boolean;
  pendingSeatGameId: string | null;
  pendingSeatId: SeatId | null;
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
  onToggleSeat: (gameId: string, seatId: SeatId) => void;
  onEnterSeat: (gameId: string, seatId: SeatId) => void;
  enterSeatHref: (gameId: string, seatId: SeatId) => string;
  onChangeBotFillMode: (mode: BotFillMode) => void;
  onRefreshGames: () => void;
}

type PlayerDisplayKind = "empty" | "human" | BotPolicyName;

const TABLE_CAPACITY = 4;
const BOT_FILL_MODES: readonly {
  mode: BotFillMode;
  label: string;
}[] = [
  { mode: "none", label: "不填充" },
  { mode: "llm", label: "LLM" },
  { mode: "auto", label: "AUTO" },
];
const LOBBY_SEATS: readonly {
  seat: SeatId;
  label: string;
  area: string;
}[] = [
  { seat: "a", label: "A", area: "top" },
  { seat: "b", label: "B", area: "right" },
  { seat: "c", label: "C", area: "bottom" },
  { seat: "d", label: "D", area: "left" },
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
    state.pendingSeatGameId !== null || state.deletingGameId !== null;
  createButton.addEventListener("click", callbacks.onCreateGame);

  const refreshButton = el("button", {
    class: "lobby-button lobby-button--secondary",
    type: "button",
  }, state.loading ? "刷新中" : "刷新");
  refreshButton.disabled = state.loading ||
    state.pendingSeatGameId !== null || state.deletingGameId !== null;
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
  const busy = state.pendingSeatGameId !== null ||
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
  for (const seatId of SEAT_IDS) {
    const status = playerStatus(game, seatId);
    const className = status?.player !== null &&
        status?.player !== undefined
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
    highlightedGame?.seats.every((seat) => seat.player !== null) ??
      false;
  const hasEmptyPlayer =
    highlightedGame?.seats.some((seat) => seat.player === null) ??
      false;
  const pendingSelectedGame = highlightedGame !== null &&
    state.pendingSeatGameId === highlightedGame.gameId;
  const busy = state.pendingSeatGameId !== null ||
    state.deletingGameId !== null;
  const onPlayerClick = highlightedGame === null ||
      busy
    ? null
    : (seatId: SeatId) =>
      callbacks.onToggleSeat(highlightedGame.gameId, seatId);
  const onEnterClick = highlightedGame === null || myPlayer === null ||
      !allPlayersOccupied || busy
    ? null
    : () =>
      callbacks.onEnterSeat(highlightedGame.gameId, myPlayer.seat);
  const enterHref = highlightedGame === null || myPlayer === null ||
      !allPlayersOccupied || busy
    ? null
    : callbacks.enterSeatHref(highlightedGame.gameId, myPlayer.seat);
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
      ...LOBBY_SEATS.map((seat) =>
        renderPreviewPlayer(
          seat,
          highlightedGame,
          pendingSelectedGame ? state.pendingSeatId : null,
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
        : `${occupiedCount}/${capacity} 人 · 玩家 ${myPlayer.seat}`,
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
  seat: { seat: SeatId; label: string; area: string },
  game: ListedGame | null,
  joiningSeatId: SeatId | null,
  onPlayerClick: ((seatId: SeatId) => void) | null,
): HTMLButtonElement {
  const status = game === null ? null : playerStatus(game, seat.seat);
  const occupied = status?.player !== null &&
    status?.player !== undefined;
  const mine = status?.player?.kind === "human" &&
    status.player.mine;
  const kind = playerKind(status);
  const pending = joiningSeatId === seat.seat;
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
      "data-seat": seat.seat,
      "data-seat-area": seat.area,
      type: "button",
    },
    el(
      "span",
      { class: "lobby-preview-player__label" },
      seat.label,
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
      () => onPlayerClick(seat.seat),
    );
  }
  return playerButton;
}

function previewPlayerClassName(
  selected: boolean,
  mine: boolean,
  pending: boolean,
  kind: PlayerDisplayKind,
): string {
  const classes = ["lobby-preview-player"];
  if (selected) {
    classes.push("lobby-preview-player--filled");
  }
  if (kind === "llm" || kind === "auto") {
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

function playerKind(status: ListedSeat | null): PlayerDisplayKind {
  if (status?.player === null || status === null) {
    return "empty";
  }
  if (status.player.kind === "human") {
    return "human";
  }
  return status.player.policy;
}

function previewPlayerStatusText(
  kind: PlayerDisplayKind,
  mine: boolean,
  pending: boolean,
): string | null {
  if (mine || pending) {
    return "你";
  }
  if (kind === "llm") {
    return "LLM";
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
  return game.seats.filter((seat) => seat.player !== null).length;
}

function playerStatus(
  game: ListedGame,
  seatId: SeatId,
): ListedSeat | null {
  return game.seats.find((player) => player.seat === seatId) ??
    null;
}

function currentMinePlayer(game: ListedGame): ListedSeat | null {
  return game.seats.find((seat) =>
    seat.player?.kind === "human" && seat.player.mine
  ) ?? null;
}

function shortGameId(gameId: string): string {
  return gameId.slice(0, 8);
}
