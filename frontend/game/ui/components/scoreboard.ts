import type { StateSnapshot } from "../../core/types.ts";
import { el } from "../dom.ts";
import {
  PLAYER_INDEXES,
  type PlayerIndex,
  type TeamIndex,
} from "../../config.ts";
import { playerView, teamLabelForViewer } from "../player-view.ts";
import type { ConnectionStatus } from "../types.ts";

const CONNECTION_LABELS: Record<ConnectionStatus, string> = {
  connecting: "连接中",
  connected: "已连接",
  failed: "连接失败",
};

/**
 * Render a scoreboard sidebar showing team levels and defender points.
 */
export function renderScoreboard(
  snapshot: StateSnapshot,
  viewerPlayer?: PlayerIndex | null,
  connectionStatus?: ConnectionStatus,
): HTMLElement {
  const scoreboard = el("div", { class: "scoreboard" });

  scoreboard.appendChild(renderScoreboardTitle(connectionStatus));

  const levelGrid = el("div", { class: "scoreboard__level-grid" });
  levelGrid.appendChild(
    renderTeamLevel(0, snapshot.team0_level, viewerPlayer),
  );
  levelGrid.appendChild(
    renderTeamLevel(1, snapshot.team1_level, viewerPlayer),
  );
  scoreboard.appendChild(levelGrid);

  scoreboard.appendChild(renderPlayerStatus(snapshot, viewerPlayer));
  scoreboard.appendChild(renderChatBox());

  return scoreboard;
}

function renderTeamLevel(
  team: TeamIndex,
  level: string,
  viewerPlayer?: PlayerIndex | null,
): HTMLElement {
  const teamEl = el("div", { class: `scoreboard__team team${team}` });
  teamEl.appendChild(
    el(
      "span",
      { class: `scoreboard__team-label team${team}` },
      teamLabelForViewer(team, viewerPlayer),
    ),
  );
  teamEl.appendChild(
    el("span", { class: "scoreboard__team-level" }, level),
  );
  return teamEl;
}

function renderPlayerStatus(
  snapshot: StateSnapshot,
  viewerPlayer?: PlayerIndex | null,
): HTMLElement {
  const players = el("div", { class: "scoreboard__section" });
  const playerList = el("div", { class: "scoreboard__player-list" });
  for (const playerIndex of PLAYER_INDEXES) {
    const view = playerView(playerIndex, viewerPlayer);
    const row = el("div", {
      class: `scoreboard__player-row team${view.team}`,
    });
    row.appendChild(
      el("span", { class: "scoreboard__player-name" }, view.label),
    );
    row.appendChild(
      el(
        "span",
        { class: "scoreboard__player-team" },
        view.teamLabel,
      ),
    );
    row.appendChild(
      el(
        "span",
        { class: "scoreboard__player-status" },
        playerStatus(snapshot, playerIndex),
      ),
    );
    playerList.appendChild(row);
  }
  players.appendChild(playerList);
  return players;
}

function renderScoreboardTitle(
  connectionStatus?: ConnectionStatus,
): HTMLElement {
  const title = el(
    "div",
    {
      class: "scoreboard__title scoreboard__title--with-status",
    },
    el("span", {}, "玩家"),
  );
  if (connectionStatus !== undefined) {
    title.appendChild(renderConnectionStatus(connectionStatus));
  }
  return title;
}

function renderConnectionStatus(status: ConnectionStatus): HTMLElement {
  return el(
    "span",
    { class: `scoreboard__connection ${status}` },
    CONNECTION_LABELS[status],
  );
}

function renderChatBox(): HTMLElement {
  const wrap = el("div", {
    class: "scoreboard__section scoreboard__chat",
  });
  wrap.appendChild(
    el("div", { class: "scoreboard__section-title" }, "聊天"),
  );
  const messages = el("div", { class: "scoreboard__chat-messages" });
  messages.appendChild(
    el("div", { class: "scoreboard__empty" }, "聊天功能待接入"),
  );
  wrap.appendChild(messages);
  wrap.appendChild(
    el("input", {
      class: "scoreboard__chat-input",
      type: "text",
      placeholder: "聊天功能待接入",
      disabled: "true",
    }),
  );
  return wrap;
}

function playerStatus(snapshot: StateSnapshot, player: number): string {
  const labels: string[] = [];
  if (snapshot.declarer_player === player) {
    labels.push("庄");
  }
  if (snapshot.phase === "WAITING") {
    labels.push(
      snapshot.next_round_confirmed.includes(player)
        ? "已确认"
        : "等待确认",
    );
  } else if (
    snapshot.phase === "PLAYING" &&
    snapshot.trick?.current_player === player
  ) {
    labels.push("待出牌");
  } else if (snapshot.phase === "STIRRING" && snapshot.stirring_state) {
    if (
      snapshot.stirring_state.phase === "EXCHANGING" &&
      snapshot.stirring_state.exchanging_player === player
    ) {
      labels.push("换底牌");
    } else if (
      snapshot.stirring_state.phase === "WAITING" &&
      snapshot.stirring_state.current_player === player
    ) {
      labels.push("待反主");
    }
  }
  return labels.length === 0 ? "在局" : labels.join(" / ");
}
