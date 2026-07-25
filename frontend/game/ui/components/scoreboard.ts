import type { StateSnapshot } from "../../core/types.ts";
import { el } from "../dom.ts";
import {
  type PartnershipId,
  SEAT_IDS,
  type SeatId,
} from "../../config.ts";
import { partnershipLabelForViewer, seatView } from "../seat-view.ts";
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
  viewerSeat: SeatId,
  connectionStatus?: ConnectionStatus,
): HTMLElement {
  const scoreboard = el("div", { class: "scoreboard" });

  scoreboard.appendChild(renderScoreboardTitle(connectionStatus));

  const levelGrid = el("div", { class: "scoreboard__level-grid" });
  levelGrid.appendChild(
    renderPartnershipLevel(
      "first",
      snapshot.partnership_levels.first,
      viewerSeat,
    ),
  );
  levelGrid.appendChild(
    renderPartnershipLevel(
      "second",
      snapshot.partnership_levels.second,
      viewerSeat,
    ),
  );
  scoreboard.appendChild(levelGrid);

  scoreboard.appendChild(renderPlayerStatus(snapshot, viewerSeat));
  scoreboard.appendChild(renderChatBox());

  return scoreboard;
}

function renderPartnershipLevel(
  partnership: PartnershipId,
  level: string,
  viewerSeat: SeatId,
): HTMLElement {
  const partnershipEl = el("div", {
    class: `scoreboard__team partnership-${partnership}`,
  });
  partnershipEl.appendChild(
    el(
      "span",
      {
        class: `scoreboard__team-label partnership-${partnership}`,
      },
      partnershipLabelForViewer(partnership, viewerSeat),
    ),
  );
  partnershipEl.appendChild(
    el("span", { class: "scoreboard__team-level" }, level),
  );
  return partnershipEl;
}

function renderPlayerStatus(
  snapshot: StateSnapshot,
  viewerSeat: SeatId,
): HTMLElement {
  const players = el("div", { class: "scoreboard__section" });
  const playerList = el("div", { class: "scoreboard__player-list" });
  for (const seatId of SEAT_IDS) {
    const view = seatView(seatId, viewerSeat);
    const row = el("div", {
      class: `scoreboard__player-row partnership-${view.partnership}`,
    });
    row.appendChild(
      el("span", { class: "scoreboard__player-name" }, view.label),
    );
    row.appendChild(
      el(
        "span",
        { class: "scoreboard__player-team" },
        view.partnershipLabel,
      ),
    );
    row.appendChild(
      el(
        "span",
        { class: "scoreboard__player-status" },
        playerStatus(snapshot, seatId),
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

function playerStatus(
  snapshot: StateSnapshot,
  seat: SeatId,
): string {
  const labels: string[] = [];
  if (snapshot.declarer === seat) {
    labels.push("庄");
  }
  if (snapshot.phase === "WAITING") {
    labels.push(
      snapshot.next_round_confirmed.includes(seat)
        ? "已确认"
        : "等待确认",
    );
  } else if (
    snapshot.phase === "PLAYING" &&
    snapshot.trick?.current_actor === seat
  ) {
    labels.push("待出牌");
  } else if (snapshot.phase === "STIRRING" && snapshot.stirring_state) {
    if (
      snapshot.stirring_state.phase === "EXCHANGING" &&
      snapshot.stirring_state.exchanging_actor === seat
    ) {
      labels.push("换底牌");
    } else if (
      snapshot.stirring_state.phase === "WAITING" &&
      snapshot.stirring_state.current_actor === seat
    ) {
      labels.push("待反主");
    }
  }
  return labels.length === 0 ? "在局" : labels.join(" / ");
}
