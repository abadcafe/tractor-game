import type { StateSnapshot } from "../../core/types.ts";
import { el } from "../dom.ts";
import { SEAT_MAP, TEAM_LABELS } from "../../config.ts";

/**
 * Render a scoreboard sidebar showing team levels and defender points.
 */
export function renderScoreboard(snapshot: StateSnapshot): HTMLElement {
  const scoreboard = el("div", { class: "scoreboard" });

  scoreboard.appendChild(
    el("div", { class: "scoreboard__title" }, "玩家"),
  );

  const levelGrid = el("div", { class: "scoreboard__level-grid" });
  levelGrid.appendChild(renderTeamLevel(0, snapshot.team0_level));
  levelGrid.appendChild(renderTeamLevel(1, snapshot.team1_level));
  scoreboard.appendChild(levelGrid);

  scoreboard.appendChild(renderPlayerStatus(snapshot));
  scoreboard.appendChild(renderChatBox());

  return scoreboard;
}

function renderTeamLevel(team: number, level: string): HTMLElement {
  const teamEl = el("div", { class: `scoreboard__team team${team}` });
  teamEl.appendChild(
    el(
      "span",
      { class: `scoreboard__team-label team${team}` },
      TEAM_LABELS[team],
    ),
  );
  teamEl.appendChild(
    el("span", { class: "scoreboard__team-level" }, level),
  );
  return teamEl;
}

function renderPlayerStatus(snapshot: StateSnapshot): HTMLElement {
  const seats = el("div", { class: "scoreboard__section" });
  seats.appendChild(
    el("div", { class: "scoreboard__section-title" }, "玩家"),
  );
  const seatList = el("div", { class: "scoreboard__seat-list" });
  for (const player of [0, 1, 2, 3]) {
    const seat = SEAT_MAP[player];
    const row = el("div", {
      class: `scoreboard__seat-row team${seat.team}`,
    });
    row.appendChild(
      el("span", { class: "scoreboard__seat-name" }, seat.label),
    );
    row.appendChild(
      el(
        "span",
        { class: "scoreboard__seat-team" },
        TEAM_LABELS[seat.team],
      ),
    );
    row.appendChild(
      el(
        "span",
        { class: "scoreboard__seat-status" },
        playerStatus(snapshot, player),
      ),
    );
    seatList.appendChild(row);
  }
  seats.appendChild(seatList);
  return seats;
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
