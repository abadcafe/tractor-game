import type { StateSnapshot } from "../../core/types.ts";
import { el } from "../dom.ts";
import { SEAT_MAP, TEAM_LABELS } from "../../config.ts";
import {
  cardDisplay,
  suitDisplayName,
  suitSymbol,
} from "../../core/card.ts";

/**
 * Render a scoreboard sidebar showing team levels and defender points.
 */
export function renderScoreboard(snapshot: StateSnapshot): HTMLElement {
  const scoreboard = el("div", { class: "scoreboard" });

  scoreboard.appendChild(
    el("div", { class: "scoreboard__title" }, "牌局信息"),
  );

  const levelGrid = el("div", { class: "scoreboard__level-grid" });
  levelGrid.appendChild(renderTeamLevel(0, snapshot.team0_level));
  levelGrid.appendChild(renderTeamLevel(1, snapshot.team1_level));
  scoreboard.appendChild(levelGrid);

  const trump = el("div", { class: "scoreboard__section" });
  trump.appendChild(
    el("div", { class: "scoreboard__section-title" }, "主牌"),
  );
  const trumpLine = el("div", { class: "scoreboard__trump-line" });
  trumpLine.appendChild(
    el("span", { class: "scoreboard__rank" }, snapshot.trump_rank),
  );
  if (snapshot.trump_suit) {
    const suit = el("span", {
      class: `scoreboard__suit suit-${snapshot.trump_suit}`,
    });
    suit.textContent = suitSymbol(snapshot.trump_suit);
    trumpLine.appendChild(suit);
    trumpLine.appendChild(
      el(
        "span",
        { class: "scoreboard__muted" },
        suitDisplayName(snapshot.trump_suit),
      ),
    );
  } else {
    trumpLine.appendChild(
      el("span", { class: "scoreboard__muted" }, "花色待定"),
    );
  }
  trump.appendChild(trumpLine);
  scoreboard.appendChild(trump);

  const score = el("div", {
    class: "scoreboard__section scoreboard__score-section",
  });
  score.appendChild(
    el("div", { class: "scoreboard__section-title" }, "防守方得分"),
  );
  score.appendChild(
    el(
      "div",
      { class: "scoreboard__points" },
      `${snapshot.defender_points}`,
    ),
  );
  scoreboard.appendChild(score);

  const seats = el("div", { class: "scoreboard__section" });
  seats.appendChild(
    el("div", { class: "scoreboard__section-title" }, "座位"),
  );
  const seatList = el("div", { class: "scoreboard__seat-list" });
  const confirmedSet = new Set(snapshot.next_round_confirmed);
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
        { class: "scoreboard__seat-cards" },
        `${snapshot.player_hand_counts[player] ?? 0} 张`,
      ),
    );
    if (snapshot.declarer_player === player) {
      row.appendChild(
        el("span", { class: "scoreboard__dealer" }, "庄"),
      );
    }
    if (snapshot.phase === "WAITING") {
      row.appendChild(
        el("span", { class: `scoreboard__ready ${confirmedSet.has(player) ? "ready" : "pending"}` },
          confirmedSet.has(player) ? "Ready" : "等待"),
      );
    }
    seatList.appendChild(row);
  }
  seats.appendChild(seatList);
  scoreboard.appendChild(seats);

  const log = el("div", {
    class: "scoreboard__section scoreboard__log",
  });
  log.appendChild(
    el("div", { class: "scoreboard__section-title" }, "叫牌记录"),
  );
  if (snapshot.bid_events.length === 0) {
    log.appendChild(
      el("div", { class: "scoreboard__empty" }, "暂无叫牌"),
    );
  } else {
    for (const event of snapshot.bid_events.slice(-6).reverse()) {
      const seat = SEAT_MAP[event.player];
      const cards = event.cards.map(cardDisplay).join(" ");
      log.appendChild(
        el(
          "div",
          { class: "scoreboard__log-row" },
          `${seat?.label ?? `玩家${event.player}`} ${cards}`,
        ),
      );
    }
  }
  scoreboard.appendChild(log);

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
