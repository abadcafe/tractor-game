import {
  DEFAULT_VIEWER_PLAYER,
  type PlayerIndex,
  type TeamIndex,
} from "../config.ts";

export type PlayerDirection = "north" | "west" | "south" | "east";
export type PlayerPosition = "北" | "西" | "南" | "东";

export interface PlayerView {
  player: PlayerIndex;
  label: string;
  avatarText: string;
  position: PlayerPosition;
  direction: PlayerDirection;
  team: TeamIndex;
  teamLabel: string;
  isViewer: boolean;
}

export function playerView(
  player: PlayerIndex,
  viewerPlayer: PlayerIndex | null | undefined,
): PlayerView {
  const viewer = viewerPlayer ?? DEFAULT_VIEWER_PLAYER;
  const team = playerTeam(player);
  return {
    player,
    label: player === viewer ? `玩家 ${player} / 你` : `玩家 ${player}`,
    avatarText: String(player),
    position: positionFor(player, viewer),
    direction: directionFor(player, viewer),
    team,
    teamLabel: teamLabelForViewer(team, viewer),
    isViewer: player === viewer,
  };
}

export function teamLabelForViewer(
  team: number,
  viewerPlayer: PlayerIndex | null | undefined,
): string {
  const viewer = viewerPlayer ?? DEFAULT_VIEWER_PLAYER;
  return team === playerTeam(viewer) ? "我方" : "对方";
}

export function viewerTeam(
  viewerPlayer: PlayerIndex | null | undefined,
): TeamIndex {
  return playerTeam(viewerPlayer ?? DEFAULT_VIEWER_PLAYER);
}

function positionFor(
  player: PlayerIndex,
  viewerPlayer: PlayerIndex,
): PlayerPosition {
  switch (relativePlayer(player, viewerPlayer)) {
    case 0:
      return "南";
    case 1:
      return "东";
    case 2:
      return "北";
    case 3:
      return "西";
  }
  throw new Error("invalid relative player");
}

function directionFor(
  player: PlayerIndex,
  viewerPlayer: PlayerIndex,
): PlayerDirection {
  switch (relativePlayer(player, viewerPlayer)) {
    case 0:
      return "south";
    case 1:
      return "east";
    case 2:
      return "north";
    case 3:
      return "west";
  }
  throw new Error("invalid relative player");
}

function relativePlayer(
  player: PlayerIndex,
  viewerPlayer: PlayerIndex,
): number {
  return (player - viewerPlayer + 4) % 4;
}

function playerTeam(player: PlayerIndex): TeamIndex {
  return player % 2 === 0 ? 0 : 1;
}
