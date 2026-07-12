export type SlotKind =
  | "api_request"
  | "api_response"
  | "api_error"
  | "tool_result";

export interface TranscriptRecord {
  id: number;
  event_id: number;
  created_at: string;
  player_index: number;
  seq: number;
  attempt: number;
  api_request: string | null;
  api_response: string | null;
  api_error: string | null;
  tool_result: string | null;
}

export interface ViewState {
  knownCount: number;
  newCount: number;
  stickToBottom: boolean;
}

export interface SlotDefinition {
  kind: SlotKind;
  title: string;
}

export type JsonParseResult =
  | { ok: true; value: unknown }
  | { ok: false };

export type KvEntry = readonly [label: string, value: unknown];
export type Path = readonly string[];
