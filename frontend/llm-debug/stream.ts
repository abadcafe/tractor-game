import { parseJson, transcriptRecord } from "./json.ts";
import type { SeatId, TranscriptRecord } from "./types.ts";

export class LLMTranscriptStream {
  #socket: WebSocket | null = null;
  #generation = 0;

  open(
    gameId: string,
    seat: SeatId,
    onRecord: (record: TranscriptRecord) => void,
  ): void {
    const generation = this.#generation + 1;
    this.#generation = generation;
    this.close();
    const protocol = globalThis.location.protocol === "https:"
      ? "wss:"
      : "ws:";
    const url =
      `${protocol}//${globalThis.location.host}/ws/debug/llm/${
        encodeURIComponent(gameId)
      }?seat=${seat}`;
    this.#socket = new WebSocket(url);
    this.#socket.addEventListener(
      "message",
      (event: MessageEvent<string>) => {
        if (generation !== this.#generation) return;
        const parsed = parseJson(event.data);
        if (!parsed.ok) return;
        const record = transcriptRecord(parsed.value);
        if (record === null) return;
        onRecord(record);
      },
    );
    this.#socket.addEventListener("close", () => {
      if (generation !== this.#generation) return;
      this.#socket = null;
    });
  }

  close(): void {
    if (this.#socket !== null) {
      this.#socket.close();
      this.#socket = null;
    }
  }
}
