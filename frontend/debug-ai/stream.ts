import { parseJson, transcriptRecord } from "./json.ts";
import type { TranscriptRecord } from "./types.ts";

export class AITranscriptStream {
  #socket: WebSocket | null = null;
  #generation = 0;

  open(
    gameId: string,
    player: number,
    onRecord: (record: TranscriptRecord) => void,
  ): void {
    const generation = this.#generation + 1;
    this.#generation = generation;
    this.close();
    const protocol = globalThis.location.protocol === "https:"
      ? "wss:"
      : "ws:";
    const url = `${protocol}//${globalThis.location.host}/ws/debug/ai/${
      encodeURIComponent(gameId)
    }?player=${player}`;
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
