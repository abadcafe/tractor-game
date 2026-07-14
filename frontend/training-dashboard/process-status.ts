export type ProcessErrorSource = "control" | "process";

export type ProcessConnectionState =
  | "connecting"
  | "online"
  | "reconnecting";

export interface ProcessStatusSnapshot {
  readonly label: "CONNECTING" | "ERROR" | "ONLINE" | "RECONNECTING";
  readonly tone: "offline" | "online" | "pending";
  readonly message: string;
}

export class ProcessStatus {
  readonly #errors = new Map<ProcessErrorSource, string>();
  #connection: ProcessConnectionState = "connecting";
  #warning: string | null = null;

  setConnection(state: ProcessConnectionState): void {
    this.#connection = state;
  }

  reportError(source: ProcessErrorSource, message: string): void {
    this.#errors.set(source, message);
  }

  clearError(source: ProcessErrorSource): void {
    this.#errors.delete(source);
  }

  reportWarning(message: string): void {
    this.#warning = message;
  }

  reset(): void {
    this.#errors.clear();
    this.#connection = "connecting";
    this.#warning = null;
  }

  snapshot(): ProcessStatusSnapshot {
    const errors = [...this.#errors.values()];
    const message = [this.#warning, ...errors]
      .filter((value): value is string => value !== null)
      .join("; ");
    if (errors.length > 0) {
      return { label: "ERROR", tone: "offline", message };
    }
    if (this.#connection === "connecting") {
      return { label: "CONNECTING", tone: "pending", message };
    }
    if (this.#connection === "reconnecting") {
      return { label: "RECONNECTING", tone: "pending", message };
    }
    return { label: "ONLINE", tone: "online", message };
  }
}
