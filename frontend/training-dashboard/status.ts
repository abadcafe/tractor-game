export type DashboardErrorSource =
  | "config"
  | "control"
  | "directory"
  | "metrics"
  | "process"
  | "stream"
  | "checkpoints"
  | "logs";

export type StreamConnectionState =
  | "connecting"
  | "online"
  | "reconnecting";

export interface DashboardStatusSnapshot {
  readonly label:
    | "CONNECTING"
    | "ERROR"
    | "ONLINE"
    | "RECONNECTING"
    | "REFRESHING";
  readonly tone: "offline" | "online" | "pending";
  readonly message: string;
}

export class DashboardStatus {
  readonly #errors = new Map<DashboardErrorSource, string>();
  #streamConnection: StreamConnectionState = "connecting";
  #refreshPending = false;
  #warning: string | null = null;

  setStreamConnection(state: StreamConnectionState): void {
    this.#streamConnection = state;
  }

  setRefreshPending(): void {
    this.#refreshPending = true;
  }

  setRefreshIdle(): void {
    this.#refreshPending = false;
  }

  reportError(source: DashboardErrorSource, message: string): void {
    this.#errors.set(source, message);
  }

  clearError(source: DashboardErrorSource): void {
    this.#errors.delete(source);
  }

  reportWarning(message: string): void {
    this.#warning = message;
  }

  reset(): void {
    this.#errors.clear();
    this.#streamConnection = "connecting";
    this.#refreshPending = false;
    this.#warning = null;
  }

  snapshot(): DashboardStatusSnapshot {
    const errors = [...this.#errors.values()];
    const message = [this.#warning, ...errors]
      .filter((value): value is string => value !== null)
      .join("; ");
    if (errors.length > 0) {
      return { label: "ERROR", tone: "offline", message };
    }
    if (this.#streamConnection === "connecting") {
      return { label: "CONNECTING", tone: "pending", message };
    }
    if (this.#streamConnection === "reconnecting") {
      return { label: "RECONNECTING", tone: "pending", message };
    }
    if (this.#refreshPending) {
      return { label: "REFRESHING", tone: "pending", message };
    }
    return { label: "ONLINE", tone: "online", message };
  }
}
