export interface RunSelectionSnapshot {
  readonly runDir: string;
  readonly revision: number;
}

export interface MetricSelectionSnapshot extends RunSelectionSnapshot {
  readonly sessionId: string | null;
  readonly metricRevision: number;
}

export class DashboardSelection {
  #runDir = "";
  #sessionId: string | null = null;
  #runRevision = 0;
  #metricRevision = 0;

  get runDir(): string {
    return this.#runDir;
  }

  get metricSession(): string | null {
    return this.#sessionId;
  }

  setRunDirectory(runDir: string): void {
    if (runDir === this.#runDir) return;
    this.#runDir = runDir;
    this.#sessionId = null;
    this.#runRevision += 1;
    this.#metricRevision += 1;
  }

  setMetricSession(sessionId: string | null): void {
    if (sessionId === this.#sessionId) return;
    this.#sessionId = sessionId;
    this.#metricRevision += 1;
  }

  markRunReplaced(): void {
    this.#sessionId = null;
    this.#runRevision += 1;
    this.#metricRevision += 1;
  }

  captureRun(): RunSelectionSnapshot {
    return { runDir: this.#runDir, revision: this.#runRevision };
  }

  captureMetrics(): MetricSelectionSnapshot {
    return {
      ...this.captureRun(),
      sessionId: this.#sessionId,
      metricRevision: this.#metricRevision,
    };
  }

  ownsRun(snapshot: RunSelectionSnapshot): boolean {
    return snapshot.runDir === this.#runDir &&
      snapshot.revision === this.#runRevision;
  }

  ownsMetrics(snapshot: MetricSelectionSnapshot): boolean {
    return this.ownsRun(snapshot) &&
      snapshot.sessionId === this.#sessionId &&
      snapshot.metricRevision === this.#metricRevision;
  }
}
