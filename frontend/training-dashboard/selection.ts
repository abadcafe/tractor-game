export interface RunSelectionSnapshot {
  readonly runDir: string;
  readonly revision: number;
}

export class DashboardSelection {
  #runDir = "";
  #revision = 0;

  get runDir(): string {
    return this.#runDir;
  }

  setRunDirectory(runDir: string): void {
    if (runDir === this.#runDir) return;
    this.#runDir = runDir;
    this.#revision += 1;
  }

  markRunReplaced(): void {
    this.#revision += 1;
  }

  captureRun(): RunSelectionSnapshot {
    return { runDir: this.#runDir, revision: this.#revision };
  }

  ownsRun(snapshot: RunSelectionSnapshot): boolean {
    return snapshot.runDir === this.#runDir &&
      snapshot.revision === this.#revision;
  }
}
