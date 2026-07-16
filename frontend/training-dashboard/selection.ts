export interface RunSelectionSnapshot {
  readonly runDir: string;
  readonly generation: number;
}

export class DashboardSelection {
  #runDir = "";
  #generation = 0;

  get runDir(): string {
    return this.#runDir;
  }

  setRunDirectory(runDir: string): void {
    if (runDir === this.#runDir) return;
    this.#runDir = runDir;
    this.#generation += 1;
  }

  markRunReplaced(): void {
    this.#generation += 1;
  }

  captureRun(): RunSelectionSnapshot {
    return { runDir: this.#runDir, generation: this.#generation };
  }

  ownsRun(snapshot: RunSelectionSnapshot): boolean {
    return snapshot.runDir === this.#runDir &&
      snapshot.generation === this.#generation;
  }
}
