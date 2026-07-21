import { EventStreamConnection } from "./event-source.ts";
import { parseMetrics, type TrainingMetrics } from "./types.ts";

export interface MetricEventTarget {
  readonly runDir: string;
  readonly updateLimit: number;
  readonly seriesPoints: number;
}

export interface MetricEventHandlers {
  readonly onSnapshot: (value: TrainingMetrics) => void;
  readonly onError: (message: string) => void;
}

export class MetricEventStream {
  readonly #connection: EventStreamConnection;

  constructor(
    private readonly target: () => MetricEventTarget | null,
    handlers: MetricEventHandlers,
  ) {
    this.#connection = new EventStreamConnection({
      onError: handlers.onError,
    });
    this.handlers = handlers;
  }

  private readonly handlers: MetricEventHandlers;

  connect(): void {
    this.disconnect();
    const target = this.target();
    if (target === null) return;
    this.#connection.connect(metricEventUrl(target), [
      {
        name: "metrics",
        receive: (value) =>
          this.handlers.onSnapshot(parseMetrics(value)),
      },
    ]);
  }

  disconnect(): void {
    this.#connection.disconnect();
  }
}

export function metricEventUrl(target: MetricEventTarget): string {
  const search = new URLSearchParams({
    run_dir: target.runDir,
    update_limit: String(target.updateLimit),
    series_points: String(target.seriesPoints),
  });
  return `/api/training/events/metrics?${search}`;
}
