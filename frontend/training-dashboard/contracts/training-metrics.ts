import { TRAINING_METRICS_SCHEMA_VERSION } from "./generated.ts";
import { type JsonPrimitive, primitiveRecord } from "./json.ts";
import { nullableStoreId } from "./event-store.ts";
import {
  nonNegativeInteger,
  nonNegativeNumber,
  nullableNonNegativeInteger,
  positiveInteger,
  rejectUnknownKeys,
  requiredArray,
  requiredBoolean,
  requiredRecord,
} from "./wire.ts";

export interface MetricPoint {
  readonly sequence: number;
  readonly update: number | null;
  readonly elapsed_seconds: number;
  readonly recorded_at_ms: number;
  readonly values: Readonly<Record<string, JsonPrimitive>>;
}

export interface MetricDatasets {
  readonly throughput: readonly MetricPoint[];
  readonly optimization: readonly MetricPoint[];
  readonly ppo_timing: readonly MetricPoint[];
  readonly rollout: readonly MetricPoint[];
  readonly rewards: readonly MetricPoint[];
  readonly inference: readonly MetricPoint[];
  readonly processes: readonly MetricPoint[];
}

export interface TrainingMetrics {
  readonly schema_version: typeof TRAINING_METRICS_SCHEMA_VERSION;
  readonly store_id: string | null;
  readonly through_sequence: number;
  readonly complete: boolean;
  readonly dropped_event_count: number;
  readonly totals: Readonly<Record<string, JsonPrimitive>>;
  readonly datasets: MetricDatasets;
}

export function parseMetrics(value: unknown): TrainingMetrics {
  const record = requiredRecord(value, "training metrics");
  rejectUnknownKeys(
    record,
    [
      "schema_version",
      "store_id",
      "through_sequence",
      "complete",
      "dropped_event_count",
      "totals",
      "datasets",
    ],
    "training metrics",
  );
  if (record.schema_version !== TRAINING_METRICS_SCHEMA_VERSION) {
    throw new Error("Unsupported training metrics schema");
  }
  const datasets = requiredRecord(record.datasets, "metric datasets");
  rejectUnknownKeys(
    datasets,
    [
      "throughput",
      "optimization",
      "ppo_timing",
      "rollout",
      "rewards",
      "inference",
      "processes",
    ],
    "metric datasets",
  );
  return {
    schema_version: TRAINING_METRICS_SCHEMA_VERSION,
    store_id: nullableStoreId(record.store_id),
    through_sequence: nonNegativeInteger(
      record.through_sequence,
      "through_sequence",
    ),
    complete: requiredBoolean(record.complete, "complete"),
    dropped_event_count: nonNegativeInteger(
      record.dropped_event_count,
      "dropped_event_count",
    ),
    totals: primitiveRecord(record.totals, "totals"),
    datasets: {
      throughput: metricPoints(datasets.throughput, "throughput"),
      optimization: metricPoints(
        datasets.optimization,
        "optimization",
      ),
      ppo_timing: metricPoints(datasets.ppo_timing, "ppo_timing"),
      rollout: metricPoints(datasets.rollout, "rollout"),
      rewards: metricPoints(datasets.rewards, "rewards"),
      inference: metricPoints(datasets.inference, "inference"),
      processes: metricPoints(datasets.processes, "processes"),
    },
  };
}

function metricPoints(
  value: unknown,
  label: string,
): readonly MetricPoint[] {
  return requiredArray(value, label).map((item) => {
    const record = requiredRecord(item, `${label} metric point`);
    rejectUnknownKeys(
      record,
      [
        "sequence",
        "update",
        "elapsed_seconds",
        "recorded_at_ms",
        "values",
      ],
      `${label} metric point`,
    );
    return {
      sequence: positiveInteger(record.sequence, "sequence"),
      update: nullableNonNegativeInteger(record.update, "update"),
      elapsed_seconds: nonNegativeNumber(
        record.elapsed_seconds,
        "elapsed_seconds",
      ),
      recorded_at_ms: nonNegativeInteger(
        record.recorded_at_ms,
        "recorded_at_ms",
      ),
      values: primitiveRecord(record.values, "values"),
    };
  });
}
