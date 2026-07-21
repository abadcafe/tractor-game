import { BarChart, LineChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from "echarts/components";
import { init, use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import type { ECharts, EChartsOption, LineSeriesOption } from "echarts";

import { recordValue } from "../browser/json.ts";
import type { MetricPoint, TrainingMetrics } from "./types.ts";

use([
  LineChart,
  BarChart,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  DataZoomComponent,
  CanvasRenderer,
]);

export type MetricAxis = "update" | "elapsed";

interface SeriesSpec {
  readonly key: string;
  readonly label: string;
  readonly color: string;
}

interface ChartSpec {
  readonly elementId: string;
  readonly dataset: keyof TrainingMetrics["datasets"];
  readonly unit: string;
  readonly series: readonly SeriesSpec[];
}

const SPECS: readonly ChartSpec[] = [
  {
    elementId: "chart-throughput",
    dataset: "throughput",
    unit: "per second",
    series: [
      {
        key: "samples_per_second",
        label: "Samples/s",
        color: "#1769aa",
      },
      { key: "rounds_per_second", label: "Rounds/s", color: "#138a72" },
      {
        key: "decisions_per_second",
        label: "Decisions/s",
        color: "#c26718",
      },
    ],
  },
  {
    elementId: "chart-loss",
    dataset: "optimization",
    unit: "loss",
    series: [
      { key: "policy_loss", label: "Policy loss", color: "#1769aa" },
      { key: "value_loss", label: "Value loss", color: "#c26718" },
    ],
  },
  {
    elementId: "chart-policy",
    dataset: "optimization",
    unit: "value",
    series: [
      { key: "entropy", label: "Entropy", color: "#138a72" },
      { key: "approx_kl", label: "Approx KL", color: "#b23a48" },
      {
        key: "clip_fraction",
        label: "Clip fraction",
        color: "#7253a6",
      },
    ],
  },
  {
    elementId: "chart-ppo-timing",
    dataset: "ppo_timing",
    unit: "seconds",
    series: [
      {
        key: "update_seconds",
        label: "Total update",
        color: "#20262d",
      },
      {
        key: "ppo_observation_encode_seconds",
        label: "Encode",
        color: "#1769aa",
      },
      {
        key: "ppo_action_decode_seconds",
        label: "Decode",
        color: "#7253a6",
      },
      {
        key: "ppo_backward_seconds",
        label: "Backward",
        color: "#b23a48",
      },
      {
        key: "ppo_optimizer_step_seconds",
        label: "Optimizer",
        color: "#138a72",
      },
    ],
  },
  {
    elementId: "chart-rollout",
    dataset: "rollout",
    unit: "count",
    series: [
      { key: "sample_count", label: "Samples", color: "#1769aa" },
      { key: "round_count", label: "Rounds", color: "#138a72" },
      { key: "decision_count", label: "Decisions", color: "#c26718" },
      {
        key: "dropped_sample_count",
        label: "Dropped",
        color: "#b23a48",
      },
    ],
  },
  {
    elementId: "chart-rewards",
    dataset: "rewards",
    unit: "reward",
    series: [
      { key: "team0_reward", label: "Team 0", color: "#1769aa" },
      { key: "team1_reward", label: "Team 1", color: "#c26718" },
    ],
  },
  {
    elementId: "chart-inference",
    dataset: "inference",
    unit: "seconds / ratio",
    series: [
      { key: "fill_ratio", label: "Fill ratio", color: "#138a72" },
      {
        key: "recv_seconds_avg",
        label: "Receive avg",
        color: "#1769aa",
      },
      { key: "h2d_seconds_avg", label: "H2D avg", color: "#c26718" },
      {
        key: "decode_seconds_avg",
        label: "Decode avg",
        color: "#7253a6",
      },
      {
        key: "inference_seconds_avg",
        label: "Model avg",
        color: "#b23a48",
      },
      {
        key: "inference_seconds_p95",
        label: "Model P95",
        color: "#20262d",
      },
    ],
  },
  {
    elementId: "chart-processes",
    dataset: "processes",
    unit: "count / seconds",
    series: [
      { key: "completed_rounds", label: "Rounds", color: "#138a72" },
      { key: "decision_count", label: "Decisions", color: "#1769aa" },
      {
        key: "policy_wait_seconds",
        label: "Policy wait",
        color: "#b23a48",
      },
    ],
  },
];

export class DashboardCharts {
  private readonly charts = new Map<string, ECharts>();
  private metrics: TrainingMetrics | null = null;
  private axis: MetricAxis = "update";

  constructor() {
    for (const spec of SPECS) {
      const target = document.getElementById(spec.elementId);
      if (!(target instanceof HTMLDivElement)) {
        throw new Error(`Missing chart: ${spec.elementId}`);
      }
      this.charts.set(
        spec.elementId,
        init(target, undefined, {
          renderer: "canvas",
        }),
      );
    }
    const observer = new ResizeObserver(() => this.resize());
    for (const chart of this.charts.values()) {
      observer.observe(chart.getDom());
    }
  }

  setData(metrics: TrainingMetrics, axis: MetricAxis): void {
    this.metrics = metrics;
    this.axis = axis;
    for (const spec of SPECS) {
      const chart = this.charts.get(spec.elementId);
      if (chart === undefined) {
        throw new Error("Chart was not initialized");
      }
      chart.setOption(
        spec.dataset === "processes"
          ? processOption(metrics.datasets.processes)
          : option(spec, metrics.datasets[spec.dataset], axis),
        true,
      );
    }
  }

  resize(): void {
    for (const chart of this.charts.values()) chart.resize();
  }

  clear(): void {
    this.metrics = null;
    for (const chart of this.charts.values()) chart.clear();
  }
}

function processOption(points: readonly MetricPoint[]): EChartsOption {
  const workers = points.map((point) =>
    `Worker ${numericValue(point, "worker_index") ?? "?"}`
  );
  const series = [
    { key: "completed_rounds", name: "Rounds", color: "#138a72" },
    { key: "decision_count", name: "Decisions", color: "#1769aa" },
    {
      key: "policy_wait_seconds",
      name: "Policy wait (s)",
      color: "#b23a48",
    },
  ];
  return {
    animation: false,
    color: series.map((item) => item.color),
    grid: { left: 58, right: 24, top: 48, bottom: 42 },
    legend: { top: 4 },
    tooltip: { trigger: "axis", confine: true },
    xAxis: { type: "category", data: workers },
    yAxis: {
      type: "value",
      scale: true,
      splitLine: { lineStyle: { color: "#e4e8ec" } },
    },
    series: series.map((item) => ({
      name: item.name,
      type: "bar",
      barMaxWidth: 42,
      data: points.map((point) => numericValue(point, item.key)),
    })),
  };
}

function option(
  spec: ChartSpec,
  points: readonly MetricPoint[],
  axis: MetricAxis,
): EChartsOption {
  return {
    animation: false,
    color: spec.series.map((item) => item.color),
    grid: { left: 58, right: 24, top: 48, bottom: 54 },
    legend: { top: 4, type: "scroll" },
    tooltip: {
      trigger: "axis",
      confine: true,
      formatter: metricTooltip(points, axis),
    },
    dataZoom: [
      { type: "inside", filterMode: "none" },
      { type: "slider", height: 18, bottom: 8, filterMode: "none" },
    ],
    xAxis: {
      type: "value",
      name: axis === "update" ? "Update event" : "Elapsed seconds",
      nameLocation: "middle",
      nameGap: 32,
      minInterval: axis === "update" ? 1 : undefined,
    },
    yAxis: {
      type: "value",
      name: spec.unit,
      scale: true,
      splitLine: { lineStyle: { color: "#e4e8ec" } },
    },
    series: buildSeries(spec.series, points, axis),
  };
}

function metricTooltip(
  points: readonly MetricPoint[],
  axis: MetricAxis,
): (params: unknown) => string {
  return (params) => {
    if (!Array.isArray(params) || params.length === 0) return "";
    const first = recordValue(params[0]);
    const axisValue = first?.axisValue;
    const point = points.find((item) =>
      (axis === "update" ? item.update : item.elapsed_seconds) ===
        axisValue
    );
    const lines = axis === "update"
      ? [`Update event: ${String(axisValue ?? "-")}`]
      : [`Elapsed seconds: ${String(axisValue ?? "-")}`];
    if (point !== undefined) {
      lines.push(
        `Training total_updates: ${
          String(point.values.total_updates ?? "-")
        }`,
        `Log sequence: ${point.sequence}`,
      );
    }
    for (const item of params) {
      const record = recordValue(item);
      const values = record?.value;
      const value = Array.isArray(values) ? values[1] : null;
      lines.push(
        `${String(record?.seriesName ?? "Series")}: ${
          String(value ?? "-")
        }`,
      );
    }
    return lines.join("<br>");
  };
}

function buildSeries(
  specs: readonly SeriesSpec[],
  points: readonly MetricPoint[],
  axis: MetricAxis,
): LineSeriesOption[] {
  return specs.map((spec) => ({
    name: spec.label,
    type: "line",
    showSymbol: points.length < 80,
    symbolSize: 5,
    sampling: "lttb",
    lineStyle: { width: 2 },
    emphasis: { focus: "series" },
    data: points.map((point) => [
      axis === "update" ? point.update : point.elapsed_seconds,
      numericValue(point, spec.key),
    ]),
  }));
}

function numericValue(point: MetricPoint, key: string): number | null {
  const value = point.values[key];
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : null;
}
