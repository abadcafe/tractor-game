export interface LineSeriesOption {
  readonly name?: string;
  readonly type?: "line";
  readonly showSymbol?: boolean;
  readonly symbolSize?: number;
  readonly sampling?: "lttb";
  readonly lineStyle?: { readonly width: number };
  readonly emphasis?: { readonly focus: "series" };
  readonly data?: readonly (readonly [
    number | null,
    number | null,
  ])[];
}

export interface EChartsOption {
  // Placeholder shape for dashboard chart options.
  [key: string]: unknown;
}

export interface ECharts {
  readonly getDom: () => Element;
  readonly setOption: (
    _options: EChartsOption,
    _replace?: boolean,
  ) => void;
  readonly clear: () => void;
  readonly resize: () => void;
}
