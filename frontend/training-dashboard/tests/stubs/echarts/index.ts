export interface LineSeriesOption {
  // Placeholder for ECharts line series options used in dashboard typings.
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
