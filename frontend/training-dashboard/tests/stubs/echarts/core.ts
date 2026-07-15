export class FakeChart {
  #dom: Element;

  constructor(dom: Element) {
    this.#dom = dom;
  }

  getDom(): Element {
    return this.#dom;
  }

  setOption(): void {
    // No-op: unit tests only need chart lifecycle calls.
  }

  clear(): void {
    // No-op.
  }

  resize(): void {
    // No-op.
  }
}

export function init(
  dom: Element,
  _theme?: unknown,
  _options?: unknown,
): FakeChart {
  return new FakeChart(dom);
}

export function use(_components?: readonly unknown[]): void {
  // no-op
}

export {};
