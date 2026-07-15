export function assert(
  condition: unknown,
  message = "Assertion failed",
): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

export function assertEquals<T>(
  left: T,
  right: T,
  message = "Values are not equal",
): void {
  if (!Object.is(left, right)) {
    throw new Error(`${message}: ${String(left)} !== ${String(right)}`);
  }
}

export function assertLess(
  left: number,
  right: number,
  message = "Left value is not less than right value",
): void {
  if (!(left < right)) {
    throw new Error(`${message}: ${left} >= ${right}`);
  }
}
