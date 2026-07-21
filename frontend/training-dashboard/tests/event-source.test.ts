import { EventStreamConnection } from "../event-source.ts";
import { assertEquals } from "./stubs/assert/mod.ts";

class FakeEventSource extends EventTarget {
  static readonly instances: FakeEventSource[] = [];

  readonly url: string;
  closeCount = 0;

  constructor(url: string | URL) {
    super();
    this.url = String(url);
    FakeEventSource.instances.push(this);
  }

  close(): void {
    this.closeCount += 1;
  }

  emit(
    name: string,
    data: string,
    lastEventId = "",
  ): void {
    this.dispatchEvent(
      new MessageEvent(name, { data, lastEventId }),
    );
  }
}

Deno.test("event stream leaves network reconnection to EventSource", () => {
  withFakeEventSource(() => {
    const connections: boolean[] = [];
    const errors: string[] = [];
    const stream = new EventStreamConnection({
      onConnectionChange: (connected) => connections.push(connected),
      onError: (message) => errors.push(message),
    });

    stream.connect("/api/training/events/process", []);
    const source = onlySource();
    source.dispatchEvent(new Event("open"));
    source.dispatchEvent(new Event("error"));

    assertEquals(connections.join(","), "true,false");
    assertEquals(errors.length, 0);
    assertEquals(source.closeCount, 0);
    assertEquals(FakeEventSource.instances.length, 1);
  });
});

Deno.test("event stream closes on terminal rejection", () => {
  withFakeEventSource(() => {
    const errors: string[] = [];
    const stream = new EventStreamConnection({
      onError: (message) => errors.push(message),
    });

    stream.connect("/events", []);
    const source = onlySource();
    source.emit(
      "rejected",
      JSON.stringify({ error: "run is invalid" }),
    );
    source.dispatchEvent(new Event("error"));

    assertEquals(errors.length, 1);
    assertEquals(errors[0], "run is invalid");
    assertEquals(source.closeCount, 1);
  });
});

Deno.test("event stream closes when a named event is malformed", () => {
  withFakeEventSource(() => {
    const errors: string[] = [];
    const values: unknown[] = [];
    const stream = new EventStreamConnection({
      onError: (message) => errors.push(message),
    });

    stream.connect("/events", [{
      name: "metrics",
      receive: (value) => values.push(value),
    }]);
    const source = onlySource();
    source.emit("metrics", "not-json");

    assertEquals(values.length, 0);
    assertEquals(errors.length, 1);
    assertEquals(source.closeCount, 1);
  });
});

Deno.test("event stream disconnect closes the active source", () => {
  withFakeEventSource(() => {
    const stream = new EventStreamConnection({ onError: () => {} });

    stream.connect("/events", []);
    const source = onlySource();
    stream.disconnect();
    stream.disconnect();

    assertEquals(source.closeCount, 1);
  });
});

Deno.test("event stream ignores events from a replaced source", () => {
  withFakeEventSource(() => {
    const received: unknown[] = [];
    const stream = new EventStreamConnection({ onError: () => {} });
    const listeners = [{
      name: "process",
      receive: (value: unknown) => received.push(value),
    }];

    stream.connect("/first", listeners);
    const first = onlySource();
    stream.connect("/second", listeners);
    const second = FakeEventSource.instances[1];
    if (second === undefined) throw new Error("Missing second source");
    first.emit("process", JSON.stringify({ process: null }));
    second.emit("process", JSON.stringify({ process: null }));

    assertEquals(first.closeCount, 1);
    assertEquals(received.length, 1);
    assertEquals(JSON.stringify(received[0]), '{"process":null}');
  });
});

function onlySource(): FakeEventSource {
  const source = FakeEventSource.instances.at(-1);
  if (source === undefined) throw new Error("Missing EventSource");
  return source;
}

function withFakeEventSource(run: () => void): void {
  const original = Object.getOwnPropertyDescriptor(
    globalThis,
    "EventSource",
  );
  FakeEventSource.instances.length = 0;
  Object.defineProperty(globalThis, "EventSource", {
    configurable: true,
    value: FakeEventSource,
  });
  try {
    run();
  } finally {
    if (original === undefined) {
      Reflect.deleteProperty(globalThis, "EventSource");
    } else {
      Object.defineProperty(globalThis, "EventSource", original);
    }
  }
}
