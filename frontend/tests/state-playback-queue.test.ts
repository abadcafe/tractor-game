import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { StatePlaybackQueue } from "../engine/state-playback-queue.ts";

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

Deno.test("StatePlaybackQueue renders messages in order at the configured cadence", async () => {
  const rendered: number[] = [];
  const times: number[] = [];
  const queue = new StatePlaybackQueue<number>((message) => {
    rendered.push(message);
    times.push(performance.now());
  }, { minFrameMs: 20 });

  queue.enqueue(1);
  queue.enqueue(2);
  queue.enqueue(3);

  await sleep(70);

  assertEquals(rendered, [1, 2, 3]);
  assertEquals(times[1] - times[0] >= 15, true);
  assertEquals(times[2] - times[1] >= 15, true);
});

Deno.test("StatePlaybackQueue reports caught-up state while messages are buffered", async () => {
  const caughtUpChanges: boolean[] = [];
  const queue = new StatePlaybackQueue<number>(() => {}, {
    minFrameMs: 20,
    onCaughtUpChange(caughtUp) {
      caughtUpChanges.push(caughtUp);
    },
  });

  queue.enqueue(1);
  queue.enqueue(2);

  assertEquals(queue.isCaughtUp(), false);
  await sleep(50);
  assertEquals(queue.isCaughtUp(), true);
  assertEquals(caughtUpChanges.includes(false), true);
  assertEquals(caughtUpChanges[caughtUpChanges.length - 1], true);
});

Deno.test("StatePlaybackQueue can choose cadence from the next message", async () => {
  const rendered: number[] = [];
  const times: number[] = [];
  const queue = new StatePlaybackQueue<number>((message) => {
    rendered.push(message);
    times.push(performance.now());
  }, {
    minFrameMsForMessage(message) {
      return message === 2 ? 45 : 15;
    },
  });

  queue.enqueue(1);
  queue.enqueue(2);
  queue.enqueue(3);

  await sleep(85);

  assertEquals(rendered, [1, 2, 3]);
  assertEquals(times[1] - times[0] >= 40, true);
  assertEquals(times[2] - times[1] >= 10, true);
});
