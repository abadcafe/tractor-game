import { assertEquals, assertNotEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { showErrorToast, setToastDuration, TOAST_DURATION_MS } from "../ui/error-toast.ts";

const doc = new DOMParser().parseFromString(
  `<html><body><div id="app"></div></body></html>`,
  "text/html",
);
// @ts-ignore test setup
globalThis.document = doc;

Deno.test("test_showErrorToast_creates_element", () => {
  const container = doc!.querySelector("#app")!;
  // Clear any existing toasts
  const existing = container.querySelectorAll(".error-toast");
  existing.forEach((e) => e.remove());

  showErrorToast("测试错误");
  const toast = container.querySelector(".error-toast");
  assertNotEquals(toast, null);
});

Deno.test("test_showErrorToast_shows_message", () => {
  const container = doc!.querySelector("#app")!;
  const existing = container.querySelectorAll(".error-toast");
  existing.forEach((e) => e.remove());

  showErrorToast("无效的出牌");
  const toast = container.querySelector(".error-toast");
  assertNotEquals(toast, null);
  assertEquals(toast!.textContent, "无效的出牌");
});

Deno.test("test_showErrorToast_auto_removes", async () => {
  const container = doc!.querySelector("#app")!;
  const existing = container.querySelectorAll(".error-toast");
  existing.forEach((e) => e.remove());

  // Use a very short duration so the test completes quickly
  const originalDuration = TOAST_DURATION_MS;
  setToastDuration(10);

  showErrorToast("临时错误");
  assertNotEquals(container.querySelector(".error-toast"), null);

  // Wait for the toast to auto-remove
  await new Promise((resolve) => setTimeout(resolve, 50));

  assertEquals(container.querySelector(".error-toast"), null);

  // Restore original duration
  setToastDuration(originalDuration);
});
