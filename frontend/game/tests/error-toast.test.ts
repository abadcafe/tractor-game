import {
  assertEquals,
  assertNotEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { showErrorToast } from "../ui/error-toast.ts";

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
