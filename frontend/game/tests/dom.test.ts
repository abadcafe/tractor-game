import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { DOMParser } from "jsr:@b-fuze/deno-dom@0.1.56";
import { el } from "../ui/dom.ts";

// Set up a global document for testing
const doc = new DOMParser().parseFromString(
  `<html><body><div id="root"><span class="item">a</span><span class="item">b</span></div></body></html>`,
  "text/html",
);
// @ts-ignore test setup
globalThis.document = doc;

Deno.test("test_el_creates_element", () => {
  const div = el("div");
  assertEquals(div.tagName, "DIV");
});

Deno.test("test_el_with_attrs", () => {
  const div = el("div", { id: "test", class: "foo bar" });
  assertEquals(div.id, "test");
  assertEquals(div.className, "foo bar");
});

Deno.test("test_el_with_text_child", () => {
  const div = el("div", {}, "hello");
  assertEquals(div.textContent, "hello");
});

Deno.test("test_el_with_element_child", () => {
  const child = el("span", {}, "inner");
  const div = el("div", {}, child);
  assertEquals(div.querySelector("span")!.textContent, "inner");
});

Deno.test("test_el_nested", () => {
  const inner = el("span", {}, "text");
  const outer = el("div", { class: "outer" }, inner);
  assertEquals(outer.className, "outer");
  assertEquals(outer.querySelector("span")!.textContent, "text");
});
