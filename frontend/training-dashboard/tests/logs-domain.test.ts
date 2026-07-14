import { isAtLogBottom } from "../logs-domain.ts";

Deno.test("auto-follow is true only at the log viewport bottom", () => {
  if (!isAtLogBottom(900, 100, 1000)) {
    throw new Error("Exact bottom must follow");
  }
  if (!isAtLogBottom(893, 100, 1000)) {
    throw new Error("Small layout rounding must remain at bottom");
  }
  if (isAtLogBottom(850, 100, 1000)) {
    throw new Error("Reading older rows must pause DOM updates");
  }
});
