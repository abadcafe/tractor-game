import { recordValue } from "../../browser/json.ts";
import { parseMetrics } from "../contracts/training-metrics.ts";
import { parseLogPage } from "../contracts/training-events.ts";

const FIXTURE_PATH =
  "frontend/training-dashboard/tests/fixtures/generated-contract.json";

Deno.test("frontend accepts the exact backend-generated contracts", async () => {
  const value: unknown = JSON.parse(
    await Deno.readTextFile(FIXTURE_PATH),
  );
  const fixture = recordValue(value);
  if (fixture === null) throw new Error("Invalid generated fixture");

  const logs = parseLogPage(fixture.training_log_page);
  const metrics = parseMetrics(fixture.training_metrics);

  if (
    logs.events.length !== 2 ||
    logs.events[0]?.event.schema_version !== 5 ||
    logs.events[1]?.event.error !== "checkpoint rejected"
  ) {
    throw new Error("Generated event contract was not preserved");
  }
  if (
    metrics.schema_version !== 6 ||
    metrics.datasets.optimization.length !== 1
  ) {
    throw new Error("Generated metrics contract was not preserved");
  }
});
