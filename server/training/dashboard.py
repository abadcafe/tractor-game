"""Static dashboard generation for training metric files."""

from __future__ import annotations

import html
import math
from pathlib import Path

from server import result as _result

DASHBOARD_FILENAME = "index.html"


def write_dashboard(
    run_dir: Path,
    *,
    title: str = "Training",
    telemetry_interval_seconds: float = 1.0,
) -> _result.Ok[Path] | _result.Rejected:
    """Write a small dashboard that reads metrics.jsonl."""
    assert math.isfinite(telemetry_interval_seconds)
    assert telemetry_interval_seconds > 0.0
    path = run_dir / DASHBOARD_FILENAME
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_dashboard_html(
                title=title,
                telemetry_interval_seconds=telemetry_interval_seconds,
            ),
            encoding="utf-8",
        )
    except OSError:
        return _result.Rejected(
            reason=f"dashboard write failed: {path}"
        )
    return _result.Ok(value=path)


def render_dashboard_html(
    *, title: str, telemetry_interval_seconds: float = 1.0
) -> str:
    """Return standalone dashboard HTML."""
    assert math.isfinite(telemetry_interval_seconds)
    assert telemetry_interval_seconds > 0.0
    escaped_title = html.escape(title)
    poll_interval_ms = max(
        1, int(round(telemetry_interval_seconds * 1000))
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --border: #d7dce2;
      --muted: #657080;
      --text: #1b2430;
      --panel: #ffffff;
      --track: #e7eaee;
      --good: #2f7d59;
      --warn: #9a6a1f;
      --accent: #2b6cb0;
      --stage: #6f42c1;
    }}
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      margin: 0;
      color: var(--text);
      background: #f7f8fa;
    }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 20px; }}
    h1 {{ font-size: 24px; margin: 0 0 16px; }}
    h2 {{ font-size: 16px; margin: 20px 0 10px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--border);
      padding: 12px;
      border-radius: 6px;
    }}
    .label {{ color: var(--muted); font-size: 12px; }}
    .value {{
      font-size: 22px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
    }}
    .stage-row {{
      display: grid;
      grid-template-columns: 150px minmax(160px, 1fr) 110px 110px;
      align-items: center;
      gap: 10px;
      padding: 8px 0;
      border-bottom: 1px solid var(--border);
    }}
    .stage-row:last-child {{ border-bottom: 0; }}
    .stage-label {{ font-weight: 600; overflow-wrap: anywhere; }}
    .stage-text {{ color: var(--muted); }}
    .bar {{
      height: 10px;
      background: var(--track);
      border-radius: 999px;
      overflow: hidden;
    }}
    .fill {{
      height: 100%;
      width: 0%;
      background: var(--accent);
      transition: width 180ms linear;
    }}
    .fill.rollout {{ background: var(--stage); }}
    .fill.inference {{ background: var(--accent); }}
    .fill.update {{ background: var(--warn); }}
    .fill.checkpoint {{ background: var(--good); }}
    canvas {{
      display: block;
      width: 100%;
      height: 240px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
    }}
    pre {{
      background: #111827;
      color: #e5e7eb;
      padding: 12px;
      overflow: auto;
      border-radius: 6px;
      max-height: 280px;
    }}
    @media (max-width: 760px) {{
      main {{ padding: 12px; }}
      .stage-row {{
        grid-template-columns: minmax(120px, 1fr);
        gap: 6px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>{escaped_title}</h1>
    <div id="grid" class="grid"></div>
    <h2>throughput</h2>
    <canvas id="throughput-chart" width="960" height="240"></canvas>
    <h2>processes</h2>
    <div id="processes" class="panel"></div>
    <h2>latest</h2>
    <pre id="latest">waiting for metrics.jsonl</pre>
  </main>
  <script>
    const dashboardPollMs = {poll_interval_ms};
    const metricKeys = [
      'total_games', 'total_updates', 'process_games_per_second',
      'last_round_decisions_per_second', 'last_team0_reward',
      'last_team1_reward', 'last_generated_action_count',
      'last_accepted_action_count', 'last_decision_count',
      'last_average_action_choices',
      'policy_loss', 'value_loss', 'entropy', 'approx_kl',
      'clip_fraction', 'ppo_update_seconds', 'ppo_backward_seconds',
      'ppo_optimizer_step_seconds',
      'model_rank_inference_batch_size', 'inference_frame_bytes',
      'inference_transport_wait_seconds',
      'model_rank_inference_seconds'
    ];

    async function fetchJsonl(path) {{
      const response = await fetch(path + '?ts=' + Date.now());
      if (!response.ok) return [];
      const text = await response.text();
      const lines = text.trim().split('\\n').filter(Boolean);
      const records = [];
      for (const line of lines) {{
        try {{
          const parsed = JSON.parse(line);
          if (
            parsed &&
            typeof parsed === 'object' &&
            !Array.isArray(parsed)
          ) {{
            records.push(parsed);
          }}
        }} catch (error) {{
          continue;
        }}
      }}
      return records;
    }}

    function numberText(value) {{
      if (typeof value !== 'number') return value ?? '';
      if (!Number.isFinite(value)) return '';
      if (Math.abs(value) >= 100) return value.toFixed(2);
      if (Math.abs(value) >= 1) return value.toFixed(4);
      return value.toPrecision(4);
    }}

    function renderMetrics(records) {{
      if (records.length === 0) return;
      const latest = records[records.length - 1];
      document.getElementById('latest').textContent =
        JSON.stringify(latest, null, 2);
      const grid = document.getElementById('grid');
      grid.replaceChildren();
      for (const key of metricKeys) {{
        const metric = document.createElement('div');
        metric.className = 'metric';
        const label = document.createElement('div');
        label.className = 'label';
        label.textContent = key;
        const value = document.createElement('div');
        value.className = 'value';
        value.textContent = numberText(latest[key]);
        metric.append(label, value);
        grid.append(metric);
      }}
      drawChart(records);
    }}

    function renderTelemetry(records) {{
      const container = document.getElementById('processes');
      container.replaceChildren();
      if (records.length === 0) {{
        const row = document.createElement('div');
        row.className = 'stage-text';
        row.textContent = 'waiting for telemetry.jsonl';
        container.append(row);
        return;
      }}
      const latestByProcess = new Map();
      for (const record of records) {{
        if (typeof record.process_label !== 'string') continue;
        latestByProcess.set(record.process_label, record);
      }}
      const ordered = [...latestByProcess.entries()].sort(
        (left, right) => left[0].localeCompare(right[0])
      );
      for (const [label, record] of ordered) {{
        const row = document.createElement('div');
        row.className = 'stage-row';
        const name = document.createElement('div');
        name.className = 'stage-label';
        name.textContent = label;
        const bar = document.createElement('div');
        bar.className = 'bar';
        const fill = document.createElement('div');
        fill.className = 'fill ' + String(record.stage ?? '');
        const numerator = Number(record.progress_numerator ?? 0);
        const denominator = Number(record.progress_denominator ?? 0);
        const percent = denominator <= 0
          ? 0
          : Math.max(0, Math.min(100, (numerator / denominator) * 100));
        fill.style.width = percent.toFixed(1) + '%';
        bar.append(fill);
        const stage = document.createElement('div');
        stage.className = 'stage-text';
        stage.textContent = String(record.stage ?? '');
        const counters = document.createElement('div');
        counters.className = 'stage-text';
        counters.textContent =
          String(record.total_rounds ?? 0) + ' / ' +
          String(record.total_updates ?? 0);
        row.append(name, bar, stage, counters);
        container.append(row);
      }}
    }}

    function drawChart(records) {{
      const canvas = document.getElementById('throughput-chart');
      const context = canvas.getContext('2d');
      if (!context) return;
      context.clearRect(0, 0, canvas.width, canvas.height);
      const points = records.filter(record =>
        typeof record.total_games === 'number' &&
        typeof record.process_games_per_second === 'number'
      );
      if (points.length < 2) return;
      const maxGame = Math.max(
        ...points.map(point => point.total_games),
        1
      );
      const maxY = Math.max(
        ...points.map(point => point.process_games_per_second),
        ...points.map(point =>
          Number(point.last_round_decisions_per_second ?? 0)
        ),
        1
      );
      drawSeries(context, points, maxGame, maxY,
        'process_games_per_second', '#2b6cb0');
      drawSeries(context, points, maxGame, maxY,
        'last_round_decisions_per_second', '#2f7d59');
    }}

    function drawSeries(context, points, maxGame, maxY, key, color) {{
      context.beginPath();
      context.strokeStyle = color;
      context.lineWidth = 2;
      points.forEach((point, index) => {{
        const x = 36 + (point.total_games / maxGame) * (960 - 56);
        const value = Number(point[key] ?? 0);
        const y = 210 - (value / maxY) * 180;
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }});
      context.stroke();
    }}

    async function loadDashboard() {{
      const metrics = await fetchJsonl('metrics.jsonl');
      const telemetry = await fetchJsonl('telemetry.jsonl');
      renderMetrics(metrics);
      renderTelemetry(telemetry);
    }}
    setInterval(loadDashboard, dashboardPollMs);
    loadDashboard();
  </script>
</body>
</html>
"""
