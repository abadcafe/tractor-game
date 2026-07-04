"""Static dashboard generation for training metric files."""

from __future__ import annotations

from pathlib import Path

from server import result as _result

DASHBOARD_FILENAME = "index.html"


def write_dashboard(
    run_dir: Path, *, title: str = "Training"
) -> _result.Ok[Path] | _result.Rejected:
    """Write a small dashboard that reads metrics.jsonl."""
    path = run_dir / DASHBOARD_FILENAME
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_dashboard_html(title=title), encoding="utf-8"
        )
    except OSError:
        return _result.Rejected(
            reason=f"dashboard write failed: {path}"
        )
    return _result.Ok(value=path)


def render_dashboard_html(*, title: str) -> str:
    """Return standalone dashboard HTML."""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }}
    .metric {{
      border: 1px solid #ddd;
      padding: 12px;
      border-radius: 6px;
    }}
    .label {{ color: #666; font-size: 12px; }}
    .value {{ font-size: 24px; }}
    pre {{ background: #f6f6f6; padding: 12px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div id="grid" class="grid"></div>
  <h2>latest</h2>
  <pre id="latest">waiting for metrics.jsonl</pre>
  <script>
    async function loadMetrics() {{
      const response = await fetch('metrics.jsonl?ts=' + Date.now());
      if (!response.ok) return;
      const text = await response.text();
      const lines = text.trim().split('\\n').filter(Boolean);
      if (lines.length === 0) return;
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
      if (records.length === 0) return;
      const latest = records[records.length - 1];
      document.getElementById('latest').textContent =
        JSON.stringify(latest, null, 2);
      const keys = [
        'total_games', 'total_updates', 'process_games_per_second',
        'last_round_decisions_per_second', 'last_team0_reward',
        'last_team1_reward', 'last_generated_action_count',
        'last_accepted_action_count', 'last_decision_count',
        'last_average_action_choices',
        'policy_loss', 'value_loss', 'entropy', 'approx_kl',
        'clip_fraction'
      ];
      document.getElementById('grid').innerHTML = keys.map((key) => `
        <div class="metric">
          <div class="label">${{key}}</div>
          <div class="value">${{latest[key] ?? ''}}</div>
        </div>
      `).join('');
    }}
    setInterval(loadMetrics, 2000);
    loadMetrics();
  </script>
</body>
</html>
"""
