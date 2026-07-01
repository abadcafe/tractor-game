"""Static dashboard generation for training metric files."""

from __future__ import annotations

from pathlib import Path

DASHBOARD_FILENAME = "index.html"


def write_dashboard(run_dir: Path, *, title: str = "Training") -> Path:
    """Write a small dashboard that reads metrics.jsonl."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / DASHBOARD_FILENAME
    path.write_text(
        render_dashboard_html(title=title), encoding="utf-8"
    )
    return path


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
      const latest = JSON.parse(lines[lines.length - 1]);
      document.getElementById('latest').textContent =
        JSON.stringify(latest, null, 2);
      const keys = [
        'total_games', 'total_updates', 'games_per_second',
        'decisions_per_second', 'average_reward', 'average_level_delta',
        'invalid_action_count', 'legal_action_rate',
        'resample_count', 'forced_action_count',
        'average_action_choices'
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
